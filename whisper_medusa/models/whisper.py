from collections import Counter
from typing import Optional, Tuple, Union, Callable, List, TYPE_CHECKING, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import PreTrainedModel
from whisper_medusa.utils.config_and_args import MedusaConfig
from transformers import AutoConfig
import os
import warnings
import copy
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensLogitsProcessor,
)

from transformers.generation.utils import (
    GenerateOutput, 
    logger, 
    NEED_SETUP_CACHE_CLASSES_MAPPING, 
    GenerateNonBeamOutput, 
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from transformers.utils import ModelOutput
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.integrations import is_deepspeed_zero3_enabled
import whisper_medusa.models.medusa_utils  as medusa_utils
import torch.distributed as dist
import inspect
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer
from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer

@dataclass
class WhisperMedusaGenerationOutput:
    input_ids: torch.Tensor
    count_selected_heads: Dict[str, int]

# Copied from transformers.models.whisper.modeling_whisper.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class Whisper2MedusaHeadsConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.freeze_name2func = {
            "encoder": self._freeze_encoder,
            "encoder_attn": self._freeze_encoder_and_cross_attn,
            "decoder": self._freeze_decoder,
            "all_but_lm": self.freeze_all_but_lm_head,
            "all": self._freeze_all,
            "all_but_last": self._freeze_all_but_last,
        }
    def _freeze_lm_head(self):
        for p in self.proj_out.parameters():
            p.requires_grad = False

    def freeze_all_but_lm_head(self):
        self._freeze_encoder()
        self._freeze_decoder()

    def _freeze_all(self):
        # freeze all layers but except lm_head
        self._freeze_decoder()
        self._freeze_encoder()
        self._freeze_lm_head()

    def _freeze_all_but_last(self):
        # freeze all layers but except lm_head
        self._freeze_encoder()
        self._freeze_decoder(freeze_last_layer=False)
        self._freeze_lm_head()

    def _freeze_decoder(self, freeze_last_layer: bool = True):
        # Freeze all layers and the last one in medusa model
        decoder_layers = list(self.model.decoder.children())

        for layer in decoder_layers:
            if type(layer) == nn.ModuleList:
                for idx, sub_layer in enumerate(layer):
                    if not freeze_last_layer and idx == len(layer) - 1:
                        break
                    for p in sub_layer.parameters():
                        p.requires_grad = False
            else:
                for p in layer.parameters():
                    p.requires_grad = False

    def _freeze_cross_attn(self):
        for layer in self.model.decoder.layers:
            for p in layer.encoder_attn.parameters():
                p.requires_grad = False

    def _freeze_decoder(self):
        decoder_layers = list(self.model.decoder.children())

        # Freeze all layers except the last one
        for layer in decoder_layers[:-1]:
            for p in layer.parameters():
                p.requires_grad = False

    def _freeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def _freeze_encoder_and_cross_attn(self):
        self._freeze_cross_attn()
        self._freeze_encoder()

    def freeze_model_parts(self, parts_to_freeze):
        if parts_to_freeze is None:
            return

        if parts_to_freeze not in self.freeze_name2func:
            raise ValueError(
                f"parts_to_freeze {parts_to_freeze} is not supported, "
                f"select from {list(self.freeze_name2func.keys())}"
            )

        self.freeze_name2func[parts_to_freeze]()
    def medusa_forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return outputs
  
class MedusaResBlock(nn.Module):
    """
    A Medusa Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        input_size (int): The size of the hidden layers in the block.
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))
     
class WhisperMedusaModel(PreTrainedModel):
    
    def __init__(self, config):
        # super(WhisperMedusaModel, self).__init__()
        super().__init__(config)
  
        self.whisper_model = Whisper2MedusaHeadsConditionalGeneration.from_pretrained(config.whisper_model_name)
        if self.config.medusa_heads_type  == "whisper_block":
            self.medusa_block = WhisperDecoderLayer(self.whisper_model.config)
            self.medusa_block.load_state_dict(self.whisper_model.model.decoder.layers[-1].state_dict()) # load the last layer of the whisper model
            self.medusa_heads = nn.ModuleList()
            for _ in range(self.config.medusa_num_heads):
                head_list = []
                for _ in range(self.config.medusa_num_layers):
                    head_list.append(MedusaResBlock(self.config.d_model, self.config.medusa_hidden_size))
                self.medusa_heads.append(nn.Sequential(*head_list))

        self.whisper_model.freeze_name2func.update({"whisper": self._freeze_whisper})
        self.generation_config = medusa_utils.MedusaGenerationConfig.from_model_config(self.config) if self.can_generate() else None # update generation_config to contain medusa parameters

    def _freeze_whisper(self):
        self.whisper_model._freeze_all()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
            config=config,
        )

    def get_medusa_choice(self):
        return self.config.medusa_choices 

    def prepare_inputs_for_medusa_tree_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        decoder_position_ids = kwargs.get("decoder_position_ids", None)

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }

    def _update_model_kwargs_for_medusa_generation(
        self,
        outputs: ModelOutput,
        accepted_len: int,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self.whisper_model._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + accepted_len

        return model_kwargs

    def _update_medusa_outputs(
        self,
        outputs: ModelOutput,
        tree_outputs: ModelOutput,
        select_indices: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        if getattr(outputs, 'decoder_attentions', None) is not None:
            pass #TODO- check if code changes are needed
        if getattr(outputs, 'attentions', None) is not None:
            pass #TODO- check if code changes are needed
        if getattr(outputs, 'cross_attentions', None) is not None:
            pass #TODO- check if code changes are needed
        if getattr(outputs, 'hidden_states', None) is not None:
            pass #TODO- check if code changes are needed
        if getattr(outputs, 'decoder_hidden_states', None) is not None:
            pass #TODO- check if code changes are needed
        if getattr(outputs, 'past_key_values', None) is not None:
            # past_key_values contains decoder num_layers tuples, each with a tuple contains 4 tensors. 
            # The first and second are the attention key and value with shape (batch_size, num_heads, sequence_length, embed_size_per_head)
            # The third and fourth are the cross attention (batch_size, num_heads, encoder_sequence_length, embed_size_per_head)
            all_past_key_values_tuple = ()
            for i in range(len(outputs.past_key_values)): # loop over decoder layers
                layer_tree_past_key_values = tree_outputs.past_key_values[i]
                layer_orig_past_key_values = outputs.past_key_values[i]
                layer_past_key_values_tuple = ()
                for j in range(2): # loop over attention key and value
                    tree_past = layer_tree_past_key_values[j][:, :, select_indices, :]
                    tree_past = tree_past[:, :, :accept_length, :]
                    orig_past = layer_orig_past_key_values[j]
                    layer_past_key_values_tuple += (torch.cat([orig_past, tree_past], dim=2),)
                layer_past_key_values_tuple += (layer_orig_past_key_values[2], layer_orig_past_key_values[3])
                all_past_key_values_tuple += (layer_past_key_values_tuple,)
            tree_outputs.past_key_values = all_past_key_values_tuple
        if getattr(outputs, 'mems', None) is not None:
            pass #TODO- check if code changes are needed
        if getattr(outputs, 'past_buckets_states', None) is not None:
            pass #TODO- check if code changes are needed
        if getattr(outputs, "state", None) is not None:
            pass #TODO- check if code changes are needed


    def _medusa_greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        medusa_choices: Optional[list] = None,
        temperature: float = 0.0,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,

        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model._greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = self.whisper_model.validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        # Medusa initialization

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice() # replace with something from config

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = medusa_utils.generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices
        medusa_topk = medusa_choices[1:]

        medusa_utils.reset_medusa_mode(self) 
        self.medusa_mask = medusa_buffers["medusa_attn_mask"]
        new_token = 0
        accept_length_list = []
        with torch.inference_mode():
            while self.whisper_model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                # prepare model inputs
                model_inputs = self.whisper_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                # forward pass to get next token
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need
                orig_logits = outputs.logits[0]
                medusa_logits = outputs.logits[1:]
                candidates, tree_candidates = medusa_utils.generate_candidates(
                medusa_logits,
                orig_logits,
                medusa_topk,
                medusa_buffers["tree_indices"]
                )

                # Use tree attention to verify the candidates and get predictions
                process_logits, tree_outputs = medusa_utils.tree_decoding(
                    self,
                    tree_candidates,
                    outputs["past_key_values"],# assuming that past_key_values is return from the model
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                    output_attentions,
                    output_hidden_states,
                    model_kwargs,
                )
                logits_processor(input_ids, process_logits) # NOTE - The logic of logits_processor here only functions correctly when batch_size equals 1.
                
                # Evaluate the posterior of the candidates to select the accepted candidate prefix
                best_candidate, accept_length = medusa_utils.evaluate_posterior(
                    process_logits, candidates, temperature, posterior_threshold, posterior_alpha
                )

                accept_length_list.append(accept_length.item())

                # Update the input_ids and logits

                input_ids, next_token_logits, new_token, next_tokens = medusa_utils.update_inference_inputs(
                    self,
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    tree_outputs,
                    process_logits,
                    new_token,
                    eos_token_id,
                    pad_token_id,
                    unfinished_sequences,
                )
                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_logits,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (tree_outputs.decoder_attentions,) if self.config.is_encoder_decoder else (tree_outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (tree_outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (tree_outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (tree_outputs.hidden_states,)
                        )
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                model_kwargs = self._update_model_kwargs_for_medusa_generation(
                    tree_outputs,
                    accept_length.item()+1,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id_tensor is not None: 
                    unfinished_sequences = unfinished_sequences.mul(
                    next_tokens[:,-1].unsqueeze(1).tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))# NOTE - this only check that the last token is eof 
                
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0 or input_ids.shape[1] + self.config.medusa_num_heads >= self.generation_config.max_length

        if streamer is not None:
            streamer.end()
        # print("accept_length", accept_length_list)
        # print("total_proc_time", total_proc_time)
        # print("total_proc_time_list len ", len(total_proc_time_list))

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return WhisperMedusaGenerationOutput(
                input_ids=input_ids,
                count_selected_heads=dict(
                    sorted(
                        Counter(accept_length_list).items()
                    )
                ),
            )

    def _multi_heads_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self.whisper_model._validate_model_class()
        generation_config, model_kwargs = self.whisper_model._prepare_generation_config(generation_config, **kwargs) # TODO- CHECK THIS WITH MEDUSA GENERATION CONFIG
        self.whisper_model._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self.whisper_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.whisper_model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self.whisper_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self.whisper_model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # otherwise the total length [inputs-embeds-len + new-tokens-len] will go beyond indicated `max_length``
        elif (
            model_input_name == "inputs_embeds"
            and inputs_tensor.shape[:-1] != input_ids.shape
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]
            generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if generation_config.cache_implementation == "static":
                if model_kwargs.get("past_key_values", False) is not False:
                    raise ValueError(
                        "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
                    )
                cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
                if not callable(getattr(self, "_setup_cache", None)):
                    raise ValueError(
                        "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                        " Make sure it has a `_setup_cache` function."
                    )
                self.whisper_model._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

        self.whisper_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self.whisper_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self.whisper_model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            result = self._medusa_greedy_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                temperature=generation_config.temperature,
                posterior_threshold=generation_config.posterior_threshold,
                posterior_alpha=generation_config.posterior_alpha,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.BEAM_SEARCH:
            raise Exception("Beam search is not supported with medusa for now")
        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            raise Exception("Beam search is not supported with medusa for now")
        
        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if not callable(getattr(self, "_reset_cache", None)):
                raise ValueError(
                    "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
                    " Make sure this model implements a `_reset_cache` function."
                )
            self._reset_cache()

        return result

    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, is_shortform, num_beams):
        if generation_config.return_timestamps is True: 
            timestamp_processor = medusa_utils.MedusaWhisperTimeStampLogitsProcessor(generation_config, begin_index=begin_index)
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = medusa_utils.MedusaSuppressTokensLogitsProcessor(generation_config.suppress_tokens)
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = medusa_utils.MedusaSuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None and not is_shortform:
            no_speech_detector = medusa_utils.MedusaWhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )
            no_speech_detector.set_model(self)

        return logits_processor
    
    def freeze_model_parts(self, parts_to_freeze):
        self.whisper_model.freeze_model_parts(parts_to_freeze)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        disable_medusa: Optional[bool] = False,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        
        whisper_model_outputs = self.whisper_model.medusa_forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = whisper_model_outputs[0]
        medusa_logits = []
        

        if self.config.medusa_heads_type == "base_head":
            for i in range(len(self.medusa_heads)):
                head_out = self.medusa_heads[i](hidden_states)
                head_proj = self.whisper_model.proj_out(head_out)
                if i == 0:
                    base_logits = head_proj
                medusa_logits.append(head_proj)  
                if disable_medusa:
                    break
                      
        else:
            base_logits = self.whisper_model.proj_out(hidden_states)
            medusa_logits.append(base_logits)
            if not disable_medusa:
                if self.config.medusa_heads_type == "linear":
                    for i in range(len(self.medusa_heads)):
                        head_out = self.medusa_heads[i](hidden_states)
                        head_proj = self.whisper_model.proj_out(head_out)
                        medusa_logits.append(head_proj)
                elif self.config.medusa_heads_type == "lstm":
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    lstm_hidden_states = hidden_states.reshape(batch_size*seq_len, 1, hidden_dim)
                    hidden, cell = self.medusa_heads.init_hidden(lstm_hidden_states.shape[0], hidden_states.device)
                    for i in range(self.config.medusa_num_heads):
                        head_out, hidden, cell = self.medusa_heads(lstm_hidden_states, hidden, cell)
                        head_out = head_out.squeeze(1).reshape(batch_size, seq_len, hidden_dim)
                        head_proj = self.whisper_model.proj_out(head_out)
                        medusa_logits.append(head_proj)  
                elif self.config.medusa_heads_type == "whisper_block":
                    use_cache = use_cache if use_cache is not None else self.config.use_cache
                    past_key_value = past_key_values[-1] if past_key_values is not None else None 
                    medusa_block_decoder_outputs = self.medusa_block(
                        whisper_model_outputs.last_hidden_state,
                        attention_mask=attention_mask,
                        encoder_hidden_states=whisper_model_outputs.encoder_last_hidden_state,
                        layer_head_mask=(head_mask[-1] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(cross_attn_head_mask[-1] if cross_attn_head_mask is not None else None ),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                                )
                    if use_cache:
                        whisper_model_outputs.past_key_values += (medusa_block_decoder_outputs[3 if output_attentions else 1],)

                    if output_attentions:
                        whisper_model_outputs.decoder_attentions += (medusa_block_decoder_outputs[1],)

                        if whisper_model_outputs.encoder_last_hidden_state is not None:
                            whisper_model_outputs.cross_attention += (medusa_block_decoder_outputs[2],)

                    for i in range(len(self.medusa_heads)):
                        head_out = self.medusa_heads[i](medusa_block_decoder_outputs[0])
                        head_proj = self.whisper_model.proj_out(head_out)
                        medusa_logits.append(head_proj)
                else:
                    raise ValueError("Invalid medusa_heads_type")
            else:
                if self.config.medusa_heads_type == "whisper_block":
                    use_cache = use_cache if use_cache is not None else self.config.use_cache
                    past_key_value = past_key_values[-1] if past_key_values is not None else None 
                    medusa_block_decoder_outputs = self.medusa_block(
                        whisper_model_outputs.last_hidden_state,
                        attention_mask=attention_mask,
                        encoder_hidden_states=whisper_model_outputs.encoder_last_hidden_state,
                        layer_head_mask=(head_mask[-1] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(cross_attn_head_mask[-1] if cross_attn_head_mask is not None else None ),
                        past_key_value=past_key_value,
                        # past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                                )
                    if use_cache:
                        whisper_model_outputs.past_key_values += (medusa_block_decoder_outputs[3 if output_attentions else 1],)

                    if output_attentions:
                        whisper_model_outputs.decoder_attentions += (medusa_block_decoder_outputs[1],)

                        if whisper_model_outputs.encoder_last_hidden_state is not None:
                            whisper_model_outputs.cross_attention += (medusa_block_decoder_outputs[2],)

        stack_heads_logits = torch.stack(medusa_logits, dim=0)


        loss = None

        if not return_dict:
            output = (stack_heads_logits,) + whisper_model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=stack_heads_logits,
            past_key_values=whisper_model_outputs.past_key_values,
            decoder_hidden_states=whisper_model_outputs.decoder_hidden_states,
            decoder_attentions=whisper_model_outputs.decoder_attentions,
            cross_attentions=whisper_model_outputs.cross_attentions,
            encoder_last_hidden_state=whisper_model_outputs.encoder_last_hidden_state,
            encoder_hidden_states=whisper_model_outputs.encoder_hidden_states,
            encoder_attentions=whisper_model_outputs.encoder_attentions,
        )

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    ):
        # # 0. deprecate old inputs
        assert input_features.shape[0] == 1, "Only support batch size 1 for now!!"
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )
        # 1. copy generation config
        if generation_config is None:
            generation_config = copy.deepcopy(self.whisper_model.generation_config)
            generation_config = medusa_utils.MedusaGenerationConfig(**generation_config.to_dict())# TODO- load this from medusa config
            # generation_config = self.generation_config
        else:
            generation_config = copy.deepcopy(generation_config)
            

        # 2. set global generate variables
        input_stride = self.whisper_model.model.encoder.conv1.stride[0] * self.whisper_model.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        batch_size, total_input_frames = self.whisper_model._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        if is_shortform:
            # warn user of ignored inputs
            self.whisper_model._maybe_warn_unused_inputs(
                condition_on_prev_tokens=condition_on_prev_tokens,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                total_input_frames=total_input_frames,
            )

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        self.whisper_model._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            is_shortform=is_shortform,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        self.whisper_model._set_return_timestamps(
            return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
        )
        
        if isinstance(language, list):
            if is_multilingual is None:
                is_multilingual = False
            if len(set(language)) != 1:
                language = None
                is_multilingual = True or is_multilingual
            else:
                language = language[0]
                is_multilingual = False or is_multilingual
   
        is_multilingual = True
        self.whisper_model._set_language_and_task(
            language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
        )
        self.whisper_model._set_token_ids(generation_config=generation_config, config=self.config, kwargs=kwargs)
        self.whisper_model._set_num_frames(
            return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
        )
        self.whisper_model._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )
        self.whisper_model._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type=prompt_condition_type,
        )

        # pass self.config for backward compatibility
        init_tokens = self.whisper_model._retrieve_init_tokens(
            input_features,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self.whisper_model._check_decoder_input_ids(kwargs=kwargs)

        # 3. Retrieve logits processors
        begin_index = len(init_tokens)

        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            is_shortform=is_shortform,
            num_beams=kwargs.get("num_beams", 1),
        )

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            if temperature is not None:
                kwargs["temperature"] = temperature

            decoder_input_ids = kwargs.pop("decoder_input_ids", None)
            if decoder_input_ids is None:
                one_tensor = torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)

            if prompt_ids is not None:
                decoder_input_ids = torch.cat(
                    [prompt_ids[None].repeat(decoder_input_ids.shape[0], 1), decoder_input_ids], dim=-1
                )

            if kwargs.get("max_new_tokens", 0) + decoder_input_ids.shape[-1] > self.config.max_target_positions:
                max_new_tokens = kwargs.get("max_new_tokens", 0)
                raise ValueError(
                    f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
                    f"is {max_new_tokens}. Thus, the combined length of "
                    f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                    f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                    "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                    f"so that their combined length is less than {self.config.max_target_positions}."
                )
            
            outputs = self._multi_heads_generate(input_features,
                        generation_config=generation_config,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                        synced_gpus=synced_gpus,
                        decoder_input_ids=decoder_input_ids,
                        **kwargs,)

            if generation_config.return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=generation_config.num_frames
                )

            return outputs

        # 6. Else we're in longform mode which is more complex.
        # We need to chunk the audio input depending on when the model generates timestamp tokens

        # 6.1 Set and retrieve global longform generation variables
        self.whisper_model._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
        )

        timestamp_begin = generation_config.no_timestamps_token_id + 1
        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]
        batch_size = input_features.shape[0]

        max_frames, seek = self.whisper_model._retrieve_max_frames_and_seek(
            batch_size=batch_size, attention_mask=attention_mask, total_input_frames=total_input_frames
        )

        # 6.2 Preppare running variables, list for generation
        cur_bsz = batch_size
        current_segments = self.whisper_model._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=batch_size,
            generation_config=generation_config,
        )

        batch_idx_map = list(range(batch_size))
        do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(batch_size)]

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self.whisper_model._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.4 cut out next 30s segment from input features
            segment_input = self.whisper_model._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # 6.5 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )
            decoder_input_ids, kwargs = self.whisper_model._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                prompt_ids=prompt_ids,
                generation_config=generation_config,
                config=self.config,
                device=segment_input.device,
                suppress_tokens=suppress_tokens,
                kwargs=kwargs,
            )

            # 6.6 set max new tokens or max length
            kwargs = self.whisper_model._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
                kwargs=kwargs,
            )

            # 6.7 Set current `begin_index` for all logit processors
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

            # TODO-  change to multi heads!!!
            # 6.8 Run generate with fallback
            seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens = self.whisper_model.generate_with_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
                seek=seek,
                num_segment_frames=num_segment_frames,
                max_frames=max_frames,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                kwargs=kwargs,
            )

            # 6.9 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = batch_idx_map[i]

                if should_skip[i]:
                    seek[prev_i] += seek_num_frames[prev_i]
                    continue

                segments, segment_offset = self.whisper_model._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                    return_token_timestamps=return_token_timestamps,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        final_segments = (
            [x[1:] for x in current_segments]
            if (prompt_ids is not None and generation_config.prompt_condition_type == "first-segment")
            else current_segments
        )
        sequences = _pad_to_max_length(final_segments, generation_config.pad_token_id, padding="right")

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": final_segments}

        return sequences

    def generate_with_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        batch_idx_map,
        seek,
        num_segment_frames,
        max_frames,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        kwargs,
    ):
        # TODO -THIS FUNCTION WASN'T CHECKED YET FOR MEDUSA!
        # 6.6 Batch generate current chunk
        seek_sequence_list = [None for _ in range(cur_bsz)]
        seek_outputs_list = [None for _ in range(cur_bsz)]
        needs_fallback = [False for _ in range(cur_bsz)]
        should_skip = [False for _ in range(cur_bsz)]
        fallback_index_map = list(range(cur_bsz))

        if generation_config.no_speech_threshold is not None:
            self.whisper_model._setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs)

        for fallback_idx, temperature in enumerate(temperatures):
            generation_config.do_sample = temperature is not None and temperature > 0.0

            generation_config.temperature = temperature if generation_config.do_sample else 1.0
            generation_config.num_beams = kwargs.pop("num_beams", 1) if not generation_config.do_sample else 1

            
            seek_outputs = super().generate(
                segment_input,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )

            # post-process sequence tokens and outputs to be in list form
            seek_sequences, seek_outputs = self.whisper_model._postprocess_outputs(
                seek_outputs=seek_outputs,
                decoder_input_ids=decoder_input_ids,
                return_token_timestamps=return_token_timestamps,
                generation_config=generation_config,
            )

            # 6.7 Extract cut sequences from every sequence and check if fallback should be applied
            # Loop over each decoded audio individually as each decoding can be of a different length
            new_fallback_index_map = []
            new_segment_input = []
            new_decoder_input_ids = []
            new_decoder_attention_mask = []

            for i, seek_sequence in enumerate(seek_sequences):
                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                prev_i = batch_idx_map[fallback_index_map[i]]
                is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]

                # remove eos token id
                if is_not_final and seek_sequence[-1] == generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]
                    if return_token_timestamps:
                        seek_outputs[i]["token_timestamps"] = seek_outputs[i]["token_timestamps"][:-1]

                # remove all padding tokens
                if seek_sequence[-1] == generation_config.pad_token_id:
                    num_paddings = (seek_sequence == generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]
                    if return_token_timestamps:
                        seek_outputs[i]["token_timestamps"] = seek_outputs[i]["token_timestamps"][:-num_paddings]

                # check which sequences in batch need fallback & which should be skipped
                needs_fallback[i], should_skip[i] = self.whisper_model._need_fallback(
                    seek_sequence,
                    seek_outputs,
                    i,
                    logits_processor,
                    generation_config,
                    self.config.vocab_size,
                    temperature,
                )

                seek_sequence_list[fallback_index_map[i]] = seek_sequence
                seek_outputs_list[fallback_index_map[i]] = seek_outputs[i]
                is_low_temperature = temperature is None or temperature < 0.5
                do_condition_on_prev_tokens[fallback_index_map[i]] = (
                    generation_config.condition_on_prev_tokens and is_low_temperature
                )

                if needs_fallback[i]:
                    new_fallback_index_map.append(fallback_index_map[i])
                    new_segment_input.append(segment_input[i])
                    new_decoder_input_ids.append(decoder_input_ids[i])
                    if "decoder_attention_mask" in kwargs:
                        new_decoder_attention_mask.append(kwargs["decoder_attention_mask"][i])

            fallback_index_map = new_fallback_index_map

            # if no sequence needs to be run with temperature fallback, we're finished
            if len(fallback_index_map) == 0 or fallback_idx == len(temperatures) - 1:
                seek_sequences = seek_sequence_list
                seek_outputs = seek_outputs_list
                break

            # if we're still in the loop, make sure that decoder_input_ids and segment inputs are tensors
            decoder_input_ids = torch.stack(new_decoder_input_ids)
            segment_input = torch.stack(new_segment_input)
            if "decoder_attention_mask" in kwargs:
                kwargs["decoder_attention_mask"] = torch.stack(new_decoder_attention_mask)

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens

def get_model(args_i):

    if args_i.model_type == "medusa":
        if not os.path.exists(args_i.whisper_model_name):
            config = MedusaConfig(medusa_num_heads=args_i.medusa_num_heads, 
                                medusa_num_layers=args_i.medusa_num_layers, 
                                whisper_model_name=args_i.whisper_model_name,
                                medusa_hidden_size=args_i.medusa_hidden_size,
                                medusa_heads_type=args_i.medusa_heads_type,
                                medusa_choices=args_i.medusa_choices,
                                medusa_kl_loss=args_i.medusa_kl_loss,
                                medusa_kl_weight=args_i.medusa_kl_weight,
                                medusa_loss_on_original=args_i.medusa_loss_on_original)
            model = WhisperMedusaModel(config)
        else:
            model = WhisperMedusaModel.from_pretrained(args_i.whisper_model_name)
    else:
        raise NotImplementedError("Invalid model type")

    return model