import copy
import inspect
import os
import warnings
from collections import Counter
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import (AutoConfig, PreTrainedModel,
                          WhisperForConditionalGeneration)
from transformers.generation.configuration_utils import (GenerationConfig,
                                                         GenerationMode)
from transformers.generation.logits_process import (
    LogitsProcessorList, SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (NEED_SETUP_CACHE_CLASSES_MAPPING,
                                           GenerateDecoderOnlyOutput,
                                           GenerateEncoderDecoderOutput,
                                           GenerateNonBeamOutput,
                                           GenerateOutput, logger)
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
from transformers.models.whisper.generation_whisper import _get_attr_from_logit_processors, _pad_to_max_length
from transformers.utils import is_torchdynamo_compiling
import torch.nn.functional as F
from transformers.utils import ModelOutput

import whisper_medusa.models.medusa_utils as medusa_utils
from whisper_medusa.utils.config_and_args import MedusaConfig

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

from transformers.models.whisper.modeling_whisper import shift_tokens_right

from whisper_medusa.utils.losses import MedusaCrossEntropyLoss, MedusaKLDivLoss


class Whisper2MedusaHeadsConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.freeze_name2func = {
            "all_but_last": self._freeze_all_but_last,
            "whisper": self._freeze_all,
        }

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
        **kwargs,
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

    def _freeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False

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

    def _freeze_lm_head(self):
        for p in self.proj_out.parameters():
            p.requires_grad = False

    def _freeze_all_but_last(self):
        # freeze all layers but last decoder layer
        self._freeze_encoder()
        self._freeze_decoder(freeze_last_layer=False)
        self._freeze_lm_head()

    def _freeze_all(self):
        # freeze all layers
        self._freeze_decoder()
        self._freeze_encoder()
        self._freeze_lm_head()

    def freeze_model_parts(self, parts_to_freeze):
        if parts_to_freeze is None:
            return
        if parts_to_freeze not in self.freeze_name2func:
            raise ValueError(
                f"parts_to_freeze {parts_to_freeze} is not supported, "
                f"select from {list(self.freeze_name2func.keys())}"
            )

        self.freeze_name2func[parts_to_freeze]()


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

        self.whisper_model = Whisper2MedusaHeadsConditionalGeneration.from_pretrained(
            config.whisper_model_name
        )
        medusa_init_func = dict(
            base_head=self._initialize_medusa_base_heads,
            medusa_block=self._initialize_medusa_block_head,
        )
        if config.medusa_heads_type not in medusa_init_func:
            raise ValueError(
                f"medusa_heads_type {config.medusa_heads_type} is not supported, "
                f"select from {list(medusa_init_func.keys())}"
            )
        medusa_init_func[config.medusa_heads_type]()
        self.update_generation_config(self.config)
        if self.config.output_whisper_original:
            self._set_output_whisper_original()

    def _initialize_medusa_base_heads(self, use_base_head: bool = True):
        self.medusa_heads = nn.ModuleList()
        additional_head = 1 if use_base_head else 0
        for _ in range(
            self.config.medusa_num_heads + additional_head
        ):  # create head to time "0"
            head_list = []
            for _ in range(self.config.medusa_num_layers):
                head_list.append(
                    MedusaResBlock(self.config.d_model, self.config.medusa_hidden_size)
                )
            self.medusa_heads.append(nn.Sequential(*head_list))

    def _initialize_medusa_block_head(self):
        self.medusa_block = WhisperDecoderLayer(self.whisper_model.config, layer_idx=len(self.whisper_model.model.decoder.layers))
        self.medusa_block.load_state_dict(
            self.whisper_model.model.decoder.layers[-1].state_dict()
        )  # load the last layer of the whisper model
        self._initialize_medusa_base_heads(use_base_head=False)

    def update_generation_config(self, config):
        generation_config_dict = self.whisper_model.generation_config.to_dict()
        self.generation_config = medusa_utils.MedusaGenerationConfig(
            **generation_config_dict
        )
        self.generation_config.update(**config.to_dict())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
            config=config,
        )
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = (
                    medusa_utils.MedusaGenerationConfig.from_pretrained(
                        pretrained_model_name_or_path
                    )
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass
        return model

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


    def _update_medusa_outputs(
        self,
        outputs: ModelOutput,
        tree_outputs: ModelOutput,
        select_indices: torch.Tensor,
        selected_tree_indices: torch.Tensor,
        accept_length: torch.Tensor,
        prev_indices: torch.Tensor,
        use_base_logits: bool = False,
    ):
        if getattr(outputs, "decoder_attentions", None) is not None:
            prev_and_accept = torch.cat([prev_indices, select_indices], dim=0)
            orig_decoder_attentions = outputs.decoder_attentions
            tree_decoder_attentions = tree_outputs.decoder_attentions
            all_decoder_attentions = ()
            for i in range(len(orig_decoder_attentions)):
                orig_attentions = orig_decoder_attentions[i]
                tree_attentions = tree_decoder_attentions[i][
                    :, :, selected_tree_indices
                ]
                if len(tree_attentions.shape) < 4:
                    tree_attentions = tree_attentions.unsqueeze(2)
                tree_attentions = tree_attentions[:, :, :, prev_and_accept]
                all_decoder_attentions += ((orig_attentions, tree_attentions),)
            tree_outputs.decoder_attentions = all_decoder_attentions
        if getattr(outputs, "encoder_attentions", None) is not None:
            tree_outputs.encoder_attentions = outputs.encoder_attentions
        if getattr(outputs, "cross_attentions", None) is not None:
            orig_cross_attentions = outputs.cross_attentions
            tree_cross_attentions = tree_outputs.cross_attentions
            all_cross_attentions = ()
            for i in range(len(orig_cross_attentions)):
                current_orig_cross_attentions = orig_cross_attentions[i]
                current_tree_cross_attentions = tree_cross_attentions[i][
                    :, :, selected_tree_indices
                ]
                all_cross_attentions += (
                    (current_orig_cross_attentions, current_tree_cross_attentions),
                )
            tree_outputs.cross_attentions = all_cross_attentions
        if getattr(outputs, "encoder_hidden_states", None) is not None:
            tree_outputs.encoder_hidden_states = outputs.encoder_hidden_states
        if getattr(outputs, "decoder_hidden_states", None) is not None:
            orig_decoder_hidden_states = outputs.decoder_hidden_states
            tree_decoder_hidden_states = tree_outputs.decoder_hidden_states
            all_decoder_hidden_states = ()
            for i in range(len(orig_decoder_hidden_states)):
                current_orig_decoder_hidden_states = orig_decoder_hidden_states[i]
                current_tree_decoder_hidden_states = tree_decoder_hidden_states[i][
                    :, selected_tree_indices
                ]
                all_decoder_hidden_states += (
                    torch.cat(
                        [
                            current_orig_decoder_hidden_states,
                            current_tree_decoder_hidden_states,
                        ],
                        dim=1,
                    ),
                )
            tree_outputs.decoder_hidden_states = all_decoder_hidden_states
        if getattr(outputs, "past_key_values", None) is not None:
            # past_key_values contains decoder num_layers tuples, each with a tuple contains 4 tensors.
            # The first and second are the attention key and value with shape (batch_size, num_heads, sequence_length, embed_size_per_head)
            # The third and fourth are the cross attention (batch_size, num_heads, encoder_sequence_length, embed_size_per_head)
            all_past_key_values_tuple = ()
            for i in range(len(outputs.past_key_values)):  # loop over decoder layers
                layer_tree_past_key_values = tree_outputs.past_key_values[i]
                layer_orig_past_key_values = outputs.past_key_values[i]
                layer_past_key_values_tuple = ()
                for j in range(2):  # loop over attention key and value
                    tree_past = layer_tree_past_key_values[j][:, :, select_indices, :]
                    if use_base_logits:
                        tree_past = tree_past[:, :, : accept_length + 1, :]
                    else:
                        tree_past = tree_past[:, :, :accept_length, :]
                    orig_past = layer_orig_past_key_values[j]
                    layer_past_key_values_tuple += (
                        torch.cat([orig_past, tree_past], dim=2),
                    )
                layer_past_key_values_tuple += (
                    layer_orig_past_key_values[2],
                    layer_orig_past_key_values[3],
                )
                all_past_key_values_tuple += (layer_past_key_values_tuple,)
            tree_outputs.past_key_values = all_past_key_values_tuple

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
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = self.whisper_model.validate_stopping_criteria(
                stopping_criteria, max_length
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        # Medusa initialization

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = (
                self.get_medusa_choice()
            )  # replace with something from config

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

        new_token = 0
        accept_length_list = []
        with torch.inference_mode():
            while self.whisper_model._has_unfinished_sequences(
                this_peer_finished, synced_gpus, device=input_ids.device
            ):
                # prepare model inputs
                model_inputs = self.whisper_model.prepare_inputs_for_generation(
                    input_ids, **model_kwargs
                )
                # forward pass to get next token
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                orig_logits = logits_processor(
                    input_ids, outputs.logits[0].squeeze(0)
                ).unsqueeze(0)
                batch, num_medusa, seq_len, vocab_size = outputs.logits[1:].shape
                medusa_logits = logits_processor(
                    input_ids,
                    outputs.logits[1:].reshape(
                        batch * num_medusa * seq_len, vocab_size
                    ),
                )
                medusa_logits = medusa_logits.reshape(
                    batch, num_medusa, seq_len, vocab_size
                )

                candidates, tree_candidates = medusa_utils.generate_candidates(
                    medusa_logits,
                    orig_logits,
                    medusa_topk,
                    medusa_buffers["tree_indices"],
                )

                # Use tree attention to verify the candidates and get predictions
                process_logits, tree_outputs = medusa_utils.tree_decoding(
                    self,
                    tree_candidates,
                    outputs[
                        "past_key_values"
                    ],  # assuming that past_key_values is return from the model
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                    output_attentions,
                    output_hidden_states,
                    model_kwargs,
                )

                batch, seq_len, vocab_size = process_logits.shape
                process_logits = logits_processor(
                    input_ids,
                    process_logits.reshape(batch * seq_len, process_logits.shape[-1]),
                )
                process_logits = process_logits.reshape(batch, seq_len, vocab_size)

                # Evaluate the posterior of the candidates to select the accepted candidate prefix
                best_candidate, accept_length = medusa_utils.evaluate_posterior(
                    process_logits,
                    candidates,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                )

                accept_length_list.append(accept_length.item())
                # Tree_decoding generates logits for the next token across all Medusa options include the base head.
                # When no additional Medusa heads are used beyond the base_head,set use_base_logits to True.
                # The code then uses the first logit from tree_decoding along with the base logit calculated by the model.
                # Resulting in generating 2 tokens per step.
                if accept_length.item() == 0:
                    use_base_logits = True
                else:
                    use_base_logits = False

                # Update the input_ids and logits
                (
                    input_ids,
                    next_token_logits,
                    new_token,
                    next_tokens,
                ) = medusa_utils.update_inference_inputs(
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
                    use_base_logits,
                )
                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_logits,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (tree_outputs.decoder_attentions,)
                            if self.config.is_encoder_decoder
                            else (tree_outputs.attentions,)
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

                if use_base_logits:
                    len2use = accept_length.item() + 2
                else:
                    len2use = accept_length.item() + 1

                model_kwargs = self.whisper_model._update_model_kwargs_for_generation(
                    tree_outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    num_new_tokens = len2use
                )

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        (
                            ~torch.any(
                                next_tokens.tile(eos_token_id_tensor.shape[0], 1).eq(
                                    eos_token_id_tensor.unsqueeze(1)
                                ),
                                axis=1,
                            )
                        ).prod(dim=0)
                    )  # NOTE - this check if one of the tokens is eos_token_id_tensor

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids, scores
                )
                this_peer_finished = (
                    unfinished_sequences.max() == 0
                    or input_ids.shape[1] + self.config.medusa_num_heads
                    >= self.generation_config.max_length
                )

        if streamer is not None:
            streamer.end()

        # Set the last token to eos_token_id if it's not already assigned, ensuring that all Medusa heads' outputs are also set to eos_token_id
        new_input_ids = input_ids.clone()
        is_end_token = input_ids.tile(eos_token_id_tensor.shape[0], 1).eq(
            eos_token_id_tensor.unsqueeze(1)
        )
        had_end = torch.any(is_end_token, axis=1)
        for idx in range(len(had_end)):
            if had_end[idx]:
                for j in range(is_end_token.shape[1]):
                    if is_end_token[idx, j]:
                        new_input_ids[idx, j + 1 :] = eos_token_id_tensor.item()
                        break
        input_ids = new_input_ids

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
            return input_ids

    def _multi_heads_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
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
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self.whisper_model._prepare_generation_config(
            generation_config, **kwargs
        )
        self.whisper_model._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self.whisper_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")
        
        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self.whisper_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self.whisper_model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
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
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )
        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache
        
        if generation_mode == GenerationMode.GREEDY_SEARCH:

            # 11. run greedy search
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
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

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and not is_torchdynamo_compiling()
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _retrieve_logit_processors(
        self, generation_config, logits_processor, begin_index, num_beams, device):
        if generation_config.return_timestamps is True:
            raise NotImplementedError(
                "return_timestamps is not supported with medusa for now"
            )
            # TODO - Implement return_timestamps

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens, device=device)
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index, device=device
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None:
            raise NotImplementedError(
                "no_speech_detection is not supported with medusa for now"
            )
            # TODO - Implement no_speech_detection

        return logits_processor
    
    @staticmethod
    def _retrieve_max_frames_and_seek(batch_size, attention_mask, total_input_frames, is_shortform):
        if not is_shortform:
            raise NotImplementedError("Longform generation is not supported yet") 
        else:
            max_frames = torch.ones((batch_size,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((batch_size,), dtype=torch.long)

        return max_frames, seek
    
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
        **kwargs,
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
        if self.config.output_whisper_original:
            orig_logits = self._output_whisper_original(
                whisper_model_outputs,
                use_cache,
                past_key_values,
                head_mask,
                attention_mask,
                cross_attn_head_mask
            )
        medusa_logits = []

        if self.config.medusa_heads_type == "base_head":
            for i in range(len(self.medusa_heads)):
                head_out = self.medusa_heads[i](hidden_states)
                head_proj = self.whisper_model.proj_out(head_out)
                if i == 0:
                    base_logits = head_proj
                medusa_logits.append(head_proj)
                if disable_medusa:
                    # If Medusa is disabled, only the base head should be calculated.
                    # This occurs during the evaluation phase of the Medusa heads' output.
                    break

        else:
            base_logits = self.whisper_model.proj_out(hidden_states)
            medusa_logits.append(base_logits)
            self._forward_medusa_block(use_cache, attention_mask, head_mask, cross_attn_head_mask, past_key_values, output_attentions, whisper_model_outputs, medusa_logits, disable_medusa)

        stack_heads_logits = torch.stack(medusa_logits, dim=0)

        loss = None
        if labels is not None:
            loss_fct = MedusaCrossEntropyLoss(
                loss_on_original=self.config.medusa_loss_on_original
            )
            if getattr(self.config, "medusa_kl_loss", False):
                kl_fct = MedusaKLDivLoss(
                    lamda=self.config.medusa_kl_weight,
                    loss_on_original=self.config.medusa_loss_on_original,
                )
                if self.config.output_whisper_original:
                    kl_base_logits = orig_logits.detach()
                else:
                    kl_base_logits = base_logits.detach()

            # move labels to correct device to enable PP
            labels = labels.to(hidden_states.device)
            if self.config.medusa_loss_on_original:
                loss = loss_fct(stack_heads_logits, labels)
                if getattr(self.config, "medusa_kl_loss", False):
                    kl_loss = kl_fct(stack_heads_logits, kl_base_logits)
                    loss = loss + kl_loss
            else:
                loss = loss_fct(
                    stack_heads_logits[1:], labels
                )  # skip the first head - which is the base model
                if getattr(self.config, "medusa_kl_loss", False):
                    kl_loss = kl_fct(stack_heads_logits[1:], kl_base_logits)
                    loss = loss + kl_loss

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

    def _forward_medusa_block(
        self,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        whisper_model_outputs: Optional[Seq2SeqLMOutput] = None,
        medusa_logits: Optional[List[torch.FloatTensor]] = None,
        disable_medusa: Optional[bool] = False,
    ):
        current_past_key_values = whisper_model_outputs.past_key_values
        return_legacy_cache = False
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        if use_cache or current_past_key_values is not None:
            if isinstance(current_past_key_values, Cache) and not isinstance(current_past_key_values, EncoderDecoderCache):
                past_key_value = EncoderDecoderCache(current_past_key_values, DynamicCache())
            elif not isinstance(current_past_key_values, EncoderDecoderCache):
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                return_legacy_cache = True
                past_key_value = EncoderDecoderCache.from_legacy_cache(current_past_key_values)

        medusa_block_decoder_outputs = self.medusa_block(
            whisper_model_outputs.last_hidden_state,
            attention_mask=attention_mask,
            encoder_hidden_states=whisper_model_outputs.encoder_last_hidden_state,
            layer_head_mask=(head_mask[-1] if head_mask is not None else None),
            cross_attn_layer_head_mask=(
                cross_attn_head_mask[-1] if cross_attn_head_mask is not None else None
            ),
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if use_cache:
            if return_legacy_cache:
                next_cache = past_key_value.to_legacy_cache()
                whisper_model_outputs.past_key_values = next_cache
            else:
                whisper_model_outputs.past_key_values = past_key_value

        if output_attentions:
            whisper_model_outputs.decoder_attentions += (
                medusa_block_decoder_outputs[1],
            )

            if whisper_model_outputs.encoder_last_hidden_state is not None:
                whisper_model_outputs.cross_attentions += (
                    medusa_block_decoder_outputs[2],
                )
        if disable_medusa:
            # If Medusa is disabled, only the base head should be calculated, but the past key values must still be computed.
            # Therefore, in disable_medusa mode, the forward pass is performed only for the additional Whisper block, skipping the Medusa heads.
            return
        for i in range(len(self.medusa_heads)):
            head_out = self.medusa_heads[i](medusa_block_decoder_outputs[0])
            head_proj = self.whisper_model.proj_out(head_out)
            medusa_logits.append(head_proj)

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
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
        time_precision_features: float = 0.01,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        force_unique_generate_call: Optional[bool] = None,
        **kwargs,
    ):
        
        assert input_features.shape[0] == 1, "Only support batch size 1 for now!!"
        # # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )
        # 1. copy generation config
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        else:
            generation_config = copy.deepcopy(generation_config)

        # 2. set global generate variables
        input_stride = (
            self.whisper_model.model.encoder.conv1.stride[0]
            * self.whisper_model.model.encoder.conv2.stride[0]
        )
        num_segment_frames = input_stride * self.config.max_source_positions
        
        batch_size,total_input_frames = self.whisper_model._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        self.whisper_model._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
            # is_shortform=is_shortform,
        )
        
        timestamp_begin = self.whisper_model._set_return_timestamps(
            return_timestamps=return_timestamps,
            is_shortform=is_shortform,
            generation_config=generation_config,
        )
        # if isinstance(language, list):
        #     if is_multilingual is None:
        #         is_multilingual = False
        #     if len(set(language)) != 1:
        #         language = None
        #         is_multilingual = True or is_multilingual
        #     else:
        #         language = language[0]
        #         is_multilingual = False or is_multilingual

        # is_multilingual = True
        self.whisper_model._set_language_and_task(
            language=language,
            task=task,
            is_multilingual=is_multilingual,
            generation_config=generation_config,
        )
        self.whisper_model._set_num_frames(
            return_token_timestamps=return_token_timestamps,
            generation_config=generation_config,
            kwargs=kwargs,
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
            batch_size=batch_size,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self.whisper_model._check_decoder_input_ids(kwargs=kwargs)

        # 3. Retrieve logits processors
        device = kwargs["encoder_outputs"][0].device if "encoder_outputs" in kwargs else input_features.device
        begin_index = init_tokens.shape[1]
        num_beams = kwargs.get(
            "num_beams",
            generation_config.num_beams
            if hasattr(generation_config, "num_beams") and generation_config.num_beams is not None
            else 1,
        )

        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            num_beams=num_beams,
            device=device,
        )
        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]
        
        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size,
            attention_mask=attention_mask,
            total_input_frames=total_input_frames,
            is_shortform=is_shortform,
        )
        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        num_return_sequences = generation_config.num_return_sequences
        (
            batch_idx_map,
            cur_bsz,
            input_features,
            seek,
            max_frames,
            init_tokens,
            do_condition_on_prev_tokens,
        ) = self.whisper_model._expand_variables_for_generation(
            input_features=input_features,
            seek=seek,
            max_frames=max_frames,
            init_tokens=init_tokens,
            batch_size=batch_size,
            condition_on_prev_tokens=condition_on_prev_tokens,
            generation_config=generation_config,
        )
        current_segments = self.whisper_model._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=cur_bsz,
            generation_config=generation_config,
        )
        # 6 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            input_features, cur_bsz, batch_idx_map = self.whisper_model._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = (
                seek.to(torch.float32 if device.type == "mps" else torch.float64) * time_precision / input_stride
            )
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)
            # 6.2 cut out next 30s segment from input features
            segment_input = self.whisper_model._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            # 6.3 prepare decoder input ids
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
                device=init_tokens.device,
                suppress_tokens=suppress_tokens,
                timestamp_begin=timestamp_begin,
                kwargs=kwargs,
            )
            # 6.4 set max new tokens or max length
            self.whisper_model._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
            )
            # 6.5 Set current `begin_index` for all logit processors
            if logits_processor is not None:
                for proc in logits_processor:
                    if hasattr(proc, "set_begin_index"):
                        proc.set_begin_index(begin_index)

            # 6.6 Run generate with fallback
            (
                seek_sequences,
                seek_outputs,
                should_skip,
                do_condition_on_prev_tokens,
                model_output_type,
            ) = self.generate_with_fallback(
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
                is_shortform=is_shortform,
                batch_size=batch_size,
                attention_mask=attention_mask,
                kwargs=kwargs,
            )

            # 6.7 In every generated sequence, split by timestamp tokens and extract segments
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
                    time_precision_features=time_precision_features,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                    return_token_timestamps=return_token_timestamps,
                    decoder_input_ids=decoder_input_ids,
                )

                seek[prev_i] += segment_offset

                current_segments[prev_i] += segments

            if force_unique_generate_call:
                break

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        final_segments = (
            [x[1:] for x in current_segments]
            if (prompt_ids is not None and generation_config.prompt_condition_type == "first-segment")
            else current_segments
        )

        # if return_dict_in_generate=True and we forced a unique call to generate or return_timestamps=False, meaning we are sure only one call to generate has been made,
        # -> we can return a ModelOutput
        # otherwise, return_dict_in_generate is applied in the 'result' of each segment in final_segments
        if (
            return_dict_in_generate
            and generation_config.return_dict_in_generate
            and (force_unique_generate_call or not return_timestamps)
        ):
            # only one call to generate_with_fallback, we can return a ModelOutput
            outputs = self.whisper_model._stack_split_outputs(seek_outputs, model_output_type, self.device, kwargs)
            if num_return_sequences > 1:
                if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
                    outputs.encoder_attentions = tuple(
                        outputs.encoder_attentions[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_attentions))
                    )
                if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
                    outputs.encoder_hidden_states = tuple(
                        outputs.encoder_hidden_states[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_hidden_states))
                    )
            return outputs

        padded_outputs = _pad_to_max_length(
            current_segments=final_segments,
            pad_token_id=generation_config.pad_token_id,
            device=self.device,
            padding_side="right",
            return_token_timestamps=return_token_timestamps,
            force_unique_generate_call=force_unique_generate_call,
        )

        if return_dict_in_generate and generation_config.return_dict_in_generate:
            logger.warning_once(
                "You have passed `return_dict_in_generate=True` and `return_timestamps=True`, this automatically sets `return_segments=True` to access the resuls of the underlying calls to GenerationMixin's generate in the returned `segments`."
            )
            return_segments = True
        elif not return_segments and not return_token_timestamps:
            return padded_outputs

        if return_token_timestamps:
            sequences, token_timestamps = padded_outputs
            outputs = {
                "sequences": sequences,
                "token_timestamps": token_timestamps,
            }
        else:
            sequences = padded_outputs
            outputs = {
                "sequences": sequences,
            }

        if return_segments:
            outputs["segments"] = final_segments

        return outputs

        if is_shortform:
            if temperature is not None:
                kwargs["temperature"] = temperature

            decoder_input_ids = kwargs.pop("decoder_input_ids", None)
            if decoder_input_ids is None:
                one_tensor = torch.ones(
                    (batch_size, 1), device=self.device, dtype=torch.long
                )
                decoder_input_ids = torch.cat(
                    [t * one_tensor for t in init_tokens], dim=-1
                )

            if prompt_ids is not None:
                decoder_input_ids = torch.cat(
                    [
                        prompt_ids[None].repeat(decoder_input_ids.shape[0], 1),
                        decoder_input_ids,
                    ],
                    dim=-1,
                )

            if (
                kwargs.get("max_new_tokens", 0) + decoder_input_ids.shape[-1]
                > self.config.max_target_positions
            ):
                max_new_tokens = kwargs.get("max_new_tokens", 0)
                raise ValueError(
                    f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
                    f"is {max_new_tokens}. Thus, the combined length of "
                    f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                    f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                    "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                    f"so that their combined length is less than {self.config.max_target_positions}."
                )

            outputs = self._multi_heads_generate(
                input_features,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )

            if generation_config.return_token_timestamps and hasattr(
                generation_config, "alignment_heads"
            ):
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs,
                    generation_config.alignment_heads,
                    num_frames=generation_config.num_frames,
                )

            return outputs

        else:
            raise NotImplementedError("Longform generation is not supported yet")

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
        is_shortform,
        batch_size,
        attention_mask,
        kwargs,
    ):
        kwargs = copy.copy(kwargs)

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
            if generation_config.do_sample:
                generation_config.num_beams = 1

            generate_kwargs = copy.copy(kwargs)
            for key in ["do_sample", "temperature", "num_beams"]:
                if key in generate_kwargs:
                    del generate_kwargs[key]

            cur_bsz = decoder_input_ids.shape[0]
            if generation_config.cache_implementation == "static" and cur_bsz < batch_size:
                segment_input = F.pad(segment_input, (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0)
                decoder_input_ids = F.pad(
                    decoder_input_ids, (0, 0, 0, batch_size - cur_bsz), value=generation_config.pad_token_id
                )
                if generate_kwargs.get("decoder_attention_mask") is not None:
                    generate_kwargs["decoder_attention_mask"] = F.pad(
                        generate_kwargs["decoder_attention_mask"], (0, 0, 0, batch_size - cur_bsz), value=True
                    )
                if generate_kwargs.get("encoder_outputs") is not None:
                    generate_kwargs["encoder_outputs"] = F.pad(
                        generate_kwargs["encoder_outputs"], (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0
                    )
            seek_outputs = self._multi_heads_generate(
                segment_input,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )

            model_output_type = type(seek_outputs)

            # post-process sequence tokens and outputs to be in list form
            seek_sequences, seek_outputs = self.whisper_model._postprocess_outputs(
                seek_outputs=seek_outputs,
                decoder_input_ids=decoder_input_ids,
                return_token_timestamps=return_token_timestamps,
                generation_config=generation_config,
                is_shortform=is_shortform,
            )

            if cur_bsz < batch_size:
                seek_sequences = seek_sequences[:cur_bsz]
                seek_outputs = seek_outputs[:cur_bsz]

            # 6.7 Extract cut sequences from every sequence and check if fallback should be applied
            # Loop over each decoded audio individually as each decoding can be of a different length
            new_fallback_index_map = []
            new_segment_input = []
            new_decoder_input_ids = []
            new_decoder_attention_mask = []

            for i, seek_sequence in enumerate(seek_sequences):
                # remove all padding tokens, except for the eos token
                if seek_sequence[-1] == generation_config.pad_token_id:
                    num_paddings = (seek_sequence == generation_config.pad_token_id).sum()
                    if generation_config.pad_token_id == generation_config.eos_token_id:
                        # we do not remove the eos token id since it is needed for avg logprob calculation in _need_fallback
                        num_paddings -= 1
                    if num_paddings != 0:
                        seek_sequence = seek_sequence[:-num_paddings]

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

                # remove eos token
                if seek_sequence[-1] == generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]

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

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens, model_output_type

    def freeze_model_parts(self, parts_to_freeze: str):
        self.whisper_model.freeze_model_parts(parts_to_freeze)

    def _set_output_whisper_original(self):
        self.whisper_model.config.output_hidden_states = True
        self.config.output_hidden_states = True
        self.whisper_layer = WhisperDecoderLayer(self.whisper_model.config, layer_idx=len(self.whisper_model.model.decoder.layers)-1)
        self.whisper_layer.load_state_dict(
            self.whisper_model.model.decoder.layers[-1].state_dict()
        )  # load the last layer of the whisper model
        for param in self.whisper_layer.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _output_whisper_original(
        self,
        whisper_model_outputs: Optional[Seq2SeqLMOutput] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        head_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
    ):
        current_past_key_values = whisper_model_outputs.past_key_values
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        if use_cache or current_past_key_values is not None:
            if isinstance(current_past_key_values, Cache) and not isinstance(current_past_key_values, EncoderDecoderCache):
                past_key_value = EncoderDecoderCache(current_past_key_values, DynamicCache())
            elif not isinstance(current_past_key_values, EncoderDecoderCache):
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_value = EncoderDecoderCache.from_legacy_cache(current_past_key_values)
            past_key_value = past_key_value.to_legacy_cache()[:-1] # last layer
            past_key_value = EncoderDecoderCache.from_legacy_cache(past_key_value)

        orig_hidden_state = self.whisper_layer(
            whisper_model_outputs.decoder_hidden_states[-2],
            attention_mask=attention_mask,
            encoder_hidden_states=whisper_model_outputs.encoder_last_hidden_state,
            layer_head_mask=(head_mask[-2] if head_mask is not None else None),
            cross_attn_layer_head_mask=(
                cross_attn_head_mask[-2] if cross_attn_head_mask is not None else None
            ),
            past_key_value=past_key_value,
            output_attentions=None,
            use_cache=use_cache,
        )
        norm_state = self.whisper_model.model.decoder.layer_norm(orig_hidden_state[0])
        logits = self.whisper_model.proj_out(norm_state)
        return logits


def get_model(args_i):
    if not os.path.exists(args_i.whisper_model_name):
        config = MedusaConfig(
            medusa_num_heads=args_i.medusa_num_heads,
            medusa_num_layers=args_i.medusa_num_layers,
            whisper_model_name=args_i.whisper_model_name,
            medusa_hidden_size=args_i.medusa_hidden_size,
            medusa_heads_type=args_i.medusa_heads_type,
            medusa_choices=args_i.medusa_choices,
            medusa_kl_loss=args_i.medusa_kl_loss,
            medusa_kl_weight=args_i.medusa_kl_weight,
            medusa_loss_on_original=args_i.medusa_loss_on_original,
            output_whisper_original=args_i.output_whisper_original,
        )
        model = WhisperMedusaModel(config)
    else:
        model = WhisperMedusaModel.from_pretrained(args_i.whisper_model_name)

    return model
