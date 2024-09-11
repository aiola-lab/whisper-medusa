from dataclasses import dataclass
from typing import List

from transformers import AutoConfig
from transformers.models.whisper import WhisperConfig
from transformers import AutoConfig, Seq2SeqTrainingArguments


def remove_duplicates_config(config, kwargs):
    keys_to_remove = [key for key in kwargs if key in config]
    # Remove the keys from kwargs
    for key in keys_to_remove:
        del kwargs[key]
    
    return kwargs

@dataclass
class MedusaConfig(WhisperConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 4.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        whisper_model_name (str, optional): The name or path of the base model. Default is "openai/whisper-large-v2".
        medusa_choices (List[int], optional): The beam size for the medusa heads. Default is [1,6,5,4,3].
        medusa_heads_type (str, optional): The type of the medusa heads. Default is "base_head".
        medusa_loss_on_original (bool, optional): Whether to include the original logits in the loss calculation. Default is False.
        medusa_kl_loss (bool, optional): Whether to include KL divergence loss. Default is False.
        medusa_kl_weight (float, optional): The weight of the KL divergence loss. Default is 0.
        output_whisper_original (bool, optional): Whether to output the original whisper model logits. Default is False. (for kl loss)
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads: int = 4,
        medusa_num_layers: int = 1,
        medusa_hidden_size: int = 1280,
        whisper_model_name: str = "openai/whisper-large-v2",
        medusa_choices: List[int] = [1, 1, 1, 1, 1],
        medusa_heads_type: str = "base_head",
        medusa_loss_on_original: bool = False,
        medusa_kl_loss: bool = False,
        medusa_kl_weight: float = 0,
        output_whisper_original: bool = False,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(whisper_model_name)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.whisper_model_name = whisper_model_name
        self.medusa_hidden_size = medusa_hidden_size
        self.medusa_choices = medusa_choices
        self.medusa_heads_type = medusa_heads_type
        self.medusa_loss_on_original = medusa_loss_on_original
        self.medusa_kl_loss = medusa_kl_loss
        self.medusa_kl_weight = medusa_kl_weight
        self.output_whisper_original = output_whisper_original
        config_dict = config.to_dict()
        kwargs = remove_duplicates_config(config_dict, kwargs)
        super().__init__(**config_dict, **kwargs)


def get_training_args(arguments):
    training_args = Seq2SeqTrainingArguments(
        output_dir=arguments.output_path,  # change to a repo name of your choice
        per_device_train_batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.lr,
        warmup_steps=arguments.warmup_steps,
        max_steps=arguments.max_steps,
        gradient_checkpointing=False,
        fp16=arguments.fp16,
        evaluation_strategy="steps",
        save_total_limit=2,
        per_device_eval_batch_size=arguments.batch_size,
        predict_with_generate=arguments.predict_with_generate,
        generation_max_length=225,
        save_steps=arguments.save_steps,
        eval_steps=arguments.eval_steps,
        logging_steps=1,
        report_to=["wandb"] if arguments.wandb_logging else ["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_loss",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=4, 
        optim=arguments.optim,
        label_names=["labels"],
        save_safetensors=arguments.save_safetensors,
        lr_scheduler_type= arguments.lr_scheduler_type, 
    )
    return training_args
