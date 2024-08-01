from dataclasses import dataclass
from typing import List

from transformers import AutoConfig
from transformers.models.whisper import WhisperConfig


@dataclass
class MedusaConfig(WhisperConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 4.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        whisper_model_name (str, optional): The name or path of the base model. Default is "openai/whisper-large-v2".
        medusa_choices (List[int], optional): The beam size for the medusa heads. Default is [1,6,5,4,3].
        init_from_proj (bool, optional): Whether to initialize the medusa heads from the whisper base projection layer. Default is True.
        loss_on_orig (bool, optional): Whether to include the original logits in the loss calculation. Default is False.
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
        super().__init__(**config.to_dict(), **kwargs)
