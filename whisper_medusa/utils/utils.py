import argparse
import logging
import random

import numpy as np
import torch
import wandb

def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str_or_list(s):
    if s is None:
        return s
    else:
        new_s = [i.strip() for i in s.split(",")]
        if len(new_s) == 1:  # case it is a single string
            return new_s[0]
        else:
            return new_s


def str_int_list(s):
    if s is None:
        return s
    else:
        new_s = [int(i.strip()) for i in s.split(",")]
        if len(new_s) == 1:  # case it is a single int
            return new_s[0]
        else:
            return new_s


def get_device(gpu_id="0"):
    if torch.cuda.is_available():
        logging.info(f"GPU available. Using GPU {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available():
        logging.info("MPS available. Using MPS")
        return torch.device("mps")
    else:
        logging.info("Using CPU")
        return torch.device("cpu")

def token_padding(features, processor, batch_key, pad_token_id=-100):
    # get the tokenized label sequences
    token_features = [{"input_ids": feature[batch_key]} for feature in features]
    # pad the labels to max length
    token_batch = processor.tokenizer.pad(token_features, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    padded_tokens = token_batch["input_ids"].masked_fill(
        token_batch.attention_mask.ne(1), pad_token_id
    )

    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (padded_tokens[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        padded_tokens = padded_tokens[:, 1:]

    return padded_tokens


def parse_args():
    parser = argparse.ArgumentParser("Medusa training")

    parser.register("type", "custom_bool", str2bool)

    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="training data path",
    )
    parser.add_argument(
        "--validation-data-path",
        type=str,
        required=True,
        help="validation data path",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="test data path",
    )
    parser.add_argument(
        "--debug-mode",
        type="custom_bool",
        default=True,
        help="use limited amount of eval data",
    )
    parser.add_argument(
        "--debug-examples",
        type=int,
        default=1000,
        help="number of examples to use in debug mode",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="gradient accumulation steps",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
        help="number of update steps to train for",
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="number of every steps to save model checkpoint",
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="number of every steps to evaluate the model",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="batch size",
    )

    parser.add_argument("--seed", type=int, default=42, help="seed to use")

    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="warmup steps for scheduler"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/whisper_ft",
        help="where (path) to output the results",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Experiment name",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="language",
    )

    parser.add_argument(
        "--fp16",
        type="custom_bool",
        default="True",
        help="use fp16 training",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )

    parser.add_argument(
        "--optim",
        type=str,
        default="adafactor",
        help="optimization strategy",
    )

    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        choices=["linear", "constant"],
        help="scheduler type",
    )

    parser.add_argument(
        "--predict-with-generate",
        type="custom_bool",
        default="True",
        help="use generate for prediction",
    )

    parser.add_argument(
        "--whisper-model-name",
        type=str,
        default="openai/whisper-large-v2",
        help="open ai's whisper model name",
    )
    parser.add_argument(
        "--parts-to-freeze",
        type=str,
        default=None,
        choices=[
            "whisper",
            "all_but_last",
            None,
        ],
        help="which model parts to freeze",
    )
    parser.add_argument(
        "--medusa_num_layers",
        type=int,
        default=1,
        help="medusa number of layers",
    )
    parser.add_argument(
        "--medusa-num-heads",
        type=int,
        default=10,
        help="medusa number of heads",
    )
    parser.add_argument(
        "--medusa-hidden-size",
        type=int,
        default=1280,
        help="medusa hidden size",
    )
    parser.add_argument(
        "--medusa-choices",
        type=str_int_list,
        default="1,1,1,1,1,1,1,1,1,1,1",
        help="he beam size for the medusa head",
    )
    parser.add_argument(
        "--medusa-heads-type",
        type=str,
        default="base_head",
        choices=["base_head", "medusa_block"],
        help="which medusa heads type to use",
    )
    parser.add_argument(
        "--medusa-loss-on-original",
        type="custom_bool",
        default=False,
        help="run medusa with loss on original logits",
    )
    parser.add_argument(
        "--medusa-kl-loss",
        type="custom_bool",
        default=False,
        help="run medusa with kl loss",
    )
    parser.add_argument(
        "--medusa-kl-weight",
        type=float,
        default=0,
        help="lamda for kl loss",
    )
    parser.add_argument(
        "--output-whisper-original",
        type="custom_bool",
        default=False,
        help="output the original whisper logits",
    )
    parser.add_argument(
        "--save-safetensors",
        type="custom_bool",
        default=True,
        help="If true, use safetensors else use regular torch.save",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type="custom_bool",
        default=False,
        help="resume training from checkpoint, assumes checkpoint is in the output-path",
    )

    parser.add_argument(
        "--wandb-logging",
        type="custom_bool",
        default=False,
        help="If true, use wandb to report training metrics.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="The wandb project to log to",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="The wandb entity to log to",
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="The wandb id to resume from",
    )

    
    args_ = parser.parse_args()

    if args_.wandb_logging:
        if args_.wandb_entity is None or args_.wandb_project is None:
            raise ValueError("wandb-entity and wandb-project must be provided when using wandb logging")
        if args_.resume_from_checkpoint and not args_.wandb_id is None:
            wandb.init(
                id=args_.wandb_id,
                project=args_.wandb_project,
                entity=args_.wandb_entity,
                resume="must",
            )
        else:
            name = (
                f"medusa_{args_.whisper_model_name}_{args_.language}_{args_.exp_name}"
            )
            wandb.init(name=name, project=args_.wandb_project, entity=args_.wandb_entity)
            wandb.config.update(args_)

    return args_

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)