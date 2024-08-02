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
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
