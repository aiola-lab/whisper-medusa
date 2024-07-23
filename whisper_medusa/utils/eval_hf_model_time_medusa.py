import argparse
import logging
import warnings
import json

import pandas as pd
import torch
import torchaudio

from whisper_medusa.models.whisper import WhisperMedusaModel, WhisperMedusaGenerationOutput
from whisper_medusa.utils.utils import str2bool
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_medusa.utils.metrics import compute_wer, compute_cer
from tqdm import tqdm
import os
import numpy as np
warnings.filterwarnings("ignore")

architecture2module = {
    "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
    "WhisperMedusaModel": WhisperMedusaModel,
}


def evaluate_model(args):

    if os.path.exists(args.whisper_model_name):
        config = json.load(open(args.whisper_model_name + "/config.json"))
        suppress_ids = config.get("suppress_ids", [-1])

        model_module = architecture2module.get(config["architectures"][0], None)
        if model_module is None:
            raise ValueError(
                f"Unsupported architecture: {config['architecture'][0]}, select from {architecture2module.keys()}"
            )

        processor = WhisperProcessor.from_pretrained(args.whisper_model_name)
        model = model_module.from_pretrained(
            args.whisper_model_name,
        )
    else:
        processor = WhisperProcessor.from_pretrained(args.whisper_model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.whisper_model_name,
        )  
        suppress_ids = [-1]

    data = pd.read_csv(
        args.data_path,
    )
    data = data.fillna("")

    logging.info(f"Using prompts: {args.use_prompts}")
    if args.cuda:
        is_available = torch.cuda.is_available()
        if is_available:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    preds = []
    gts = []
    data_list = []
    lang_list = []
    with torch.no_grad():
        for i, row in tqdm(data.iterrows(), total=len(data)):
            prompt_ids = (
                processor.get_prompt_ids(row.prompt) if args.use_prompts else None
            )
            input_speech, sr = torchaudio.load(row.audio)
            input_features = processor(
                input_speech.squeeze(),
                return_tensors="pt",
                sampling_rate=16000,  # dtype=torch.float16
            ).input_features
            input_features = input_features.to(device)

            model_output = model.generate(
                input_features,
                prompt_ids=prompt_ids,
                language=args.language,
                # suppress_tokens=suppress_ids,
            )
            if isinstance(model_output, WhisperMedusaGenerationOutput):
                count_selected_heads = model_output.count_selected_heads
                predict_ids = model_output.input_ids[0]
            else:
                count_selected_heads = {} # regular whisper model
                predict_ids = model_output[0]
            pred = processor.decode(predict_ids, skip_special_tokens=True)
            preds.append(pred)
            gts.append(row.sentence)
            lang_list.append(args.language)
            
    wer, wers = compute_wer(preds, gts)
    cer, cers = compute_cer(preds, gts)
    logging.info(f"=======================")
    logging.info(f"WER: {wer}")
    logging.info(f"CER: {cer}")
    logging.info(f"=======================")

    results = pd.DataFrame(
        {"audio":[i[0] for i in data_list], "label": gts, "prediction": preds, "wer": wers, "cer": cers, "language": lang_list}
    )
    out_path = os.path.dirname(args.out_file_path)
    base_filename = os.path.basename(args.out_file_path).replace(".csv", "")
    wer_filename = os.path.join(out_path, f"{base_filename}_wer_results.csv")
    results.to_csv(
        wer_filename,
        index=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "custom_bool", str2bool)
    parser.add_argument(
        "--whisper-model-name",
        type=str,
        required=True,
        help="Path to trained Whisper model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/ec2-user/datasets/metro/training_data_files/30-10-2023_19-11-2023/test.csv",
        help="Path to test data csv file",
    )
    parser.add_argument(
        "--out-file-path",
        type=str,
        default="/home/ec2-user/workspaces/projects/faster-whisper/medusa_speed_cuda.csv",
        help="Path to output test data csv file",
    )
    parser.add_argument(
        "--num-generate-per-file",
        type=int,
        default=10,
        help="How much time to run generate function per file",
    )
    parser.add_argument(
        "--use-prompts", type=str2bool, default=False, help="Whether to use prompts"
    )

    parser.add_argument(
        "--language", type=str, default="en", help="transcribe language"
    )
    parser.add_argument(
        "--cuda",
        type="custom_bool",
        default=False,
        help="use CUDA",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate_model(args)
