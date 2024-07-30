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

SR = 16000
def evaluate_model(args):


    processor = WhisperProcessor.from_pretrained(args.whisper_model_name)
    model = WhisperMedusaModel.from_pretrained(
        args.whisper_model_name,
    )
    data = pd.read_csv(
        args.data_path,
    )
    data = data.fillna("")
    # TODO- use get_device function
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
    lang_list = []
    audio_list = []
    with torch.no_grad():
        for i, row in tqdm(data.iterrows(), total=len(data)):
            input_speech, sr = torchaudio.load(row.audio)
            if sr != SR:
                input_speech = torchaudio.transforms.Resample(sr, SR)(input_speech)
            input_features = processor(
                input_speech.squeeze(),
                return_tensors="pt",
                sampling_rate=SR, 
            ).input_features
            input_features = input_features.to(device)

            model_output = model.generate(
                input_features,
                language=args.language,
            )
            if isinstance(model_output, WhisperMedusaGenerationOutput): # TODO - change this to work on both cases
                count_selected_heads = model_output.count_selected_heads
                predict_ids = model_output.input_ids[0]
            else:
                count_selected_heads = {} # regular whisper model
                predict_ids = model_output[0]
            pred = processor.decode(predict_ids, skip_special_tokens=True)
            preds.append(pred)
            gts.append(row.sentence)
            lang_list.append(args.language)
            audio_list.append(row.audio)
            
    wer, wers = compute_wer(preds, gts)
    cer, cers = compute_cer(preds, gts)
    logging.info(f"=======================")
    logging.info(f"WER: {wer}")
    logging.info(f"CER: {cer}")
    logging.info(f"=======================")

    results = pd.DataFrame(
        {"audio":audio_list, "label": gts, "prediction": preds, "wer": wers, "cer": cers, "language": lang_list}
    )
    out_path = os.path.dirname(args.out_file_path)
    results.to_csv(
        out_path,
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
        required=True,
        help="Path to test data csv file",
    )
    parser.add_argument(
        "--out-file-path",
        type=str,
        required=True,
        help="Path to output test data csv file",
    )
    parser.add_argument(
        "--language", type=str, default="en", help="transcribe language"
    )
    parser.add_argument(
        "--cuda",
        type="custom_bool",
        default=False,
        help="use CUDA", # TODO- use get_device function
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate_model(args)
