import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor

from whisper_medusa.models import WhisperMedusaModel
from whisper_medusa.utils.metrics import compute_cer, compute_wer
from whisper_medusa.utils.utils import get_device, set_logger, str2bool

warnings.filterwarnings("ignore")

SAMPLING_RATE = 16000


def evaluate_model(args, device):
    data = pd.read_csv(
        args.data_path,
    )
    data = data.fillna("")

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperMedusaModel.from_pretrained(
        args.model_name,
    )
    model = model.to(device)

    preds = []
    gts = []
    lang_list = []
    audio_list = []

    with torch.no_grad():
        for i, row in tqdm(data.iterrows(), total=len(data)):
            input_speech, sr = torchaudio.load(row.audio)
            if sr != SAMPLING_RATE:
                input_speech = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(
                    input_speech
                )
            input_features = processor(
                input_speech.squeeze(),
                return_tensors="pt",
                sampling_rate=SAMPLING_RATE,
            ).input_features
            input_features = input_features.to(device)

            model_output = model.generate(
                input_features,
                language=args.language,
            )

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
        {
            "audio": audio_list,
            "label": gts,
            "prediction": preds,
            "wer": wers,
            "cer": cers,
            "language": lang_list,
        }
    )

    output_file_path = Path(args.out_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file_path, index=False)
    logging.info(f"Results saved to {output_file_path.as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "custom_bool", str2bool)
    parser.add_argument(
        "--model-name",
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

    args = parser.parse_args()
    device = get_device()
    set_logger()
    evaluate_model(args, device)
