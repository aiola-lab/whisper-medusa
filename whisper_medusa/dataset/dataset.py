import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd
import torch
import torchaudio
from transformers import WhisperProcessor

from whisper_medusa.utils.utils import token_padding

SAMPLE_RATE = 16_000


class ASRDataSet(torch.utils.data.Dataset):
    """
    ASR Dataset

    Parameters
    ----------
    data_path : str. Path to csv file with columns: "sentence", "path", and "language".
    """

    def __init__(
        self,
        data_path,
        split,
        processor: WhisperProcessor,
        target_sample=16_000,
    ):
        assert split in [
            "train",
            "val",
            "test",
        ]  # sanity for case we will use split later.
        self.split = split

        # read csv file + create dataset
        self.data_path = data_path
        self.dataset_df = pd.DataFrame()
        self._init_dataset_obj()

        self.dataset = self.dataset_df.to_dict("records")
        self.length = len(self.dataset)

        self.target_sample_rate = target_sample

        self.processor = processor

    def _init_dataset_obj(self):
        self.dataset_df = pd.read_csv(self.data_path)
        self.dataset_df.sentence = self.dataset_df.sentence.fillna("")
        self.dataset = self.dataset_df.to_dict("records")

        # check case language is not specified
        if "language" not in self.dataset_df.columns:
            logging.info("[NOTE]: No language specified, using tokenizer's language")

    def __len__(self):
        return self.length

    @staticmethod
    def speech_file_to_array(path, resampling_to=16_000):
        batch = {}
        # todo: consider using to soundfile to prevent issues with mp3 files with 'ffmpeg>=5'.
        #  If you encounter issues with mp3 files use `conda install 'ffmpeg<5'`
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, resampling_to)
        array = resampler(speech_array)[0].numpy()
        batch["audio"] = {"array": array, "sampling_rate": resampling_to}
        return batch

    def prepare_dataset(self, batch):
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # encode target text to label ids
        transcript = batch.pop("sentence")
        d_lang = self.processor.tokenizer.language
        self.processor.tokenizer.set_prefix_tokens(language=batch["language"])
        batch["labels"] = self.processor.tokenizer(transcript).input_ids
        batch["decoder_input_ids"] = batch["labels"]
        self.processor.tokenizer.set_prefix_tokens(language=d_lang)
        return batch

    def __getitem__(self, item):
        record = self.dataset[item]
        data = record["audio"]
        example = self.speech_file_to_array(
            data,
            resampling_to=self.target_sample_rate,
        )
        # preprocess text
        example["sentence"] = record["sentence"]
        example["language"] = record.get("language", self.processor.tokenizer.language)

        prepared_datasets = self.prepare_dataset(example)
        return prepared_datasets


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        batch["labels"] = token_padding(features, self.processor, batch_key="labels")
        batch["labels"] = batch["labels"][:, 1:]

        if features[0].get("decoder_input_ids", None) is not None:
            batch["decoder_input_ids"] = token_padding(
                features,
                self.processor,
                batch_key="decoder_input_ids",
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
            batch["decoder_input_ids"] = batch["decoder_input_ids"][:, :-1]

        batch["language"] = [f["language"] for f in features]
        return batch


def get_dataset(args_i, processor):
    dataset = dict()

    dataset["train"] = ASRDataSet(
        data_path=args_i.train_data_path,
        split="train",
        processor=processor,
        target_sample=SAMPLE_RATE,
    )

    dataset["test"] = ASRDataSet(
        data_path=args_i.test_data_path,
        split="test",
        processor=processor,
        target_sample=SAMPLE_RATE,
    )

    dataset["validation"] = ASRDataSet(
        data_path=args_i.validation_data_path,
        split="val",
        processor=processor,
        target_sample=SAMPLE_RATE,
    )

    return dataset
