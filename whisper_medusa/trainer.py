import json
import logging

from functools import partial
from pathlib import Path

from transformers import WhisperProcessor

from whisper_medusa.dataset.dataset import (
    DataCollatorSpeechSeq2SeqWithPadding,
    get_dataset,
)
from whisper_medusa.models import get_model
from whisper_medusa.utils.config_and_args import get_training_args
from whisper_medusa.utils.medusa_trainer_cls import MedusaTrainer
from whisper_medusa.utils.metrics import compute_metrics
from whisper_medusa.utils.utils import parse_args, set_logger, set_seed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _train(args_i, training_args, callbacks=None):
    set_seed(args_i.seed)

    model = get_model(args_i)

    processor = WhisperProcessor.from_pretrained(
        args_i.whisper_model_name, language=args_i.language, task="transcribe"
    )

    dataset_dict = get_dataset(args_i, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    compute_metrics_arg = (
        partial(compute_metrics, tokenizer=processor.tokenizer)
        if args_i.compute_wer
        else None
    )

    trainer = MedusaTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset={"validation": dataset_dict["validation"]},
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics_arg,
        callbacks=callbacks,
    )

    model.freeze_model_parts(args_i.parts_to_freeze)
    logging.info(
        f"Network type: {args_i.whisper_model_name}, net size {count_parameters(model)}"
    )
    trainer.train(resume_from_checkpoint=args_i.resume_from_checkpoint)

    model_comp_path_obj = Path(args_i.output_path) / "model_components"
    model_comp_path_obj.mkdir(parents=True, exist_ok=True)
    model_comp_path_str = model_comp_path_obj.as_posix()

    model.save_pretrained(model_comp_path_str)
    processor.tokenizer.save_pretrained(model_comp_path_str)
    processor.save_pretrained(model_comp_path_str)

    results = trainer.evaluate(eval_dataset=dataset_dict["test"])

    message = f"loss: {results['eval_loss']}"
    if args_i.compute_wer:
        message += f", WER: {results['eval_wer']}"
    logging.info(message)


def main(args_i, training_args):
    _train(args_i, training_args)


if __name__ == "__main__":
    set_logger()
    args = parse_args()
    print(json.dumps(args.__dict__, indent=2))
    train_args = get_training_args(args)
    assert train_args.greater_is_better == (
        "loss" not in train_args.metric_for_best_model
    ), "training_args.greater_is_better should be set to True if your measuring metric is 'loss'"
    main(args, train_args)
