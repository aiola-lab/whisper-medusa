import jiwer.transforms as tr
from jiwer import compute_measures


def compute_wer(predictions, references):
    wer_standardize = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            tr.RemoveKaldiNonWords(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.RemovePunctuation(),
            tr.Strip(),
            tr.ReduceToListOfListOfWords(),
        ]
    )

    incorrect = 0
    total = 0
    wers = []
    for prediction, reference in zip(predictions, references):
        if not wer_standardize(reference)[0]:
            reference = "EMPTY"
        if not wer_standardize(prediction)[0]:
            prediction = "EMPTY"
        measures = compute_measures(
            truth=reference,
            hypothesis=prediction,
            truth_transform=wer_standardize,
            hypothesis_transform=wer_standardize,
        )
        wers.append(measures["wer"])
        incorrect += (
            measures["substitutions"] + measures["deletions"] + measures["insertions"]
        )
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total, wers


def compute_cer(predictions, references):
    cer_standardize = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.RemovePunctuation(),
            tr.Strip(),
            tr.ReduceToListOfListOfChars(),
        ]
    )
    incorrect = 0
    total = 0
    cers = []
    for prediction, reference in zip(predictions, references):
        if not cer_standardize(reference)[0]:
            reference = "EMPTY"
        if not cer_standardize(prediction)[0]:
            prediction = "EMPTY"
        measures = compute_measures(
            truth=reference,
            hypothesis=prediction,
            truth_transform=cer_standardize,
            hypothesis_transform=cer_standardize,
        )
        cers.append(measures["wer"])
        incorrect += (
            measures["substitutions"] + measures["deletions"] + measures["insertions"]
        )
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total, cers


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer, _ = compute_wer(pred_str, label_str)
    return dict(wer=wer)
