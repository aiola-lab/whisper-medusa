import logging

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_medusa.models.whisper import (
    WhisperMedusaModel,
    WhisperMedusaGenerationOutput,
)
import torch
import time
import torchaudio
from argparse import ArgumentParser
import json
from pathlib import Path


def main(whisper_model, audio_file, output_file, arch, max_tokens=100):
    architecture2module = {
        "whisper": WhisperForConditionalGeneration,
        "medusa": WhisperMedusaModel,
    }
    logging.info(f"Using model: {arch}")
    # Load processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = architecture2module[arch].from_pretrained(whisper_model)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load an example audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=False)

    # resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Preprocess the audio to the required format
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Initialize variables
    processor.tokenizer.set_prefix_tokens(language="en")
    output_ids = processor.tokenizer(
        ""
    ).input_ids  # [d[1] for d in model.config.forced_decoder_ids]
    token_times = []

    model.generation_config.return_dict_in_generate = True
    # Start token-by-token generation
    with torch.no_grad():
        # get encoder output
        n_heads = 10
        max_new_tokens = 1 if arch == "whisper" else n_heads
        extra_kwargs = {} if arch == "whisper" else {"max_steps": 1}
        past_key_values = None
        encoder_output = model.get_encoder()(input_features)
        # todo: consider also timing the encoder part
        start_time = time.time()  # NOTE: we are not timing the encoder!!!
        for _ in range(max_tokens):  # limit the number of generated tokens
            # Get the current input (including previously generated tokens)
            current_input_ids = torch.tensor(output_ids, device=device).unsqueeze(0)
            curr_seq_len = current_input_ids.shape[-1]

            # Generate the next token
            outputs = model.generate(
                encoder_outputs=encoder_output,
                decoder_input_ids=current_input_ids,
                do_sample=False,
                return_dict_in_generate=True,
                num_beams=1,
                max_length=current_input_ids.shape[-1] + max_new_tokens,
                language="en",
                past_key_values=past_key_values,
                **extra_kwargs,
            )

            # Extract the new token
            new_token_ids = outputs.sequences[0, curr_seq_len:].tolist()
            past_key_values = outputs.past_key_values
            output_ids.extend(new_token_ids)

            # Record the time taken to generate this token
            token_time = time.time() - start_time

            # Break if the model generates the end-of-sequence token
            if new_token_ids[-1] == processor.tokenizer.eos_token_id:
                break

            token_times.extend(
                (new_token_id, token_time if i == 0 else token_time + i * 0.001)
                for i, new_token_id in enumerate(new_token_ids)
            )

    # Decode the generated tokens to text
    generated_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)

    # Print the generated text and timing information
    print("Generated Text:", generated_text)
    print("Token Timing Info:")
    for token_id, timing in token_times:
        token_text = processor.tokenizer.decode([token_id])
        print(f"Token: {token_text}, Time: {timing:.4f} seconds")

    print([(processor.tokenizer.decode([t]), s) for t, s in token_times])

    # save the timing info to json
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump([(processor.tokenizer.decode([t]), s) for t, s in token_times], f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--whisper-model",
        type=str,
        required=True,
        help="The path or name of the whisper model",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="The path to the audio file to generate from",
    )
    parser.add_argument(
        "--output-file", type=str, default="./outputs/vanilla_timing.json"
    )
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--arch", type=str, default="whisper", choices=["whisper", "medusa"]
    )
    args = parser.parse_args()
    main(
        args.whisper_model,
        args.audio_file,
        output_file=args.output_file,
        max_tokens=args.max_tokens,
        arch=args.arch,
    )
