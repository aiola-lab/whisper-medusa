import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
from itertools import groupby


def group_words_by_time(word_time_pairs):
    """Group words by their timestamps."""
    grouped_word_time_pairs = []
    for key, group in groupby(word_time_pairs, lambda x: round(x[1], 2)):
        words = [item[0] for item in group]
        grouped_word_time_pairs.append(("".join(words), key))
    return grouped_word_time_pairs


def create_image_with_text(
    text,
    elapsed_time,
    image_size=(500, 200),
    font_size=20,
    line_spacing=10,
    font_path="/System/Library/Fonts/Supplemental/Arial.ttf",
):
    """Create an image with the given text and desired font size using a TrueType font."""
    # Create a new image with a black background
    img = Image.new("RGB", image_size, color="black")
    draw = ImageDraw.Draw(img)

    # Load the TrueType font with the desired size
    font = ImageFont.truetype(font_path, font_size)

    # Break text into lines that fit within the image width
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Test adding the new word to the current line
        test_line = current_line + word + " "
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width < image_size[0]:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "

    if current_line:
        lines.append(current_line)

    # Draw each line on the final image
    y_text = 0
    for line in lines:
        draw.text((0, y_text), line, fill="white", font=font)
        bbox = draw.textbbox((0, y_text), line, font=font)
        y_text += bbox[3] - bbox[1] + line_spacing

    # Add the elapsed time counter to the bottom left corner
    elapsed_time_text = f"{elapsed_time:.1f}s"
    bbox = draw.textbbox((0, 0), elapsed_time_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        (10, image_size[1] - text_height - 10),
        elapsed_time_text,
        fill="white",
        font=font,
    )

    return img


def create_gif(
    word_time_pairs,
    output_path="output.gif",
    font_size=20,
    line_spacing=10,
    font_path="/System/Library/Fonts/Supplemental/Arial.ttf",
    frame_rate=30,
):
    """Create a GIF from a list of (word, time) pairs."""
    frames = []
    durations = []
    current_text = ""
    previous_time = 0
    # group words by time
    word_time_pairs = group_words_by_time(word_time_pairs)
    for i, (word, current_time) in enumerate(word_time_pairs):
        current_text += word
        img = create_image_with_text(
            current_text.strip(),
            elapsed_time=current_time,
            font_size=font_size,
            line_spacing=line_spacing,
            font_path=font_path,
        )

        if i == 0:
            duration = current_time  # First frame duration
        else:
            duration = current_time - previous_time

        num_frames = int(np.round(duration * frame_rate))

        for _ in range(num_frames):
            frames.append(img)
            durations.append(1000 // frame_rate)

        # durations.append(duration)
        previous_time = current_time

    # Save the frames as a GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to the input json file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/generation.gif",
        help="Path to the output GIF file",
    )
    parser.add_argument(
        "--font-size", type=int, default=20, help="Font size for the text in the GIF"
    )
    parser.add_argument(
        "--line-spacing", type=int, default=10, help="Spacing between lines in the GIF"
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default="/System/Library/Fonts/Supplemental/Arial.ttf",
        help="Path to the TrueType font file",
    )
    args = parser.parse_args()

    with open(args.input_path, "r") as f:
        word_time_pairs = json.load(f)

    create_gif(
        word_time_pairs,
        output_path=args.output_path,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        font_path=args.font_path,
    )
