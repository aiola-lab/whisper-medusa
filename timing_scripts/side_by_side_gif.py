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
        grouped_word_time_pairs.append((" ".join(words), key))
    return grouped_word_time_pairs


def create_image_with_text(
    text,
    elapsed_time,
    title,
    image_size=(500, 250),  # Increase height to accommodate title
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

    # Draw the title at the top center
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((image_size[0] - title_width) // 2, 10), title, fill="white", font=font)

    # Adjust text drawing starting point after the title
    text_start_y = title_bbox[3] + 20

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
    y_text = text_start_y
    for line in lines:
        draw.text((0, y_text), line, fill="white", font=font)
        bbox = draw.textbbox((0, y_text), line, font=font)
        y_text += bbox[3] - bbox[1] + line_spacing

    # Add the elapsed time counter to the bottom left corner
    elapsed_time_text = f"{elapsed_time:.1f}s"
    bbox = draw.textbbox((0, 0), elapsed_time_text, font=font)
    text_height = bbox[3] - bbox[1]
    draw.text(
        (10, image_size[1] - text_height - 10),
        elapsed_time_text,
        fill="white",
        font=font,
    )

    return img


def create_frames(
    word_time_pairs, font_size, line_spacing, font_path, frame_rate, max_time, title
):
    """Create frames and durations for a sequence of words."""
    frames = []
    durations = []
    current_text = ""
    word_time_pairs = group_words_by_time(word_time_pairs)
    frame_interval = 1 / frame_rate
    total_frames = int(np.ceil(max_time * frame_rate))

    word_idx = 0
    last_elapsed_time = 0
    for i in range(total_frames):
        current_time = i * frame_interval
        while (
            word_idx < len(word_time_pairs)
            and word_time_pairs[word_idx][1] <= current_time
        ):
            current_text += word_time_pairs[word_idx][0] + " "
            last_elapsed_time = word_time_pairs[word_idx][1]
            word_idx += 1
        img = create_image_with_text(
            current_text.strip(),
            elapsed_time=current_time
            if word_idx < len(word_time_pairs)
            else last_elapsed_time,
            title=title,
            font_size=font_size,
            line_spacing=line_spacing,
            font_path=font_path,
        )
        frames.append(img)
        durations.append(1000 // frame_rate)

    return frames, durations


def combine_frames(frames1, frames2, spacer_width=20):
    """Combine two sets of frames side by side, padding the shorter sequence."""
    combined_frames = []
    max_frames = max(len(frames1), len(frames2))
    width = frames1[0].size[0] + frames2[0].size[0] + spacer_width
    height = max(frames1[0].size[1], frames2[0].size[1])

    for i in range(max_frames):
        frame1 = frames1[i] if i < len(frames1) else frames1[-1]
        frame2 = frames2[i] if i < len(frames2) else frames2[-1]

        combined_frame = Image.new("RGB", (width, height), color="black")
        combined_frame.paste(frame1, (0, 0))
        combined_frame.paste(frame2, (frame1.size[0] + spacer_width, 0))

        combined_frames.append(combined_frame)

    return combined_frames


def create_gif(
    word_time_pairs1,
    word_time_pairs2,
    title1,
    title2,
    output_path="output.gif",
    font_size=20,
    line_spacing=10,
    font_path="/System/Library/Fonts/Supplemental/Arial.ttf",
    frame_rate=30,
):
    """Create a side-by-side GIF from two lists of (word, time) pairs."""
    max_time1 = max([time for _, time in word_time_pairs1])
    max_time2 = max([time for _, time in word_time_pairs2])
    max_time = max(max_time1, max_time2)

    frames1, durations1 = create_frames(
        word_time_pairs1,
        font_size,
        line_spacing,
        font_path,
        frame_rate,
        max_time,
        title1,
    )
    frames2, durations2 = create_frames(
        word_time_pairs2,
        font_size,
        line_spacing,
        font_path,
        frame_rate,
        max_time,
        title2,
    )

    combined_frames = combine_frames(frames1, frames2)

    combined_durations = [1000 // frame_rate] * len(combined_frames)

    # Ensure the last frame is held for a bit longer to show the final text
    final_hold_duration = 2 * 1000  # 2 seconds
    combined_durations[-1] += final_hold_duration

    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=combined_durations,
        loop=0,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path-medusa",
        type=str,
        help="Path to the first input json file",
        required=True,
    )
    parser.add_argument(
        "--input-path-baseline",
        type=str,
        help="Path to the second input json file",
        required=True,
    )
    parser.add_argument("--title-medusa", type=str, default="w/ Medusa")
    parser.add_argument("--title-baseline", type=str, default="w/o Medusa")
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

    with open(args.input_path_medusa, "r") as f:
        word_time_pairs1 = json.load(f)

    with open(args.input_path_baseline, "r") as f:
        word_time_pairs2 = json.load(f)

    create_gif(
        word_time_pairs1,
        word_time_pairs2,
        title1=args.title_medusa,
        title2=args.title_baseline,
        output_path=args.output_path,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        font_path=args.font_path,
    )
