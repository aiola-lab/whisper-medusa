import json
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser


def create_image_with_text(
    text,
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

    return img


def create_gif(word_time_pairs, output_path="output.gif"):
    """Create a GIF from a list of (word, time) pairs."""
    frames = []
    durations = []
    current_text = ""

    for word, duration in word_time_pairs:
        current_text += word
        img = create_image_with_text(current_text.strip())  # Adjust font size and path
        frames.append(img)
        durations.append(duration)

    # Save the frames as a GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[int(d * 1000) for d in durations],
        loop=0,
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to the input json file", required=True)
    parser.add_argument("--output-path", type=str, default="outputs/generation.gif", help="Path to the output GIF file")
    args = parser.parse_args()

    with open(args.input_path, "r") as f:
        word_time_pairs = json.load(f)

    create_gif(word_time_pairs, output_path=args.output_path)
