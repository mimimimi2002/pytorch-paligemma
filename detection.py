"""Decode PaliGemma's `<loc>` tokens into bounding boxes and draw them.

PaliGemma expresses object detection as ordinary text generation: the prompt
`detect cat` makes the model emit

    <loc0222><loc0113><loc0933><loc0824> cat

The four `<loc>` tokens are the box corners quantised to 1024 bins, in the
order **y_min, x_min, y_max, x_max**. They come out of the same softmax as
words because the tokenizer reserves ids 256000-257023 for them (see the
vocabulary table in the README).

Mapping the bins back to pixels is a plain linear rescale: `process_images`
squashes every image to `image_size` x `image_size` with `resize()`, which does
not preserve the aspect ratio and adds no padding, so bin 0 is always the top
(or left) edge of the *original* image and bin 1023 the bottom (or right) edge.
"""

import re
from typing import List, NamedTuple, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# One detection: four <loc> tokens followed by the label, which runs until the
# next `<loc` or the `;` that separates multiple requested classes.
_DETECTION_RE = re.compile(
    r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^<;]*)"
)

# `<loc>` tokens address 1024 bins per axis.
_NUM_BINS = 1024

# Distinct, reasonably colour-blind-safe outline colours, cycled per label.
_PALETTE = [
    (255, 87, 51),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
]

# Fonts that ship with macOS; falls back to PIL's bitmap font elsewhere.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


class Detection(NamedTuple):
    """A single box in original-image pixel coordinates, plus its label."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str


def parse_detections(text: str, width: int, height: int) -> List[Detection]:
    """Turn the model's decoded text into boxes scaled to `width` x `height`.

    Text without any `<loc>` tokens simply yields an empty list, so this is safe
    to call on captioning or VQA output.
    """
    detections = []
    for match in _DETECTION_RE.finditer(text):
        y_min, x_min, y_max, x_max = (int(v) for v in match.group(1, 2, 3, 4))
        label = match.group(5).strip()
        detections.append(
            Detection(
                x_min=x_min / _NUM_BINS * width,
                y_min=y_min / _NUM_BINS * height,
                x_max=x_max / _NUM_BINS * width,
                y_max=y_max / _NUM_BINS * height,
                label=label,
            )
        )
    return detections


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _colour_for(label: str, seen: List[str]) -> Tuple[int, int, int]:
    """Give every distinct label its own colour, stable across boxes."""
    if label not in seen:
        seen.append(label)
    return _PALETTE[seen.index(label) % len(_PALETTE)]


def draw_detections(
    image: Image.Image, detections: List[Detection]
) -> Image.Image:
    """Return a copy of `image` with each detection outlined and labelled."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    # Scale the line and text with the image so the result is legible whether
    # the input is a thumbnail or a full-resolution photo.
    short_side = min(canvas.size)
    line_width = max(2, round(short_side * 0.006))
    font = _load_font(max(14, round(short_side * 0.04)))

    seen: List[str] = []
    for det in detections:
        colour = _colour_for(det.label, seen)
        draw.rectangle(
            (det.x_min, det.y_min, det.x_max, det.y_max),
            outline=colour,
            width=line_width,
        )

        if not det.label:
            continue

        # Draw the label on a filled strip so it stays readable over the image.
        text_box = draw.textbbox((0, 0), det.label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        pad = line_width * 2

        # Sit the strip above the box, or inside it when there is no room.
        strip_top = det.y_min - text_h - 2 * pad
        if strip_top < 0:
            strip_top = det.y_min
        draw.rectangle(
            (det.x_min, strip_top, det.x_min + text_w + 2 * pad,
             strip_top + text_h + 2 * pad),
            fill=colour,
        )
        draw.text(
            (det.x_min + pad - text_box[0], strip_top + pad - text_box[1]),
            det.label,
            fill=(255, 255, 255),
            font=font,
        )

    return canvas


def save_detections(
    image_file_path: str, decoded_text: str, output_file: str
) -> Optional[str]:
    """Parse `decoded_text`, draw the boxes over the source image and save it.

    Returns the output path, or None when the text held no `<loc>` tokens.
    """
    image = Image.open(image_file_path)
    detections = parse_detections(decoded_text, *image.size)
    if not detections:
        return None

    draw_detections(image, detections).save(output_file)
    return output_file
