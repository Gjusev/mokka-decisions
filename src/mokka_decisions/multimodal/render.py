"""Deterministic UI renderer for the synthetic Mokka Demo Bank app.

Renders phone-sized (360x640) screens from a template id + parameters, using
the DejaVu font bundled with matplotlib (identical on Windows and Kaggle), so
a (template, params) pair produces the same layout everywhere.

The renderer is intentionally primitive — cards, banners, rows of text — but
the *decisive* content (status banners, error codes, amounts, dates) is real
text in the image, which is what OCR and vision backends must recover.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

W, H = 360, 640
BG = (246, 247, 249)
CARD = (255, 255, 255)
INK = (24, 26, 31)
MUTED = (110, 115, 125)
ACCENT = (20, 90, 190)
RED = (196, 43, 43)
GREEN = (22, 125, 72)
AMBER = (191, 131, 12)

_fonts: dict[int, Any] = {}


def _font(size: int):
    if size not in _fonts:
        from PIL import ImageFont

        try:
            import matplotlib.font_manager as fm

            path = fm.findfont("DejaVu Sans")
            _fonts[size] = ImageFont.truetype(path, size)
        except Exception:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def _new_canvas():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def _text(draw, xy, text, size=13, fill=INK, bold=False, anchor=None):
    draw.text(xy, text, font=_font(size + (1 if bold else 0)), fill=fill, anchor=anchor)


def _card(draw, xy, wh, fill=CARD, outline=(228, 230, 234)):
    x, y = xy
    draw.rounded_rectangle((x, y, x + wh[0], y + wh[1]), 10, fill=fill, outline=outline, width=1)


def _chrome(draw, title: str):
    _card(draw, (0, 0), (W, 56), fill=(255, 255, 255))
    _text(draw, (16, 18), "Mokka Demo Bank", size=14, bold=True)
    _text(draw, (W - 16, 20), title, size=11, fill=MUTED, anchor="ra")
    draw.line((0, 56, W, 56), fill=(228, 230, 234), width=1)


def _banner(draw, y: int, text: str, color=RED):
    from PIL import ImageDraw

    # banner spans the width; wrap to at most two lines
    words = text.split()
    lines, cur = [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if len(trial) > 44:
            lines.append(cur)
            cur = w_
        else:
            cur = trial
    lines.append(cur)
    height = 30 + 17 * len(lines)
    draw.rounded_rectangle((12, y, W - 12, y + height), 8, fill=(253, 235, 235), outline=color, width=1)
    ty = y + 10
    for line in lines:
        _text(draw, (24, ty), line, size=12, fill=color)
        ty += 17
    return y + height + 10


def render_screen(template: str, params: dict) -> "object":
    """Render one screen; returns a PIL Image."""
    from PIL import Image, ImageDraw

    img, draw = _new_canvas()
    _chrome(draw, params.get("title", ""))

    if template == "card_status":
        _text(draw, (16, 76), params["headline"], size=16, bold=True)
        _card(draw, (12, 110), (W - 24, 120))
        _text(draw, (28, 128), params["card_name"], size=13)
        _text(draw, (28, 152), params["status_line"], size=12, fill=params["status_color"])
        _text(draw, (28, 176), params["detail"], size=11, fill=MUTED)
        _text(draw, (28, 200), params["ordered"], size=11, fill=MUTED)
        y = _banner(draw, 250, params["banner"], params["banner_color"])
        _text(draw, (16, y + 8), "Support reference", size=11, fill=MUTED)
        _text(draw, (16, y + 26), params["ref"], size=12)

    elif template == "tx_history":
        _text(draw, (16, 76), params["headline"], size=16, bold=True)
        y = _banner(draw, 104, params["banner"], params["banner_color"])
        rows = params["rows"]
        ry = y + 8
        for label, amount, when, color in rows:
            _card(draw, (12, ry), (W - 24, 58))
            _text(draw, (28, ry + 10), label, size=12, bold=True)
            _text(draw, (28, ry + 30), when, size=10, fill=MUTED)
            _text(draw, (W - 28, ry + 18), amount, size=13, fill=color, anchor="ra")
            ry += 66
        _text(draw, (16, ry + 6), params["footnote"], size=10, fill=MUTED)

    elif template == "receipt":
        _text(draw, (16, 76), params["headline"], size=16, bold=True)
        _card(draw, (12, 108), (W - 24, 210))
        fields = params["fields"]  # list of (label, value)
        fy = 126
        for label, value in fields:
            _text(draw, (28, fy), label, size=10, fill=MUTED)
            _text(draw, (W - 28, fy), value, size=12, anchor="ra")
            fy += 26
        y = _banner(draw, 336, params["banner"], params["banner_color"])
        _text(draw, (16, y + 6), params["ref"], size=11, fill=MUTED)

    elif template == "settings":
        # generic screen with no decision-relevant content
        _text(draw, (16, 76), "Settings", size=16, bold=True)
        items = ["Personal details", "Security", "Notifications", "Language", "Theme", "Legal", "Log out"]
        iy = 112
        for item in items:
            _card(draw, (12, iy), (W - 24, 44))
            _text(draw, (28, iy + 14), item, size=12)
            iy += 52

    elif template == "otp_prompt":
        _text(draw, (16, 76), params["headline"], size=16, bold=True)
        _card(draw, (12, 110), (W - 24, 120))
        _text(draw, (28, 130), params["body"], size=12, fill=INK)
        _text(draw, (28, 156), params["hint"], size=11, fill=MUTED)
        for i in range(6):
            draw.rounded_rectangle((28 + i * 52, 190, 68 + i * 52, 230), 8, outline=ACCENT, width=2)
        y = _banner(draw, 250, params["banner"], params["banner_color"])

    elif template == "marketing":
        _text(draw, (16, 76), params["headline"], size=16, bold=True)
        _card(draw, (12, 108), (W - 24, 160), fill=(240, 246, 255))
        _text(draw, (28, 128), params["offer"], size=14, bold=True, fill=ACCENT)
        _text(draw, (28, 156), params["detail"], size=11, fill=INK)
        _text(draw, (28, 182), params["apr"], size=12, fill=GREEN)
        _text(draw, (28, 210), params["cta"], size=10, fill=MUTED)

    else:
        raise KeyError(f"unknown template {template!r}")

    if params.get("overlay_note"):
        _text(draw, (16, H - 34), params["overlay_note"], size=9, fill=MUTED)
    return img


def apply_degradation(img, kind: str, seed: int):
    """Deterministic degradations: blur or partial occlusion (never label info)."""
    import random

    from PIL import ImageFilter

    rng = random.Random(seed)
    if kind == "blur_strong":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.5, 6.0)))
    if kind == "blur_mild":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.6, 2.4)))
    if kind == "lowcontrast":
        from PIL import ImageEnhance

        return ImageEnhance.Contrast(img).enhance(0.45)
    if kind == "none":
        return img
    raise KeyError(f"unknown degradation {kind!r}")


def image_sha256(img) -> str:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def save_png(img, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return str(path)
