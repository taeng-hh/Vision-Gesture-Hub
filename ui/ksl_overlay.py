"""Render Korean subtitles with Pillow and a locally available Hangul font."""
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def get_font():
    candidates = [os.environ.get("KSL_FONT", ""), "C:/Windows/Fonts/malgun.ttf",
                  "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path, 18)
    raise RuntimeError("Hangul font missing: set KSL_FONT to a Korean .ttf/.ttc font")


def draw_ksl_overlay(frame, lines):
    image = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    font = get_font()
    wrapped = []
    for line in lines:
        current = ""
        for char in line:
            if current and draw.textlength(current + char, font=font) > image.width - 24:
                wrapped.append(current)
                current = ""
            current += char
        wrapped.append(current)
    wrapped = wrapped[-max(1, (image.height - 160) // 26):]
    y = image.height - len(wrapped) * 26 - 8
    draw.rectangle((0, y - 4, image.width, image.height), fill=(20, 20, 20))
    for line in wrapped:
        draw.text((10, y), line, font=font, fill=(245, 245, 245))
        y += 26
    frame[:] = np.asarray(image)[:, :, ::-1]
    return frame
