"""Prompt presets that steer models toward photographic realism."""

from __future__ import annotations

from dataclasses import dataclass

REALISM_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, 3d render, cgi, plastic skin, "
    "oversaturated, blurry, low resolution, jpeg artifacts, watermark, text, logo, "
    "deformed, disfigured, extra fingers, extra limbs, bad anatomy, bad hands"
)


@dataclass(frozen=True)
class Style:
    name: str
    description: str
    suffix: str
    negative: str = REALISM_NEGATIVE


STYLES: dict[str, Style] = {
    s.name: s
    for s in [
        Style(
            "photo",
            "Natural, true-to-life photograph",
            "photograph, shot on a full-frame DSLR, 35mm lens, natural lighting, "
            "realistic textures, sharp focus, high dynamic range, highly detailed",
        ),
        Style(
            "portrait",
            "Studio-quality portrait with realistic skin",
            "professional portrait photograph, 85mm lens, f/1.8, shallow depth of field, "
            "soft key light, detailed skin texture with pores, catchlights in the eyes",
        ),
        Style(
            "cinematic",
            "Film still with dramatic lighting",
            "cinematic film still, anamorphic lens, dramatic lighting, color graded, "
            "shallow depth of field, film grain, 35mm film",
        ),
        Style(
            "landscape",
            "Wide nature or city photograph",
            "landscape photograph, wide-angle lens, golden hour, f/11, crisp detail, "
            "rich natural colors, atmospheric perspective",
        ),
        Style(
            "product",
            "Clean commercial product shot",
            "commercial product photograph, studio softbox lighting, seamless background, "
            "crisp reflections, macro detail, advertising quality",
        ),
        Style(
            "street",
            "Candid documentary photograph",
            "candid street photograph, 28mm lens, available light, documentary style, "
            "authentic moment, subtle film grain",
        ),
        Style("none", "Use the prompt exactly as written", "", ""),
    ]
}


def apply_style(prompt: str, negative_prompt: str, style: str) -> tuple[str, str]:
    """Return (prompt, negative_prompt) with the style's modifiers merged in."""
    try:
        preset = STYLES[style]
    except KeyError:
        raise ValueError(f"unknown style {style!r}; choose from {list(STYLES)}") from None
    styled_prompt = f"{prompt.rstrip(' ,.')}, {preset.suffix}" if preset.suffix else prompt
    negatives = [n for n in (negative_prompt.strip(), preset.negative) if n]
    return styled_prompt, ", ".join(negatives)
