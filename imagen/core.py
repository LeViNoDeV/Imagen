from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# Width x height presets, all multiples of 64 and around one megapixel,
# which is what SDXL/FLUX-class models are trained on.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
}


class GenerationError(RuntimeError):
    """Raised when a backend fails to produce an image."""


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int | None = None  # None -> backend/model default
    guidance: float | None = None  # None -> backend/model default
    seed: int | None = None  # None -> random
    num_images: int = 1

    def __post_init__(self) -> None:
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not 1 <= self.num_images <= 8:
            raise ValueError("num_images must be between 1 and 8")
        for name in ("width", "height"):
            value = getattr(self, name)
            if not 256 <= value <= 2048:
                raise ValueError(f"{name} must be between 256 and 2048")
            # Diffusion latents need dimensions divisible by 8; round down to 64 to be safe.
            setattr(self, name, value - value % 64)

    @classmethod
    def from_aspect(cls, prompt: str, aspect: str = "1:1", **kwargs) -> GenerationRequest:
        if aspect not in ASPECT_RATIOS:
            raise ValueError(f"unknown aspect ratio {aspect!r}; choose from {list(ASPECT_RATIOS)}")
        width, height = ASPECT_RATIOS[aspect]
        return cls(prompt=prompt, width=width, height=height, **kwargs)


@dataclass
class GeneratedImage:
    image: Image.Image
    seed: int | None
    path: Path | None = None


@dataclass
class GenerationResult:
    request: GenerationRequest
    backend: str
    model: str
    images: list[GeneratedImage] = field(default_factory=list)
    elapsed_seconds: float = 0.0
