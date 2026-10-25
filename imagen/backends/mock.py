from __future__ import annotations

import random

from PIL import Image, ImageDraw

from imagen.backends.base import Backend, resolve_seeds
from imagen.core import GeneratedImage, GenerationRequest


class MockBackend(Backend):
    """Deterministic placeholder images. For tests, demos, and UI work without a model."""

    name = "mock"
    default_model = "gradient"

    def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        results = []
        for seed in resolve_seeds(request):
            rng = random.Random(f"{request.prompt}:{seed}")
            top = tuple(rng.randrange(256) for _ in range(3))
            bottom = tuple(rng.randrange(256) for _ in range(3))
            image = Image.new("RGB", (request.width, request.height))
            draw = ImageDraw.Draw(image)
            for y in range(request.height):
                t = y / max(request.height - 1, 1)
                color = tuple(round(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
                draw.line([(0, y), (request.width, y)], fill=color)
            draw.text((16, 16), request.prompt[:80], fill=(255, 255, 255))
            results.append(GeneratedImage(image=image, seed=seed))
        return results
