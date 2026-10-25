from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from PIL.PngImagePlugin import PngInfo

from imagen.backends import Backend, get_backend
from imagen.core import GenerationRequest, GenerationResult
from imagen.styles import apply_style


def _slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].rstrip("-") or "image"


class ImageGenerator:
    """Applies a realism style to a request, runs a backend, and saves the PNGs."""

    def __init__(
        self,
        backend: Backend | str = "auto",
        model: str | None = None,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.backend = get_backend(backend, model) if isinstance(backend, str) else backend
        self.output_dir = Path(output_dir)

    def generate(
        self, request: GenerationRequest, style: str = "photo", save: bool = True
    ) -> GenerationResult:
        prompt, negative = apply_style(request.prompt, request.negative_prompt, style)
        styled = replace(request, prompt=prompt, negative_prompt=negative)

        start = time.perf_counter()
        images = self.backend.generate(styled)
        result = GenerationResult(
            request=styled,
            backend=self.backend.name,
            model=self.backend.model,
            images=images,
            elapsed_seconds=time.perf_counter() - start,
        )
        if save:
            self._save(result, original_prompt=request.prompt, style=style)
        return result

    def _save(self, result: GenerationResult, original_prompt: str, style: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for index, generated in enumerate(result.images):
            meta = {
                **asdict(result.request),
                "seed": generated.seed,
                "num_images": 1,
                "original_prompt": original_prompt,
                "style": style,
                "backend": result.backend,
                "model": result.model,
            }
            info = PngInfo()
            info.add_text("imagen", json.dumps(meta))
            info.add_text("parameters", result.request.prompt)  # read by common image viewers
            seed = generated.seed if generated.seed is not None else index
            name = f"{stamp}-{_slug(original_prompt)}-{seed}-{uuid.uuid4().hex[:6]}.png"
            path = self.output_dir / name
            generated.image.save(path, pnginfo=info)
            generated.path = path
