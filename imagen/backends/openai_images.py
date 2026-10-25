from __future__ import annotations

import base64
import io
import os

import httpx
from PIL import Image

from imagen.backends.base import Backend
from imagen.core import GeneratedImage, GenerationError, GenerationRequest

API_URL = "https://api.openai.com/v1/images/generations"


def closest_size(width: int, height: int) -> str:
    """gpt-image models accept only square, landscape and portrait sizes."""
    ratio = width / height
    if ratio > 1.2:
        return "1536x1024"
    if ratio < 1 / 1.2:
        return "1024x1536"
    return "1024x1024"


class OpenAIBackend(Backend):
    """OpenAI image generation API (needs OPENAI_API_KEY). Seeds are not supported."""

    name = "openai"
    default_model = "gpt-image-1"

    def __init__(self, model: str | None = None, client: httpx.Client | None = None) -> None:
        super().__init__(model)
        self.client = client or httpx.Client(timeout=300)

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    @classmethod
    def unavailable_reason(cls) -> str:
        return "set OPENAI_API_KEY"

    def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise GenerationError(self.unavailable_reason())

        # No negative-prompt parameter, so state the exclusions in the prompt itself.
        prompt = request.prompt
        if request.negative_prompt:
            prompt += f"\n\nAvoid: {request.negative_prompt}."

        response = self.client.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": self.model,
                "prompt": prompt,
                "n": request.num_images,
                "size": closest_size(request.width, request.height),
                "quality": "high",
            },
        )
        if response.status_code != 200:
            raise GenerationError(f"OpenAI returned {response.status_code}: {response.text[:300]}")

        results = []
        for item in response.json().get("data", []):
            image = Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
            image.load()
            results.append(GeneratedImage(image=image.convert("RGB"), seed=None))
        if not results:
            raise GenerationError("OpenAI returned no images")
        return results
