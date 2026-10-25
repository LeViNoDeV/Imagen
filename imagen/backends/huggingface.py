from __future__ import annotations

import io
import os

import httpx
from PIL import Image

from imagen.backends.base import Backend, resolve_seeds
from imagen.core import GeneratedImage, GenerationError, GenerationRequest

DEFAULT_BASE_URL = "https://router.huggingface.co/hf-inference/models"


class HuggingFaceBackend(Backend):
    """Hosted inference through the Hugging Face Inference API (needs HF_TOKEN)."""

    name = "huggingface"
    default_model = "black-forest-labs/FLUX.1-schnell"

    def __init__(self, model: str | None = None, client: httpx.Client | None = None) -> None:
        super().__init__(model)
        self.base_url = os.environ.get("IMAGEN_HF_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.client = client or httpx.Client(timeout=300)

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.environ.get("HF_TOKEN"))

    @classmethod
    def unavailable_reason(cls) -> str:
        return "set HF_TOKEN to a Hugging Face access token"

    def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise GenerationError(self.unavailable_reason())

        results = []
        for seed in resolve_seeds(request):
            parameters = {"width": request.width, "height": request.height, "seed": seed}
            if request.negative_prompt:
                parameters["negative_prompt"] = request.negative_prompt
            if request.steps:
                parameters["num_inference_steps"] = request.steps
            if request.guidance is not None:
                parameters["guidance_scale"] = request.guidance

            response = self.client.post(
                f"{self.base_url}/{self.model}",
                headers={"Authorization": f"Bearer {token}", "Accept": "image/png"},
                json={"inputs": request.prompt, "parameters": parameters},
            )
            if response.status_code != 200:
                raise GenerationError(
                    f"Hugging Face returned {response.status_code}: {response.text[:300]}"
                )
            try:
                image = Image.open(io.BytesIO(response.content))
                image.load()
            except Exception as exc:
                raise GenerationError("Hugging Face response was not an image") from exc
            results.append(GeneratedImage(image=image.convert("RGB"), seed=seed))
        return results
