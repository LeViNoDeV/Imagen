from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass

from imagen.backends.base import Backend, resolve_seeds
from imagen.core import GeneratedImage, GenerationError, GenerationRequest


@dataclass(frozen=True)
class ModelPreset:
    repo_id: str
    steps: int
    guidance: float
    supports_negative: bool = True


# Short names for well-known photorealistic checkpoints on the Hugging Face Hub.
# Any other Hub repo id or local path also works; it gets the generic defaults.
MODEL_PRESETS: dict[str, ModelPreset] = {
    "flux-schnell": ModelPreset("black-forest-labs/FLUX.1-schnell", 4, 0.0, False),
    "flux-dev": ModelPreset("black-forest-labs/FLUX.1-dev", 28, 3.5, False),
    "realvis-xl": ModelPreset("SG161222/RealVisXL_V5.0", 30, 5.0),
    "sdxl": ModelPreset("stabilityai/stable-diffusion-xl-base-1.0", 30, 6.0),
    "sdxl-turbo": ModelPreset("stabilityai/sdxl-turbo", 4, 0.0),
}
GENERIC_PRESET = ModelPreset("", 30, 6.0)


class DiffusersBackend(Backend):
    """Runs open-weight diffusion models locally with Hugging Face diffusers."""

    name = "local"
    default_model = "realvis-xl"

    # Pipelines are large; share them across backend instances and load each once.
    _pipelines: dict[str, object] = {}
    _lock = threading.Lock()

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model)
        preset = MODEL_PRESETS.get(self.model, GENERIC_PRESET)
        self.preset = preset
        self.repo_id = preset.repo_id or self.model

    @classmethod
    def is_available(cls) -> bool:
        return all(importlib.util.find_spec(m) for m in ("torch", "diffusers"))

    @classmethod
    def unavailable_reason(cls) -> str:
        return 'install the local extras: pip install -e ".[local]"'

    @staticmethod
    def _device_and_dtype():
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _pipeline(self):
        with self._lock:
            if self.repo_id not in self._pipelines:
                from diffusers import AutoPipelineForText2Image

                device, dtype = self._device_and_dtype()
                pipe = AutoPipelineForText2Image.from_pretrained(self.repo_id, torch_dtype=dtype)
                if device == "cuda":
                    # Keeps VRAM use manageable for FLUX-sized models on consumer GPUs.
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(device)
                self._pipelines[self.repo_id] = pipe
            return self._pipelines[self.repo_id]

    def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        import torch

        if not self.is_available():
            raise GenerationError(self.unavailable_reason())
        try:
            pipe = self._pipeline()
        except Exception as exc:  # network, auth for gated repos, bad repo id...
            raise GenerationError(f"could not load model {self.repo_id!r}: {exc}") from exc

        kwargs = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps or self.preset.steps,
            "guidance_scale": (
                request.guidance if request.guidance is not None else self.preset.guidance
            ),
        }
        if self.preset.supports_negative and request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt

        results = []
        # One call per seed so each image is individually reproducible.
        for seed in resolve_seeds(request):
            generator = torch.Generator("cpu").manual_seed(seed)
            with self._lock:
                image = pipe(generator=generator, **kwargs).images[0]
            results.append(GeneratedImage(image=image, seed=seed))
        return results
