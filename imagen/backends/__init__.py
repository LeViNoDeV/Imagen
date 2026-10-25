from __future__ import annotations

from imagen.backends.base import Backend
from imagen.backends.diffusers_local import MODEL_PRESETS, DiffusersBackend
from imagen.backends.huggingface import HuggingFaceBackend
from imagen.backends.mock import MockBackend
from imagen.backends.openai_images import OpenAIBackend

# Order matters: "auto" picks the first available backend.
BACKENDS: dict[str, type[Backend]] = {
    b.name: b for b in (DiffusersBackend, HuggingFaceBackend, OpenAIBackend, MockBackend)
}


def get_backend(name: str = "auto", model: str | None = None) -> Backend:
    if name == "auto":
        for cls in BACKENDS.values():
            if cls is not MockBackend and cls.is_available():
                return cls(model)
        hints = "; ".join(
            f"{n}: {c.unavailable_reason()}" for n, c in BACKENDS.items() if c is not MockBackend
        )
        raise RuntimeError(f"no image backend is available ({hints}). Use 'mock' to try the UI.")
    try:
        return BACKENDS[name](model)
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {list(BACKENDS)}") from None


__all__ = ["BACKENDS", "MODEL_PRESETS", "Backend", "get_backend"]
