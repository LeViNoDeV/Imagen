from __future__ import annotations

import random
from abc import ABC, abstractmethod

from imagen.core import GeneratedImage, GenerationRequest


class Backend(ABC):
    """A text-to-image engine. Subclasses turn a request into PIL images."""

    name: str = ""
    default_model: str = ""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or self.default_model

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend can run here (credentials set, packages installed)."""
        return True

    @classmethod
    def unavailable_reason(cls) -> str:
        return ""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> list[GeneratedImage]: ...


def resolve_seeds(request: GenerationRequest) -> list[int]:
    """One seed per image: consecutive from the given seed, or random."""
    base = request.seed if request.seed is not None else random.randrange(2**32)
    return [(base + i) % 2**32 for i in range(request.num_images)]
