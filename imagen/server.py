from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from imagen.backends import BACKENDS, MODEL_PRESETS, Backend, get_backend
from imagen.core import ASPECT_RATIOS, GenerationError, GenerationRequest
from imagen.generator import ImageGenerator
from imagen.styles import STYLES

STATIC_DIR = Path(__file__).parent / "static"


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    backend: str = "auto"
    model: str | None = None
    style: str = "photo"
    aspect: str = "1:1"
    num_images: int = Field(default=1, ge=1, le=4)
    steps: int | None = Field(default=None, ge=1, le=150)
    guidance: float | None = Field(default=None, ge=0, le=30)
    seed: int | None = Field(default=None, ge=0, lt=2**32)


def create_app(output_dir: str | Path = "outputs") -> FastAPI:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Imagen")
    backends: dict[tuple[str, str | None], Backend] = {}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/options")
    def options() -> dict:
        return {
            "backends": [
                {
                    "name": name,
                    "default_model": cls.default_model,
                    "available": cls.is_available(),
                    "reason": "" if cls.is_available() else cls.unavailable_reason(),
                }
                for name, cls in BACKENDS.items()
            ],
            "local_models": list(MODEL_PRESETS),
            "styles": {name: s.description for name, s in STYLES.items()},
            "aspects": list(ASPECT_RATIOS),
        }

    # Plain def: FastAPI runs it in a worker thread, so slow generations don't block the loop.
    @app.post("/api/generate")
    def generate(body: GenerateBody) -> dict:
        try:
            key = (body.backend, body.model or None)
            if key not in backends:
                backends[key] = get_backend(*key)
            request = GenerationRequest.from_aspect(
                body.prompt,
                body.aspect,
                negative_prompt=body.negative_prompt,
                num_images=body.num_images,
                steps=body.steps,
                guidance=body.guidance,
                seed=body.seed,
            )
            generator = ImageGenerator(backends[key], output_dir=output_dir)
            result = generator.generate(request, style=body.style)
        except GenerationError as exc:
            raise HTTPException(502, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

        return {
            "backend": result.backend,
            "model": result.model,
            "prompt": result.request.prompt,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "images": [
                {"url": f"/outputs/{img.path.name}", "seed": img.seed} for img in result.images
            ],
        }

    app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")
    return app
