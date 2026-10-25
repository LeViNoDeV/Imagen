# Imagen

Generate realistic, photographic images from text. It comes with a web UI, a CLI and a Python API. You can run open-weight diffusion models on your own GPU or use a hosted API.

```bash
pip install -e .                 # web UI, CLI, and the hosted-API backends
pip install -e ".[local]"        # + PyTorch/diffusers to run models on your own GPU

imagen serve                     # open http://127.0.0.1:8000
imagen generate "an elderly fisherman mending nets on a wooden dock at dawn" -a 3:2 -s portrait
```

## Backends

| Backend       | Runs on                         | Setup                          | Default model                       |
|---------------|---------------------------------|--------------------------------|-------------------------------------|
| `local`       | Your GPU (CUDA or Apple MPS)    | `pip install -e ".[local]"`    | `realvis-xl` (RealVisXL V5.0)       |
| `huggingface` | Hugging Face Inference API      | `export HF_TOKEN=hf_...`       | `black-forest-labs/FLUX.1-schnell`  |
| `openai`      | OpenAI Images API               | `export OPENAI_API_KEY=sk-...` | `gpt-image-1`                       |
| `mock`        | Anywhere; draws gradients       | none                           | for tests and UI work               |

`--backend auto` (the default) picks the first backend in the table that is ready. `imagen backends` shows which ones are ready on your machine.

The local backend accepts these short model names and fills in good step and guidance defaults:

| Preset         | Hub repo                                   | Notes |
|----------------|--------------------------------------------|-------|
| `realvis-xl`   | `SG161222/RealVisXL_V5.0`                  | Photorealistic SDXL fine-tune, ~12 GB VRAM at 1024px |
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell`         | Very high quality in 4 steps; large (~24 GB weights, CPU offload enabled) |
| `flux-dev`     | `black-forest-labs/FLUX.1-dev`             | Best quality; gated repo and non-commercial license; run `huggingface-cli login` first |
| `sdxl`         | `stabilityai/stable-diffusion-xl-base-1.0` | Base SDXL |
| `sdxl-turbo`   | `stabilityai/sdxl-turbo`                   | Fast drafts (try `-a 1:1 --steps 4`) |

You can also pass any other diffusers-compatible Hub repo id or local path with `-m`. Weights download on first use and are cached in `~/.cache/huggingface`. CPU works but is very slow.

## Getting realistic results

Every prompt goes through a **style** preset. The preset adds camera, lens and lighting terms and a negative prompt that steers away from cartoon, CGI and anatomy artifacts:

`photo` (default), `portrait`, `cinematic`, `landscape`, `product`, `street`, `none` (the prompt is used exactly as you wrote it).

Tips:
- Describe the subject, setting, lighting and time of day concretely. For example, "overcast afternoon, soft diffuse light" works better than "nice lighting".
- Put the important things first. SD/SDXL models read only the first ~77 tokens, and the style terms are appended after your prompt.
- Lock `--seed` to iterate on a prompt while keeping the composition. With `-n`, the images use consecutive seeds.
- FLUX models ignore negative prompts. With them, describe what you *do* want.

## CLI

```
imagen generate PROMPT [-b BACKEND] [-m MODEL] [-s STYLE] [-a ASPECT] [-n N]
                       [--negative TEXT] [--steps N] [--guidance G] [--seed S] [-o DIR]
imagen serve [--host H] [--port P] [-o DIR]
imagen backends
```

Aspect ratios: `1:1 4:3 3:4 3:2 2:3 16:9 9:16`, each at about one megapixel. Images are saved as PNG in `outputs/`. Each PNG stores the full generation settings (prompt, seed, model and so on) as JSON in its `imagen` text chunk, so any image can be reproduced.

## Python

```python
from imagen import GenerationRequest, ImageGenerator

gen = ImageGenerator(backend="local", model="flux-schnell")
result = gen.generate(
    GenerationRequest.from_aspect("a red vintage bicycle against a brick wall", "3:2", seed=42)
)
print(result.images[0].path)
```

## HTTP API

`imagen serve` exposes:
- `GET /api/options`: the backends (and whether each is ready), styles, aspect ratios and local model presets.
- `POST /api/generate`: `{"prompt": "...", "backend": "auto", "style": "photo", "aspect": "1:1", "num_images": 1, "seed": 42}`. It returns image URLs under `/outputs/`.

The server has no authentication. Keep it on localhost, or put it behind a proxy that has auth before you expose it.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```
