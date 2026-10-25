from __future__ import annotations

import argparse
import sys

from imagen.backends import BACKENDS, MODEL_PRESETS
from imagen.core import ASPECT_RATIOS, GenerationError, GenerationRequest
from imagen.styles import STYLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imagen", description="Realistic text-to-image")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate images from a prompt")
    gen.add_argument("prompt")
    gen.add_argument("-b", "--backend", default="auto", choices=["auto", *BACKENDS])
    gen.add_argument("-m", "--model", help=f"model id or local preset ({', '.join(MODEL_PRESETS)})")
    gen.add_argument("-s", "--style", default="photo", choices=list(STYLES))
    gen.add_argument("-a", "--aspect", default="1:1", choices=list(ASPECT_RATIOS))
    gen.add_argument("--negative", default="", help="things to keep out of the image")
    gen.add_argument("-n", "--num-images", type=int, default=1)
    gen.add_argument("--steps", type=int)
    gen.add_argument("--guidance", type=float)
    gen.add_argument("--seed", type=int)
    gen.add_argument("-o", "--output-dir", default="outputs")

    serve = sub.add_parser("serve", help="start the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("-o", "--output-dir", default="outputs")

    sub.add_parser("backends", help="show which backends can run here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "backends":
        for name, cls in BACKENDS.items():
            status = "ready" if cls.is_available() else f"unavailable - {cls.unavailable_reason()}"
            print(f"{name:12} {cls.default_model:40} {status}")
        return 0

    if args.command == "serve":
        import uvicorn

        from imagen.server import create_app

        uvicorn.run(create_app(output_dir=args.output_dir), host=args.host, port=args.port)
        return 0

    from imagen.generator import ImageGenerator

    try:
        request = GenerationRequest.from_aspect(
            args.prompt,
            args.aspect,
            negative_prompt=args.negative,
            num_images=args.num_images,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
        )
        generator = ImageGenerator(args.backend, args.model, args.output_dir)
        print(f"Generating with {generator.backend.name} ({generator.backend.model})...")
        result = generator.generate(request, style=args.style)
    except (ValueError, RuntimeError) as exc:
        kind = "generation failed" if isinstance(exc, GenerationError) else "error"
        print(f"{kind}: {exc}", file=sys.stderr)
        return 1

    for image in result.images:
        print(f"saved {image.path}  (seed {image.seed})")
    print(f"done in {result.elapsed_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
