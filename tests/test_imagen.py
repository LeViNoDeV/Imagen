import base64
import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagen import GenerationRequest, ImageGenerator
from imagen.backends import get_backend
from imagen.backends.huggingface import HuggingFaceBackend
from imagen.backends.mock import MockBackend
from imagen.backends.openai_images import OpenAIBackend, closest_size
from imagen.cli import main
from imagen.server import create_app
from imagen.styles import apply_style


def png_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_request_validation():
    with pytest.raises(ValueError):
        GenerationRequest(prompt="   ")
    with pytest.raises(ValueError):
        GenerationRequest(prompt="x", num_images=0)
    assert GenerationRequest(prompt="x", width=1000, height=700).width == 960
    req = GenerationRequest.from_aspect("x", "16:9")
    assert (req.width, req.height) == (1344, 768)
    with pytest.raises(ValueError):
        GenerationRequest.from_aspect("x", "5:1")


def test_apply_style():
    prompt, negative = apply_style("a cat on a sofa.", "dogs", "portrait")
    assert prompt.startswith("a cat on a sofa, professional portrait")
    assert negative.startswith("dogs, cartoon")
    assert apply_style("a cat", "", "none") == ("a cat", "")
    with pytest.raises(ValueError):
        apply_style("a cat", "", "watercolor")


def test_generator_saves_png_with_metadata(tmp_path):
    gen = ImageGenerator("mock", output_dir=tmp_path)
    result = gen.generate(
        GenerationRequest(prompt="red barn", width=256, height=256, seed=0, num_images=2)
    )
    assert [img.seed for img in result.images] == [0, 1]
    paths = [img.path for img in result.images]
    assert len(set(paths)) == 2 and all(p.exists() for p in paths)
    meta = json.loads(Image.open(paths[0]).text["imagen"])
    assert meta["original_prompt"] == "red barn" and meta["seed"] == 0
    assert "photograph" in meta["prompt"]


def test_mock_is_deterministic():
    req = GenerationRequest(prompt="x", width=256, height=256, seed=7)
    a, b = MockBackend().generate(req)[0].image, MockBackend().generate(req)[0].image
    assert a.tobytes() == b.tobytes()


def test_auto_backend_without_any_configured(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "imagen.backends.DiffusersBackend.is_available", classmethod(lambda c: False)
    )
    with pytest.raises(RuntimeError, match="no image backend"):
        get_backend("auto")
    monkeypatch.setenv("HF_TOKEN", "t")
    assert get_backend("auto").name == "huggingface"


def test_huggingface_backend(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=png_bytes())

    backend = HuggingFaceBackend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    images = backend.generate(
        GenerationRequest(prompt="a fox", seed=5, num_images=2, negative_prompt="blur")
    )
    assert [i.seed for i in images] == [5, 6]
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert seen[0].url.path.endswith("/black-forest-labs/FLUX.1-schnell")
    body = json.loads(seen[0].content)
    assert body["inputs"] == "a fox"
    assert body["parameters"]["negative_prompt"] == "blur"


def test_huggingface_error(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    transport = httpx.MockTransport(lambda r: httpx.Response(503, text="loading"))
    backend = HuggingFaceBackend(client=httpx.Client(transport=transport))
    with pytest.raises(RuntimeError, match="503"):
        backend.generate(GenerationRequest(prompt="a fox"))


def test_openai_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    b64 = base64.b64encode(png_bytes()).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["size"] == "1536x1024" and body["n"] == 2
        assert "Avoid: blur" in body["prompt"]
        return httpx.Response(200, json={"data": [{"b64_json": b64}] * 2})

    backend = OpenAIBackend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    req = GenerationRequest.from_aspect("a fox", "16:9", num_images=2, negative_prompt="blur")
    assert len(backend.generate(req)) == 2


def test_closest_size():
    assert closest_size(1024, 1024) == "1024x1024"
    assert closest_size(768, 1344) == "1024x1536"


def test_server(tmp_path):
    client = TestClient(create_app(output_dir=tmp_path))
    assert "Imagen" in client.get("/").text
    opts = client.get("/api/options").json()
    assert "photo" in opts["styles"] and any(b["name"] == "mock" for b in opts["backends"])

    res = client.post(
        "/api/generate",
        json={"prompt": "a lighthouse", "backend": "mock", "aspect": "3:2", "num_images": 2},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert len(data["images"]) == 2
    img = client.get(data["images"][0]["url"])
    assert img.status_code == 200
    assert Image.open(io.BytesIO(img.content)).size == (1216, 832)

    assert client.post("/api/generate", json={"prompt": "x", "backend": "nope"}).status_code == 400
    assert client.post("/api/generate", json={"prompt": ""}).status_code == 422


def test_cli(tmp_path, capsys):
    assert main(["generate", "a red apple", "-b", "mock", "-o", str(tmp_path), "--seed", "1"]) == 0
    assert "saved" in capsys.readouterr().out
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert main(["generate", "x", "-b", "mock", "-n", "99", "-o", str(tmp_path)]) == 1
    assert main(["backends"]) == 0
