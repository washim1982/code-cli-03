"""
Minimal vision client for a local Ollama model.

Kept separate from `backends.py` because the requirements are peculiar to this
job and were established by measurement, not by reading documentation:

  * **`think: false` is mandatory.** `gemma4:latest` is a reasoning model. Asked
    to describe a screenshot it returned `response=''` with `eval_count=250` and
    `done_reason='length'` — it had spent the entire budget reasoning and never
    emitted an answer. The same call with `think: false` answers in 6s.
  * **An empty response is a failure, not an empty finding.** Silently treating
    it as "no defects" would report a broken page as clean.
  * **The model sometimes forgets it was sent an image**, replying "please
    provide the picture". That is detected and treated as a failed call.

The model is deliberately a replaceable claim generator: everything it says is
cross-checked against DOM geometry before a user sees it, so a better model
improves the output without changing the pipeline.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

import httpx

__all__ = ["VisionClient", "VisionUnavailable", "detect_vision_model",
           "DEFAULT_CANDIDATES"]

DEFAULT_URL = "http://localhost:11434"

#: Probed in order. The first that accepts an image wins.
DEFAULT_CANDIDATES = ("gemma4:latest", "llava:latest", "llama3.2-vision:latest",
                      "qwen2.5vl:latest", "minicpm-v:latest", "moondream:latest")

#: A 1x1 PNG — the cheapest possible "do you accept images" probe.
_PROBE_PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
    "7753de0000000c4944415408d763f8cfc000000301010018dd8db0000000"
    "0049454e44ae426082")).decode()

#: Replies that mean the image did not arrive, whatever the model believes.
_NO_IMAGE = re.compile(
    r"(provide|share|upload|attach|send).{0,25}(image|picture|screenshot)"
    r"|i (cannot|can't|don't) see (an? )?(image|picture)", re.I)


class VisionUnavailable(RuntimeError):
    """No local model accepted an image, or the one that did returned nothing."""


@dataclass(frozen=True)
class VisionReply:
    text: str
    model: str
    duration_s: float


def _accepts_images(model: str, base_url: str, timeout: float) -> bool:
    try:
        r = httpx.post(f"{base_url}/api/generate", timeout=timeout, json={
            "model": model, "prompt": "Reply with the word ok.",
            "images": [_PROBE_PNG], "stream": False, "think": False,
            "options": {"num_predict": 8},
        })
    except httpx.HTTPError:
        return False
    # A text-only model answers 400 with "model does not support it".
    return r.status_code == 200


def detect_vision_model(base_url: str = DEFAULT_URL,
                        candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
                        timeout: float = 60.0) -> str | None:
    """First installed model that accepts an image, or None."""
    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=10.0).json()
        installed = {m["name"] for m in tags.get("models", [])}
    except (httpx.HTTPError, ValueError, KeyError):
        return None

    ordered = [m for m in candidates if m in installed]
    # Anything with a vision-ish name that is not in the candidate list.
    ordered += sorted(m for m in installed - set(ordered)
                      if re.search(r"vision|vl|llava|moondream", m, re.I))
    for model in ordered:
        if _accepts_images(model, base_url, timeout):
            return model
    return None


class VisionClient:
    """One local vision model, with the workarounds it needs."""

    def __init__(self, model: str, *, base_url: str = DEFAULT_URL,
                 timeout: float = 300.0, num_predict: int = 700) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_predict = num_predict

    def ask(self, png: bytes, prompt: str, *, as_json: bool = False) -> VisionReply:
        """
        Send one image and a prompt. Raises `VisionUnavailable` on a dud reply.

        `think` is forced off: with it on, this model family spends the whole
        token budget reasoning and returns an empty string.
        """
        import time

        body = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(png).decode()],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_predict": self.num_predict},
        }
        if as_json:
            body["format"] = "json"

        started = time.time()
        try:
            response = httpx.post(f"{self.base_url}/api/generate",
                                  json=body, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise VisionUnavailable(f"{self.model}: {str(exc)[:120]}") from exc

        text = (data.get("response") or "").strip()
        if not text:
            raise VisionUnavailable(
                f"{self.model} returned nothing "
                f"(done_reason={data.get('done_reason')}, "
                f"eval_count={data.get('eval_count')})")
        if _NO_IMAGE.search(text):
            raise VisionUnavailable(
                f"{self.model} did not register the image it was sent")
        return VisionReply(text=text, model=self.model,
                           duration_s=time.time() - started)

    def ask_json(self, png: bytes, prompt: str) -> list[dict]:
        """
        Ask for a JSON array of claims. Returns [] rather than raising on shape
        problems — a malformed answer is no evidence, not a crash.
        """
        reply = self.ask(png, prompt, as_json=True)
        try:
            parsed = json.loads(reply.text)
        except json.JSONDecodeError:
            match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", reply.text)
            if not match:
                return []
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                return []

        if isinstance(parsed, dict):
            for key in ("findings", "defects", "problems", "issues", "items"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]
