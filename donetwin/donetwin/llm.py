"""Crusoe Managed Inference client (OpenAI-compatible) with retries and usage accounting."""
import base64
import json
import os
import random
import re
import threading
import time

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B")  # Crusoe /models id
VISION_MODEL = os.getenv("VISION_MODEL", "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B")

# USD per 1M tokens (input, output), from the hackathon model table.
PRICES = {
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": (0.30, 2.40),
    "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B": (0.30, 1.83),
    "nvidia/Nemotron-3-Nano-30B-A3B": (0.05, 0.20),
    "nvidia/nemotron-3.5-lightning-30b-a3b": (0.05, 0.20),
    "google/gemma-4-31b-it": (0.14, 0.40),
}


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

    def add(self, model, pt, ct):
        pin, pout = PRICES.get(model, (0.3, 2.4))
        with self.lock:
            self.calls += 1
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.cost += pt / 1e6 * pin + ct / 1e6 * pout

    def to_dict(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cost_usd": round(self.cost, 4)}


class LLM:
    def __init__(self, usage=None):
        from openai import OpenAI
        key = os.getenv("CRUSOE_API_KEY")
        if not key:
            raise RuntimeError("CRUSOE_API_KEY is not set (put it in .env)")
        self.client = OpenAI(base_url=BASE_URL, api_key=key, timeout=180)
        self.usage = usage or Usage()

    def chat(self, model, messages, max_tokens=4096, temperature=0.4, retries=5, extra_body=None):
        delay = 2.0
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                    extra_body=extra_body)
                if r.usage:
                    self.usage.add(model, r.usage.prompt_tokens or 0, r.usage.completion_tokens or 0)
                msg = r.choices[0].message
                text = msg.content or ""
                if not text.strip():
                    # reasoning models sometimes spend the whole budget thinking
                    text = getattr(msg, "reasoning_content", None) or ""
                return strip_think(text)
            except Exception as e:
                status = getattr(e, "status_code", None)
                if attempt == retries - 1 or status in (400, 401, 403, 404):
                    raise
                time.sleep(delay + random.random())
                delay *= 2


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def image_part(png_path):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def extract_json(text):
    """First parseable JSON object in a model reply (fenced or bare)."""
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in model reply: {text[:300]}")


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()
