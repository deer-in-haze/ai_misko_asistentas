from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Union

import google.generativeai as genai
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "prompt.txt"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def configure_gemini(model_name: str = "gemini-2.5-flash"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def identify_mushroom(
    image_file: Union[str, Path, BinaryIO],
    model_name: str = "gemini-2.5-flash",
) -> str:
    model = configure_gemini(model_name=model_name)
    prompt = load_prompt()

    if isinstance(image_file, (str, Path)):
        image = Image.open(image_file)
    else:
        image = Image.open(image_file)

    response = model.generate_content([prompt, image])

    latin_name = (response.text or "").strip()
    return latin_name
