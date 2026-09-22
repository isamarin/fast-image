#!/usr/bin/env python3
"""
Auto-caption an image folder for LoRA training.

Runs Qwen3-VL over every image in a directory and writes one <name>.txt
caption file per image (same basename, trainer-standard sidecar format),
prefixed with a trigger token so the training tool can bind the character's
identity to that token instead of to the caption's descriptive words.

Usage:
    .venv/bin/python scripts/caption_lora_dataset.py <image_dir> --trigger ohwxgeralt
"""

import argparse
import json
import os
import re
import sys

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FIELDS = ["pose", "framing", "clothing", "setting", "lighting"]

PROMPT = (
    "Analyze this image for LoRA training-data captioning. Respond with ONLY a JSON "
    "object, no markdown fences, no extra text, with exactly these keys: "
    '"pose" (body position/action), "framing" (shot type and camera angle), '
    '"clothing" (garments and accessories visible), "setting" (location/background), '
    '"lighting" (light quality and direction). '
    "Do not include any key for hair, face, eyes, skin tone, or any other physical/"
    "identity trait — those are already known and must be omitted entirely."
)


def _parse_fields(raw):
    """Pull the JSON object out of the model's reply, tolerating ```json fences."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    data = json.loads(match.group(0))
    return [str(data[k]).strip().rstrip(".") for k in FIELDS if data.get(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image_dir", help="Folder of character reference images")
    ap.add_argument("--trigger", default="ohwxgeralt", help="Trigger token prepended to every caption")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="HF repo id of the captioner")
    ap.add_argument("--device", default=None, help="mps/cuda/cpu (default: auto-detect)")
    args = ap.parse_args()

    if not os.path.isdir(args.image_dir):
        sys.exit(f"not a directory: {args.image_dir}")

    images = sorted(
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"no images found in {args.image_dir}")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    load_device = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if load_device == "cpu" else torch.bfloat16

    print(f"Loading {args.model} on {load_device}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    model.to(load_device)
    model.eval()

    written = 0
    for name in images:
        path = os.path.join(args.image_dir, name)
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(args.image_dir, f"{stem}.txt")

        image = Image.open(path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(load_device, dtype)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        try:
            parts = _parse_fields(raw)
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            print(f"  [warn] {name}: couldn't parse JSON ({e}); raw reply: {raw!r}")
            parts = [raw]

        caption = ", ".join(parts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{args.trigger}, {caption}\n")
        print(f"{name} -> {out_path}\n  {args.trigger}, {caption}")
        written += 1

    print(f"\nDone: {written} captions written to {args.image_dir}")


if __name__ == "__main__":
    main()
