"""
LoRA dataset + training pipeline for Z-Image-Turbo (the only model this app
loads LoRAs for, via zimage-full).

Two pieces, both run as jobs through engine.py's existing single-worker queue
so they can never run concurrently with image generation on the same MPS
device:

  - captioning: in-process, loads Qwen3-VL-4B, produces a JSON-structured
    caption per image (pose/framing/clothing/setting/lighting only — no
    identity/face/hair fields, so the trigger token alone anchors identity).
  - training: launches training/train_dreambooth_lora_z_image.py as a
    subprocess (that script's own argparse/global state isn't meant to be
    imported), parses its tqdm progress from stdout.

training/train_dreambooth_lora_z_image.py is HuggingFace's official
DreamBooth LoRA script, copied locally and patched for this Mac: bf16 works
fine here (verified independently) despite the upstream script's blanket
MPS+bf16 rejection, and --train_data_dir was added since passing a bare local
directory to load_dataset() picks the wrong builder and silently drops the
images whenever a metadata.jsonl is also present.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import uuid

LORA_ROOT = os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen-lora")
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "train_dreambooth_lora_z_image.py")
BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CAPTION_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
CAPTION_FIELDS = ["pose", "framing", "clothing", "setting", "lighting"]
CAPTION_PROMPT = (
    "Analyze this image for LoRA training-data captioning. Respond with ONLY a JSON "
    "object, no markdown fences, no extra text, with exactly these keys: "
    '"pose" (body position/action), "framing" (shot type and camera angle), '
    '"clothing" (garments and accessories visible), "setting" (location/background), '
    '"lighting" (light quality and direction). '
    "Do not include any key for hair, face, eyes, skin tone, or any other physical/"
    "identity trait — those are already known and must be omitted entirely."
)

STEP_RE = re.compile(r"Steps:\s*\d+%\|.*?\|\s*(\d+)/(\d+)\s*\[(\d+:\d+)<([\d:?]+).*?loss=([\d.]+)")


# ---------------------------------------------------------------------------
# Dataset storage
# ---------------------------------------------------------------------------


def _dataset_dir(dataset_id):
    d = os.path.join(LORA_ROOT, dataset_id)
    os.makedirs(d, exist_ok=True)
    return d


def _images_dir(dataset_id):
    d = os.path.join(_dataset_dir(dataset_id), "images")
    os.makedirs(d, exist_ok=True)
    return d


def _captions_path(dataset_id):
    return os.path.join(_dataset_dir(dataset_id), "captions.json")


def _lora_path(dataset_id):
    return os.path.join(_dataset_dir(dataset_id), "lora", "pytorch_lora_weights.safetensors")


def create_dataset(files):
    """files: iterable of (filename, bytes). Returns the new dataset id."""
    dataset_id = uuid.uuid4().hex[:10]
    img_dir = _images_dir(dataset_id)
    saved = []
    for name, data in files:
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        safe_name = f"{len(saved)}{ext}"
        with open(os.path.join(img_dir, safe_name), "wb") as f:
            f.write(data)
        saved.append(safe_name)
    if not saved:
        shutil.rmtree(_dataset_dir(dataset_id), ignore_errors=True)
        raise ValueError("No valid images in upload (jpg/jpeg/png/webp/bmp only).")
    return dataset_id


def delete_dataset(dataset_id):
    d = _dataset_dir(dataset_id)
    if os.path.isdir(d):
        shutil.rmtree(d)


def images_dir(dataset_id):
    return _images_dir(dataset_id)


def dataset_images(dataset_id):
    return sorted(
        f for f in os.listdir(_images_dir(dataset_id))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def list_datasets():
    if not os.path.isdir(LORA_ROOT):
        return []
    out = []
    for did in sorted(os.listdir(LORA_ROOT)):
        if not os.path.isdir(os.path.join(LORA_ROOT, did, "images")):
            continue
        lora = _lora_path(did)
        out.append({
            "id": did,
            "num_images": len(dataset_images(did)),
            "captioned": os.path.exists(_captions_path(did)),
            "trained": os.path.exists(lora),
            "lora_path": lora if os.path.exists(lora) else None,
        })
    return out


def get_captions(dataset_id):
    path = _captions_path(dataset_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_captions(dataset_id, captions):
    """captions: {filename: caption text}"""
    with open(_captions_path(dataset_id), "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Captioning job
# ---------------------------------------------------------------------------


def _parse_caption_fields(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return raw.strip()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return raw.strip()
    parts = [str(data[k]).strip().rstrip(".") for k in CAPTION_FIELDS if data.get(k)]
    return ", ".join(parts) if parts else raw.strip()


def run_captioning(job, dataset_id):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    images = dataset_images(dataset_id)
    if not images:
        raise ValueError("Dataset has no images.")
    img_dir = _images_dir(dataset_id)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    job["stage"] = f"Loading {CAPTION_MODEL}"
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(CAPTION_MODEL, dtype=dtype)
    model.to(device)
    model.eval()

    captions = {}
    try:
        for i, name in enumerate(images):
            job["stage"] = f"Captioning {i + 1}/{len(images)}"
            job["progress"] = i / len(images)
            image = Image.open(os.path.join(img_dir, name)).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(device, dtype)
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            captions[name] = _parse_caption_fields(raw)
    finally:
        del model, processor
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    save_captions(dataset_id, captions)
    job["captions"] = captions
    job["progress"] = 1.0


# ---------------------------------------------------------------------------
# Training job
# ---------------------------------------------------------------------------


def run_training(job, dataset_id, trigger, max_steps):
    trigger = trigger.strip()
    if not trigger:
        raise ValueError("Trigger token is required.")
    captions = get_captions(dataset_id)
    if not captions:
        raise ValueError("Dataset has no captions yet — run captioning first.")
    images = dataset_images(dataset_id)
    img_dir = _images_dir(dataset_id)

    with open(os.path.join(img_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for name in images:
            text = f"{trigger}, {captions[name]}" if captions.get(name) else trigger
            f.write(json.dumps({"file_name": name, "text": text}, ensure_ascii=False) + "\n")

    out_dir = os.path.join(_dataset_dir(dataset_id), "lora")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    max_steps = max(1, int(max_steps))
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", BASE_MODEL,
        "--dataset_name", "imagefolder",
        "--train_data_dir", img_dir,
        "--caption_column", "text",
        "--output_dir", out_dir,
        "--instance_prompt", trigger,
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--cache_latents",
        "--resolution", "512",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--optimizer", "adamW",
        "--learning_rate", "1e-4",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(max_steps),
        "--rank", "16",
        "--lora_alpha", "16",
        "--checkpointing_steps", str(max_steps),
        "--seed", "0",
    ]

    job["stage"] = "Loading model"
    job["step"] = 0
    job["total_steps"] = max_steps
    job["loss"] = None
    job["eta"] = None

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    log_tail = []
    buf = b""
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            buf += chunk
            # tqdm updates via \r, not \n — reading by line would buffer every
            # intermediate step until a real newline shows up (rare here).
            while b"\r" in buf or b"\n" in buf:
                idx_r = buf.find(b"\r")
                idx_n = buf.find(b"\n")
                idx = min(x for x in (idx_r, idx_n) if x != -1)
                piece, buf = buf[:idx], buf[idx + 1:]
                text = piece.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                log_tail.append(text)
                del log_tail[:-60]
                m = STEP_RE.search(text)
                if m:
                    step, total, elapsed, eta, loss = m.groups()
                    job["step"] = int(step)
                    job["total_steps"] = int(total)
                    job["loss"] = float(loss)
                    job["eta"] = eta if eta != "?" else None
                    job["progress"] = int(step) / int(total) if int(total) else None
                    job["stage"] = f"Training {step}/{total}"
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Training exited with code {proc.returncode}:\n" + "\n".join(log_tail))

    lora_file = _lora_path(dataset_id)
    if not os.path.exists(lora_file):
        raise RuntimeError("Training finished but no LoRA weights were saved.")
    job["lora_path"] = lora_file
    job["progress"] = 1.0
