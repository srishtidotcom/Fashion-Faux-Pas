#!/usr/bin/env python3
"""Generate dense factual image captions with Florence-2.

Input:
    data/raw/images/

Output:
    data/processed/captions.json

The script is resumable, batch-oriented, and skips corrupted images safely.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageFile
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# Allow truncated files to fail less aggressively; corrupted files are still skipped.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "microsoft/Florence-2-large"
INPUT_DIR = Path("data/raw/images")
OUTPUT_FILE = Path("data/processed/captions.json")

CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
DEFAULT_BATCH_SIZE = 8
DEFAULT_SAVE_EVERY = 50
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_BEAMS = 3
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Florence-2 image captions.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Directory of input images.")
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE, help="Path to captions.json.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Images per batch.")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save after N new captions.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Hugging Face Florence-2 model name.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens to generate.")
    parser.add_argument("--num-beams", type=int, default=DEFAULT_NUM_BEAMS, help="Beam search width.")
    parser.add_argument("--length-penalty", type=float, default=DEFAULT_LENGTH_PENALTY, help="Generation length penalty.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="Penalty to reduce repetitive captions.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
        help="Prevent repeated n-grams in generation.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def detect_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(device: torch.device) -> torch.dtype:
    """Choose a safe dtype for the selected device."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_existing_captions(output_file: Path) -> Dict[str, str]:
    """Load an existing captions JSON file if present."""
    if not output_file.exists():
        return {}

    try:
        with output_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to read existing captions file %s: %s", output_file, exc)

    return {}


def save_captions_atomic(output_file: Path, captions: Dict[str, str]) -> None:
    """Atomically write captions to disk."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False, sort_keys=True)

    tmp_file.replace(output_file)


def list_image_paths(input_dir: Path) -> List[Path]:
    """Return sorted image paths from the input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(image_path: Path) -> Image.Image:
    """Load an RGB image, raising on corruption or invalid files."""
    with Image.open(image_path) as img:
        img.load()
        return img.convert("RGB")


def chunked(items: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    """Yield fixed-size batches from a sequence."""
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def normalize_caption(parsed: Any, fallback: str = "") -> str:
    """Convert Florence-2 post-processed output into a clean caption string."""
    if isinstance(parsed, str):
        text = parsed
    elif isinstance(parsed, dict):
        if parsed:
            value = parsed.get(CAPTION_TASK, next(iter(parsed.values())))
            text = normalize_caption(value, fallback=fallback)
        else:
            text = fallback
    elif isinstance(parsed, (list, tuple)):
        text = " ".join(normalize_caption(item, fallback="") for item in parsed).strip()
    else:
        text = fallback

    text = " ".join(text.split()).strip()
    return text


def generate_batch_captions(
    processor: Any,
    model: Any,
    device: torch.device,
    image_paths: Sequence[Path],
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> Dict[str, str]:
    """Generate captions for a batch of images.

    Corrupted images are skipped gracefully.
    """
    valid_images: List[Image.Image] = []
    valid_paths: List[Path] = []

    for image_path in image_paths:
        try:
            valid_images.append(load_image(image_path))
            valid_paths.append(image_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping corrupted/unreadable image %s: %s", image_path.name, exc)

    if not valid_images:
        return {}

    inputs = processor(
        text=[CAPTION_TASK] * len(valid_images),
        images=valid_images,
        return_tensors="pt",
    )
    
    # Move inputs to device and cast pixel_values to the model's expected dtype (e.g., float16)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        # Match model's precision (model.dtype)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": False,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)

    batch_captions: Dict[str, str] = {}
    for image_path, image, generated_text in zip(valid_paths, valid_images, generated_texts):
        try:
            parsed = processor.post_process_generation(
                generated_text,
                task=CAPTION_TASK,
                image_size=image.size,
            )
            caption = normalize_caption(parsed)
            if caption:
                batch_captions[image_path.name] = caption
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to post-process caption for %s: %s", image_path.name, exc)

    return batch_captions


def main() -> None:
    """Run the captioning pipeline."""
    setup_logging()
    args = parse_args()

    device = detect_device()
    dtype = get_torch_dtype(device)

    logging.info("Loading processor and model: %s", args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    existing_captions = load_existing_captions(args.output_file)
    all_image_paths = list_image_paths(args.input_dir)
    pending_paths = [p for p in all_image_paths if p.name not in existing_captions]

    logging.info(
        "Found %d images, %d already captioned, %d pending.",
        len(all_image_paths),
        len(existing_captions),
        len(pending_paths),
    )

    captions = dict(existing_captions)
    new_captions_since_save = 0

    progress = tqdm(total=len(pending_paths), desc="Captioning images", unit="image")
    for batch_paths in chunked(pending_paths, args.batch_size):
        batch_captions = generate_batch_captions(
            processor=processor,
            model=model,
            device=device,
            image_paths=batch_paths,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        captions.update(batch_captions)
        new_captions_since_save += len(batch_captions)
        progress.update(len(batch_paths))

        if new_captions_since_save >= args.save_every:
            save_captions_atomic(args.output_file, captions)
            logging.info("Saved progress: %d captions total.", len(captions))
            new_captions_since_save = 0

    progress.close()
    save_captions_atomic(args.output_file, captions)
    logging.info("Done. Wrote %d captions to %s", len(captions), args.output_file)


if __name__ == "__main__":
    main()