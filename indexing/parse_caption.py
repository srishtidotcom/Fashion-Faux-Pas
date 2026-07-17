#!/usr/bin/env python3
"""Convert dense fashion captions into structured attribute JSON.

Input:
    data/processed/captions.json

Output:
    data/processed/attributes.json

This stage is resumable, validates every LLM response, and only keeps grounded
garment/object extractions to avoid hallucinated metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_INPUT_FILE = Path("data/processed/captions.json")
DEFAULT_OUTPUT_FILE = Path("data/processed/attributes.json")
DEFAULT_SAVE_EVERY = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_BEAMS = 1

POSE_TERMS = {
    "standing",
    "sitting",
    "walking",
    "leaning",
    "posing",
    "bending",
    "kneeling",
    "running",
    "lying",
    "crouching",
    "facing forward",
    "facing left",
    "facing right",
}

COLOR_TERMS = {
    "black",
    "white",
    "gray",
    "grey",
    "navy",
    "blue",
    "light blue",
    "dark blue",
    "red",
    "green",
    "olive",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "beige",
    "tan",
    "cream",
    "ivory",
    "khaki",
    "maroon",
    "burgundy",
    "teal",
    "turquoise",
    "silver",
    "gold",
    "golden",
    "denim",
}

PATTERN_TERMS = {
    "striped",
    "stripe",
    "plaid",
    "checked",
    "checkered",
    "polka dot",
    "polka dots",
    "floral",
    "patterned",
    "solid",
    "printed",
    "graphic",
    "paisley",
    "herringbone",
    "camouflage",
    "camo",
    "animal print",
    "leopard print",
    "zebra print",
    "pinstripe",
}

IMAGE_LIKE_OBJECT_HINTS = {
    "desk",
    "chair",
    "table",
    "sofa",
    "couch",
    "mirror",
    "window",
    "door",
    "lamp",
    "plant",
    "bag",
    "backpack",
    "handbag",
    "purse",
    "shelf",
    "rack",
    "sign",
    "street",
    "wall",
    "car",
    "bike",
    "bicycle",
    "phone",
    "computer",
    "laptop",
    "screen",
    "counter",
    "countertop",
    "bench",
    "hat",
    "scarf",
    "glasses",
    "sunglasses",
    "watch",
    "shoe",
    "shoes",
}

# -----------------------------
# Pydantic schema
# -----------------------------
class GarmentRecord(BaseModel):
    """A single garment entry."""

    type: Optional[str] = None
    color: Optional[str] = None
    pattern: Optional[str] = None

    class Config:
        extra = "forbid"


class AttributeRecord(BaseModel):
    """Structured metadata extracted from a caption."""

    scene: Optional[str] = None
    style: Optional[str] = None
    pose: Optional[str] = None
    objects: List[str] = Field(default_factory=list)
    garments: List[GarmentRecord] = Field(default_factory=list)

    class Config:
        extra = "forbid"


# -----------------------------
# CLI and logging
# -----------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse fashion captions into structured attributes.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="Path to captions.json.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to attributes.json.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face LLM name.")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save after N processed items.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retries for invalid outputs.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens to generate.")
    parser.add_argument("--num-beams", type=int, default=DEFAULT_NUM_BEAMS, help="Beam width for generation.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -----------------------------
# Device / model loading
# -----------------------------
def detect_device(requested: str = "auto") -> torch.device:
    """Detect or force a compute device."""
    if requested != "auto":
        return torch.device(requested)
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


def load_llm(model_name: str, device: torch.device) -> Tuple[Any, Any]:
    """Load tokenizer and causal LLM."""
    logging.info("Loading LLM: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(device),
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


# -----------------------------
# I/O
# -----------------------------
def load_json_dict(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {str(k): v for k, v in data.items()}


def save_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp_path.replace(path)


def normalize_caption_text(text: str) -> str:
    """Normalize caption text for prompt usage."""
    return " ".join(text.strip().split())


# -----------------------------
# Prompting and generation
# -----------------------------
SYSTEM_PROMPT = """You are a strict information extraction engine for fashion captions.

Return exactly one JSON object with this schema:
{
  "scene": string or null,
  "style": string or null,
  "pose": string or null,
  "objects": [string, ...],
  "garments": [
    {"type": string or null, "color": string or null, "pattern": string or null}
  ]
}

Rules:
- Return JSON only. No markdown, no code fences, no commentary.
- Never invent garments, colors, or patterns.
- Only include garments explicitly mentioned in the caption.
- garment.type must be a garment noun phrase from the caption.
- garment.color must be an explicit color word or phrase from the caption.
- garment.pattern must be an explicit pattern word or phrase from the caption.
- objects must contain only important non-garment scene objects explicitly mentioned in the caption.
- scene, style, and pose may be inferred, but should be concise lowercase labels.
- Use null when a value is unclear.
- Use [] when no objects or garments are present.

Example:
Caption: "A woman wearing a navy blazer over a white button-down shirt and black trousers standing inside a modern office next to a desk."
JSON:
{"scene":"office","style":"business formal","pose":"standing","objects":["desk"],"garments":[{"type":"blazer","color":"navy","pattern":null},{"type":"shirt","color":"white","pattern":null},{"type":"trousers","color":"black","pattern":null}]}
"""


def build_prompt(caption: str, previous_error: Optional[str] = None, previous_output: Optional[str] = None) -> str:
    """Build the extraction prompt."""
    user_block = f'Caption: "{caption}"\n\nReturn JSON only.'
    if previous_error:
        repair_block = [
            "The previous answer was invalid.",
            f"Error: {previous_error}",
        ]
        if previous_output:
            repair_block.append(f"Previous output: {previous_output}")
        repair_block.append("Return a corrected JSON object that matches the schema exactly.")
        user_block = "\n".join(repair_block) + "\n\n" + user_block
    return SYSTEM_PROMPT + "\n\n" + user_block


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    """Wrap a prompt in a chat template when available."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.split("\n\n", 1)[-1]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_text(
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    """Generate a deterministic model response."""
    formatted_prompt = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def extract_json_blob(text: str) -> str:
    """Extract the first balanced JSON object from model output."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : idx + 1]

    raise ValueError("Unbalanced JSON object in response.")


def parse_model_response(response_text: str) -> AttributeRecord:
    """Parse and validate a model response."""
    json_blob = extract_json_blob(response_text)
    payload = json.loads(json_blob)
    if hasattr(AttributeRecord, "model_validate"):
        return AttributeRecord.model_validate(payload)  # type: ignore[attr-defined]
    return AttributeRecord.parse_obj(payload)  # type: ignore[attr-defined]


# -----------------------------
# Grounding and normalization
# -----------------------------
def normalize_for_match(text: str) -> str:
    """Normalize text for robust substring checks."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def phrase_in_caption(phrase: str, caption: str) -> bool:
    """Check whether a phrase is grounded in the caption text."""
    candidate = normalize_for_match(phrase)
    haystack = normalize_for_match(caption)
    return bool(candidate) and candidate in haystack


def normalize_label(value: Optional[str], max_words: int) -> Optional[str]:
    """Normalize a concise label or return null if overly long."""
    if value is None:
        return None
    text = " ".join(str(value).strip().split()).lower()
    if not text:
        return None
    if len(text.split()) > max_words:
        return None
    return text


def is_object_grounded(obj: str, caption: str) -> bool:
    """Determine whether an object is explicitly present in the caption."""
    if phrase_in_caption(obj, caption):
        return True
    # Allow single-word object mentions that are common scene objects.
    normalized = normalize_for_match(obj)
    return any(token in normalized.split() for token in IMAGE_LIKE_OBJECT_HINTS) and phrase_in_caption(
        normalized, caption
    )


def ground_attributes(caption: str, record: AttributeRecord) -> AttributeRecord:
    """Remove ungrounded garments/objects and normalize text fields."""
    scene = normalize_label(record.scene, max_words=6)
    style = normalize_label(record.style, max_words=6)
    pose = normalize_label(record.pose, max_words=4)

    # Pose is only retained if it is concise; if the caption directly supports it,
    # keep it, otherwise null it out.
    if pose is not None and not any(term in normalize_for_match(caption) for term in normalize_for_match(pose).split()):
        pose = None

    filtered_objects: List[str] = []
    seen_objects = set()
    for obj in record.objects:
        normalized_obj = normalize_label(obj, max_words=6)
        if not normalized_obj:
            continue
        if not is_object_grounded(normalized_obj, caption):
            logging.debug("Dropping ungrounded object '%s' for caption '%s'", normalized_obj, caption)
            continue
        if normalized_obj not in seen_objects:
            seen_objects.add(normalized_obj)
            filtered_objects.append(normalized_obj)

    filtered_garments: List[GarmentRecord] = []
    seen_garments = set()
    for garment in record.garments:
        garment_type = normalize_label(garment.type, max_words=6)
        if not garment_type or not phrase_in_caption(garment_type, caption):
            logging.debug("Dropping ungrounded garment type '%s' for caption '%s'", garment.type, caption)
            continue

        color = normalize_label(garment.color, max_words=4)
        if color is not None and not phrase_in_caption(color, caption):
            color = None

        pattern = normalize_label(garment.pattern, max_words=4)
        if pattern is not None and not phrase_in_caption(pattern, caption):
            pattern = None

        grounded = GarmentRecord(type=garment_type, color=color, pattern=pattern)
        key = (grounded.type, grounded.color, grounded.pattern)
        if key not in seen_garments:
            seen_garments.add(key)
            filtered_garments.append(grounded)

    return AttributeRecord(
        scene=scene,
        style=style,
        pose=pose,
        objects=filtered_objects,
        garments=filtered_garments,
    )


# -----------------------------
# Extraction loop
# -----------------------------
def extract_attributes_for_caption(
    tokenizer: Any,
    model: Any,
    device: torch.device,
    caption: str,
    max_retries: int,
    max_new_tokens: int,
    num_beams: int,
) -> AttributeRecord:
    """Extract structured attributes from a single caption."""
    last_error: Optional[str] = None
    last_output: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        prompt = build_prompt(caption, previous_error=last_error, previous_output=last_output)
        raw_output = generate_text(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        last_output = raw_output

        try:
            parsed = parse_model_response(raw_output)
            grounded = ground_attributes(caption, parsed)
            return grounded
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            logging.warning("Invalid LLM output on attempt %d/%d: %s", attempt, max_retries, exc)

    logging.error("Falling back to empty attributes for caption: %s", caption)
    return AttributeRecord()


def load_captions_for_processing(input_file: Path, output_file: Path) -> List[Tuple[str, str]]:
    """Load captions and return items still needing processing."""
    captions = load_json_dict(input_file)
    existing = load_json_dict(output_file) if output_file.exists() else {}

    pending: List[Tuple[str, str]] = []
    for image_name, caption in sorted(captions.items()):
        if image_name in existing:
            continue
        pending.append((image_name, str(caption)))

    logging.info(
        "Loaded %d captions, %d already processed, %d pending.",
        len(captions),
        len(existing),
        len(pending),
    )
    return pending


def dump_attributes(existing: Dict[str, Any], new_items: Dict[str, Any]) -> Dict[str, Any]:
    """Merge existing and new attributes with stable ordering."""
    merged = dict(existing)
    merged.update(new_items)
    return dict(sorted(merged.items(), key=lambda kv: kv[0]))


def model_to_dict(record: AttributeRecord) -> Dict[str, Any]:
    """Convert a pydantic model into a serializable dict."""
    if hasattr(record, "model_dump"):
        return record.model_dump()  # type: ignore[attr-defined]
    return record.dict()  # type: ignore[no-any-return]


def main() -> None:
    """Run structured attribute extraction."""
    setup_logging()
    args = parse_args()

    device = detect_device(args.device)
    logging.info("Using device: %s", device)

    tokenizer, model = load_llm(args.model_name, device)

    pending = load_captions_for_processing(args.input_file, args.output_file)
    if not pending:
        logging.info("No pending captions found.")
        return

    existing_output = load_json_dict(args.output_file) if args.output_file.exists() else {}
    results: Dict[str, Any] = dict(existing_output)
    processed_since_save = 0

    progress = tqdm(total=len(pending), desc="Parsing captions", unit="caption")
    for image_name, caption in pending:
        try:
            record = extract_attributes_for_caption(
                tokenizer=tokenizer,
                model=model,
                device=device,
                caption=normalize_caption_text(caption),
                max_retries=args.max_retries,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            results[image_name] = model_to_dict(record)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to parse caption for %s: %s", image_name, exc)
            results[image_name] = model_to_dict(AttributeRecord())

        processed_since_save += 1
        progress.update(1)

        if processed_since_save >= args.save_every:
            save_json_atomic(args.output_file, dump_attributes(existing_output, results))
            logging.info("Saved progress: %d records total.", len(results))
            processed_since_save = 0

    progress.close()
    save_json_atomic(args.output_file, dump_attributes(existing_output, results))
    logging.info("Done. Wrote %d records to %s", len(results), args.output_file)


if __name__ == "__main__":
    main()