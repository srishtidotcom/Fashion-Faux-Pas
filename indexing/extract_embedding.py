#!/usr/bin/env python3
"""Generate one FashionCLIP image embedding per image.

Input:
	data/raw/images/

Output:
	data/processed/embeddings.npy
	data/processed/filenames.json
	data/processed/embedding_metadata.json

This stage is image-only. It does not compute text embeddings, perform
retrieval, or index vectors into Qdrant.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# Allow truncated files to fail less aggressively; corrupted images are still skipped.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_NAME = "patrickjohncyh/fashion-clip"
DEFAULT_INPUT_DIR = Path("data/raw/images")
DEFAULT_EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
DEFAULT_FILENAMES_FILE = Path("data/processed/filenames.json")
DEFAULT_METADATA_FILE = Path("data/processed/embedding_metadata.json")
DEFAULT_BATCH_SIZE = 16
DEFAULT_VERIFY_PAIRS = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Generate FashionCLIP image embeddings.")
	parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory of input images.")
	parser.add_argument(
		"--embeddings-file",
		type=Path,
		default=DEFAULT_EMBEDDINGS_FILE,
		help="Path to embeddings.npy.",
	)
	parser.add_argument(
		"--filenames-file",
		type=Path,
		default=DEFAULT_FILENAMES_FILE,
		help="Path to filenames.json.",
	)
	parser.add_argument(
		"--metadata-file",
		type=Path,
		default=DEFAULT_METADATA_FILE,
		help="Path to embedding_metadata.json.",
	)
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Images per batch.")
	parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		choices=["auto", "cuda", "mps", "cpu"],
		help="Execution device.",
	)
	parser.add_argument(
		"--verify",
		action="store_true",
		help="Print a small cosine-similarity sanity check after saving embeddings.",
	)
	parser.add_argument(
		"--verify-pairs",
		type=int,
		default=DEFAULT_VERIFY_PAIRS,
		help="Number of random image pairs to verify when --verify is enabled.",
	)
	return parser.parse_args()


def setup_logging() -> None:
	"""Configure application logging."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def detect_device(requested: str = "auto") -> torch.device:
	"""Detect or force the compute device."""
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


def list_image_paths(input_dir: Path) -> List[Path]:
	"""Return sorted image paths from the input directory."""
	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

	return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(image_path: Path) -> Image.Image:
	"""Load an RGB image, raising on corruption or invalid files."""
	with Image.open(image_path) as img:
		img.load()
		return img.convert("RGB")


def chunked(items: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
	"""Yield fixed-size batches from a sequence."""
	for start in range(0, len(items), batch_size):
		yield list(items[start : start + batch_size])


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
	"""Normalize an embedding or embedding matrix to unit length."""
	array = np.asarray(embedding, dtype=np.float32)
	if array.ndim == 1:
		norm = float(np.linalg.norm(array))
		if norm == 0.0:
			return array
		return array / norm

	if array.ndim != 2:
		raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}")

	norms = np.linalg.norm(array, axis=1, keepdims=True)
	norms = np.where(norms == 0.0, 1.0, norms)
	return array / norms


def load_model(model_name: str, device: torch.device) -> Tuple[Any, Any]:
	"""Load the FashionCLIP processor and image encoder."""
	logging.info("Loading model: %s", model_name)
	processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
	model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=get_torch_dtype(device))
	model.to(device)
	model.eval()
	return processor, model


def load_existing_embeddings(
	embeddings_file: Path,
	filenames_file: Path,
) -> Tuple[Dict[str, np.ndarray], int]:
	"""Load existing embeddings if both output files are present and valid.

	The returned mapping is keyed by filename so new runs can reuse already
	computed vectors without recomputing them.
	"""
	if not embeddings_file.exists() or not filenames_file.exists():
		return {}, 0

	try:
		with filenames_file.open("r", encoding="utf-8") as f:
			filenames_raw = json.load(f)
		if not isinstance(filenames_raw, list):
			raise ValueError(f"Expected a JSON list in {filenames_file}")

		filenames = [str(item) for item in filenames_raw]
		embeddings = np.load(embeddings_file, mmap_mode="r")
		if embeddings.ndim == 1:
			embeddings = embeddings.reshape(1, -1)

		if embeddings.shape[0] != len(filenames):
			raise ValueError(
				f"Embedding count mismatch: {embeddings.shape[0]} vectors for {len(filenames)} filenames"
			)

		normalized = normalize_embedding(np.asarray(embeddings, dtype=np.float32))
		existing = {name: normalized[idx] for idx, name in enumerate(filenames)}
		return existing, int(normalized.shape[1])
	except Exception as exc:  # noqa: BLE001
		logging.warning("Could not load existing embeddings from %s / %s: %s", embeddings_file, filenames_file, exc)
		return {}, 0


def _extract_image_features(model: Any, pixel_values: torch.Tensor) -> torch.Tensor:
	"""Extract image features from a Hugging Face CLIP-style model."""
	if hasattr(model, "get_image_features"):
		return model.get_image_features(pixel_values=pixel_values)

	outputs = model(pixel_values=pixel_values)
	if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
		return outputs.image_embeds
	if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
		return outputs.pooler_output
	if isinstance(outputs, tuple) and outputs:
		return outputs[0]
	raise RuntimeError("Model did not return image embeddings")


def compute_embedding(
	processor: Any,
	model: Any,
	device: torch.device,
	image: Image.Image,
) -> np.ndarray:
	"""Compute a single normalized embedding vector for one image."""
	inputs = processor(images=image, return_tensors="pt")
	pixel_values = inputs["pixel_values"].to(device=device, dtype=get_torch_dtype(device))

	with torch.inference_mode():
		embedding = _extract_image_features(model, pixel_values)

	vector = embedding[0].detach().float().cpu().numpy()
	return normalize_embedding(vector)


def compute_batch_embeddings(
	processor: Any,
	model: Any,
	device: torch.device,
	image_paths: Sequence[Path],
) -> Tuple[Dict[str, np.ndarray], int]:
	"""Compute embeddings for a batch of images.

	Corrupted images are skipped. If the full batch fails, the function falls
	back to per-image processing so one bad sample does not poison the batch.
	"""
	valid_pairs: List[Tuple[Path, Image.Image]] = []
	skipped = 0

	for image_path in image_paths:
		try:
			valid_pairs.append((image_path, load_image(image_path)))
		except Exception as exc:  # noqa: BLE001
			logging.warning("Skipping unreadable image %s: %s", image_path.name, exc)
			skipped += 1

	if not valid_pairs:
		return {}, skipped

	try:
		images = [image for _, image in valid_pairs]
		inputs = processor(images=images, return_tensors="pt")
		pixel_values = inputs["pixel_values"].to(device=device, dtype=get_torch_dtype(device))

		with torch.inference_mode():
			embeddings = _extract_image_features(model, pixel_values)

		embeddings = normalize_embedding(embeddings.detach().float().cpu().numpy())
		return {
			image_path.name: embeddings[index]
			for index, (image_path, _) in enumerate(valid_pairs)
		}, skipped
	except Exception as exc:  # noqa: BLE001
		logging.warning("Batch embedding failed for %d images; falling back to per-image mode: %s", len(valid_pairs), exc)

	batch_embeddings: Dict[str, np.ndarray] = {}
	for image_path, image in valid_pairs:
		try:
			batch_embeddings[image_path.name] = compute_embedding(processor, model, device, image)
		except Exception as exc:  # noqa: BLE001
			logging.warning("Skipping image %s after embedding failure: %s", image_path.name, exc)
			skipped += 1

	return batch_embeddings, skipped


def save_json_atomic(path: Path, data: Any) -> None:
	"""Write JSON atomically."""
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)
	tmp_path.replace(path)


def save_embeddings_atomic(path: Path, embeddings: np.ndarray) -> None:
	"""Write the embedding matrix atomically."""
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
	np.save(tmp_path, np.asarray(embeddings, dtype=np.float32))
	tmp_path.replace(path)


def save_embeddings(
	embeddings_file: Path,
	filenames_file: Path,
	metadata_file: Path,
	filenames: Sequence[str],
	embeddings: np.ndarray,
	model_name: str,
	device: torch.device,
) -> None:
	"""Persist embeddings and filenames in a deterministic order."""
	save_embeddings_atomic(embeddings_file, embeddings)
	save_json_atomic(filenames_file, list(filenames))
	save_json_atomic(
		metadata_file,
		{
			"model": model_name,
			"embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
			"normalized": True,
			"dtype": "float32",
			"num_images": int(embeddings.shape[0]),
			"output_shape": list(embeddings.shape),
			"device_used": str(device),
		},
	)


def verify_embeddings(filenames: Sequence[str], embeddings: np.ndarray, num_pairs: int) -> None:
	"""Print a small random cosine-similarity sanity check."""
	if len(filenames) < 2 or embeddings.shape[0] < 2:
		logging.info("Verification skipped: fewer than two embeddings available.")
		return

	pair_count = min(num_pairs, len(filenames) * (len(filenames) - 1) // 2)
	if pair_count <= 0:
		return

	rng = random.Random()
	used_pairs = set()
	print("Verification samples:")
	attempts = 0
	while len(used_pairs) < pair_count and attempts < pair_count * 10:
		attempts += 1
		first_index = rng.randrange(len(filenames))
		second_index = rng.randrange(len(filenames) - 1)
		if second_index >= first_index:
			second_index += 1

		pair = tuple(sorted((first_index, second_index)))
		if pair in used_pairs:
			continue
		used_pairs.add(pair)

		similarity = float(np.dot(embeddings[pair[0]], embeddings[pair[1]]))
		print(f"{filenames[pair[0]]}\n{filenames[pair[1]]}\nSimilarity: {similarity:.4f}\n")


def main() -> None:
	"""Run the FashionCLIP embedding pipeline."""
	setup_logging()
	args = parse_args()

	start_time = time.perf_counter()
	device = detect_device(args.device)

	processor, model = load_model(args.model_name, device)
	all_image_paths = list_image_paths(args.input_dir)
	existing_embeddings, embedding_dim = load_existing_embeddings(args.embeddings_file, args.filenames_file)

	logging.info(
		"Found %d images, %d already embedded.",
		len(all_image_paths),
		len(existing_embeddings),
	)

	embeddings_by_name: Dict[str, np.ndarray] = dict(existing_embeddings)
	skipped_total = 0
	pending_paths = [image_path for image_path in all_image_paths if image_path.name not in embeddings_by_name]

	progress = tqdm(total=len(pending_paths), desc="Embedding images", unit="image")
	for batch_paths in chunked(pending_paths, args.batch_size):
		batch_embeddings, batch_skipped = compute_batch_embeddings(processor, model, device, batch_paths)
		embeddings_by_name.update(batch_embeddings)
		skipped_total += batch_skipped
		progress.update(len(batch_paths))

	progress.close()

	ordered_filenames = [image_path.name for image_path in all_image_paths if image_path.name in embeddings_by_name]
	if ordered_filenames:
		ordered_embeddings = np.stack([embeddings_by_name[name] for name in ordered_filenames], axis=0).astype(
			np.float32,
			copy=False,
		)
		ordered_embeddings = normalize_embedding(ordered_embeddings)
		embedding_dim = int(ordered_embeddings.shape[1])
	else:
		ordered_embeddings = np.empty((0, embedding_dim or 0), dtype=np.float32)

	save_embeddings(
		args.embeddings_file,
		args.filenames_file,
		args.metadata_file,
		ordered_filenames,
		ordered_embeddings,
		args.model_name,
		device,
	)

	elapsed = time.perf_counter() - start_time
	print(f"Total images processed: {len(ordered_filenames)}")
	print(f"Total skipped: {skipped_total}")
	print(f"Embedding dimension: {embedding_dim}")
	print(f"Output shape: {ordered_embeddings.shape}")
	print(f"Device used: {device}")
	print(f"Elapsed runtime: {elapsed:.2f}s")

	if args.verify and len(ordered_filenames) >= 2:
		verify_embeddings(ordered_filenames, ordered_embeddings, args.verify_pairs)


if __name__ == "__main__":
	main()
