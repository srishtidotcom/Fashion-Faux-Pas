#!/usr/bin/env python3
"""Build canonical image documents from captions and structured attributes.

Input:
	data/processed/captions.json
	data/processed/attributes.json

Output:
	data/processed/documents.json

This stage merges human-readable metadata into one flattened document per
image. It does not include embeddings or perform any retrieval/indexing work.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_CAPTIONS_FILE = Path("data/processed/captions.json")
DEFAULT_ATTRIBUTES_FILE = Path("data/processed/attributes.json")
DEFAULT_OUTPUT_FILE = Path("data/processed/documents.json")


@dataclass(frozen=True)
class ImageDocument:
	"""Canonical metadata for a single image."""

	filename: str
	caption: str
	scene: Optional[str] = None
	style: Optional[str] = None
	pose: Optional[str] = None
	objects: List[str] = field(default_factory=list)
	garments: List[Dict[str, Optional[str]]] = field(default_factory=list)

	def to_dict(self) -> Dict[str, Any]:
		"""Convert the document into a JSON-serializable dictionary."""
		return {
			"filename": self.filename,
			"caption": self.caption,
			"scene": self.scene,
			"style": self.style,
			"pose": self.pose,
			"objects": list(self.objects),
			"garments": [dict(garment) for garment in self.garments],
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "ImageDocument":
		"""Build an ImageDocument from a dictionary representation."""
		if not isinstance(payload, dict):
			raise TypeError("ImageDocument payload must be a dictionary")

		filename = str(payload.get("filename", "")).strip()
		caption = str(payload.get("caption", "")).strip()
		if not filename:
			raise ValueError("Missing filename")
		if not caption:
			raise ValueError("Missing caption")

		scene = _normalize_optional_text(payload.get("scene"))
		style = _normalize_optional_text(payload.get("style"))
		pose = _normalize_optional_text(payload.get("pose"))
		objects = _normalize_string_list(payload.get("objects", []))
		garments = _normalize_garment_list(payload.get("garments", []))

		return cls(
			filename=filename,
			caption=caption,
			scene=scene,
			style=style,
			pose=pose,
			objects=objects,
			garments=garments,
		)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Build canonical image documents for Qdrant payloads.")
	parser.add_argument("--captions-file", type=Path, default=DEFAULT_CAPTIONS_FILE, help="Path to captions.json.")
	parser.add_argument(
		"--attributes-file",
		type=Path,
		default=DEFAULT_ATTRIBUTES_FILE,
		help="Path to attributes.json.",
	)
	parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to documents.json.")
	return parser.parse_args()


def setup_logging() -> None:
	"""Configure application logging."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _duplicate_object_hook(duplicates: List[str]):
	"""Create a JSON object hook that records duplicate keys while parsing."""

	def hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
		seen: Dict[str, Any] = {}
		for key, value in pairs:
			if key in seen:
				duplicates.append(key)
			seen[key] = value
		return seen

	return hook


def load_json_map(path: Path) -> Tuple[Dict[str, Any], List[str]]:
	"""Load a JSON object from disk and record duplicate keys if present."""
	if not path.exists():
		raise FileNotFoundError(f"Missing input file: {path}")

	raw_text = path.read_text(encoding="utf-8")
	if not raw_text.strip():
		raise ValueError(f"Empty JSON file: {path}")

	duplicates: List[str] = []
	try:
		data = json.loads(raw_text, object_pairs_hook=_duplicate_object_hook(duplicates))
	except json.JSONDecodeError as exc:
		raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

	if not isinstance(data, dict):
		raise ValueError(f"Expected a JSON object in {path}")

	return {str(key): value for key, value in data.items()}, duplicates


def load_captions(captions_file: Path) -> Tuple[Dict[str, str], List[str]]:
	"""Load captions keyed by filename."""
	data, duplicates = load_json_map(captions_file)
	captions: Dict[str, str] = {}
	for filename, caption in data.items():
		if isinstance(caption, str) and caption.strip():
			captions[filename] = " ".join(caption.split())
		else:
			logging.warning("Skipping invalid caption for %s", filename)
	return captions, duplicates


def load_attributes(attributes_file: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
	"""Load structured attributes keyed by filename."""
	data, duplicates = load_json_map(attributes_file)
	attributes: Dict[str, Dict[str, Any]] = {}
	for filename, payload in data.items():
		if isinstance(payload, dict):
			attributes[filename] = payload
		else:
			logging.warning("Skipping invalid attributes for %s", filename)
	return attributes, duplicates


def _normalize_optional_text(value: Any) -> Optional[str]:
	"""Normalize an optional text field to a compact lowercase string."""
	if value is None:
		return None
	text = " ".join(str(value).strip().split()).lower()
	return text or None


def _normalize_string_list(value: Any) -> List[str]:
	"""Normalize a list of strings while removing empty items and duplicates."""
	if not isinstance(value, list):
		return []

	normalized: List[str] = []
	seen = set()
	for item in value:
		if not isinstance(item, str):
			continue
		text = " ".join(item.strip().split()).lower()
		if not text or text in seen:
			continue
		seen.add(text)
		normalized.append(text)
	return normalized


def _normalize_garment_list(value: Any) -> List[Dict[str, Optional[str]]]:
	"""Normalize garment dictionaries into a stable, flattened representation."""
	if not isinstance(value, list):
		return []

	garments: List[Dict[str, Optional[str]]] = []
	for item in value:
		if not isinstance(item, dict):
			continue
		garment_type = _normalize_optional_text(item.get("type"))
		color = _normalize_optional_text(item.get("color"))
		pattern = _normalize_optional_text(item.get("pattern"))
		if garment_type is None and color is None and pattern is None:
			continue
		garments.append({"type": garment_type, "color": color, "pattern": pattern})
	return garments


def validate_inputs(
	captions: Dict[str, str],
	attributes: Dict[str, Dict[str, Any]],
	caption_duplicates: Sequence[str],
	attribute_duplicates: Sequence[str],
) -> List[str]:
	"""Validate input consistency and return the filenames that can be merged."""
	if caption_duplicates:
		logging.warning("Duplicate caption filenames detected: %s", ", ".join(sorted(set(caption_duplicates))))
	if attribute_duplicates:
		logging.warning("Duplicate attribute filenames detected: %s", ", ".join(sorted(set(attribute_duplicates))))

	caption_keys = set(captions)
	attribute_keys = set(attributes)

	missing_attributes = sorted(caption_keys - attribute_keys)
	missing_captions = sorted(attribute_keys - caption_keys)

	if missing_attributes:
		logging.warning("Missing attributes for %d filenames", len(missing_attributes))
	if missing_captions:
		logging.warning("Missing captions for %d filenames", len(missing_captions))

	return sorted(caption_keys & attribute_keys)


def merge_documents(
	filenames: Sequence[str],
	captions: Dict[str, str],
	attributes: Dict[str, Dict[str, Any]],
) -> Tuple[List[ImageDocument], int]:
	"""Merge aligned captions and attributes into typed ImageDocument objects."""
	documents: List[ImageDocument] = []
	skipped = 0

	for filename in tqdm(filenames, desc="Building documents", unit="image"):
		caption = captions.get(filename)
		attribute_payload = attributes.get(filename)
		if caption is None or attribute_payload is None:
			skipped += 1
			continue

		try:
			record = ImageDocument.from_dict(
				{
					"filename": filename,
					"caption": caption,
					"scene": attribute_payload.get("scene"),
					"style": attribute_payload.get("style"),
					"pose": attribute_payload.get("pose"),
					"objects": attribute_payload.get("objects", []),
					"garments": attribute_payload.get("garments", []),
				}
			)
		except Exception as exc:  # noqa: BLE001
			logging.warning("Skipping invalid document for %s: %s", filename, exc)
			skipped += 1
			continue

		documents.append(record)

	return documents, skipped


def save_documents(path: Path, documents: Sequence[ImageDocument]) -> None:
	"""Write the merged document list to disk atomically."""
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	payload = [document.to_dict() for document in documents]
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
	tmp_path.replace(path)


def main() -> None:
	"""Build canonical ImageDocument objects from captions and attributes."""
	setup_logging()
	args = parse_args()
	start_time = time.perf_counter()

	try:
		captions, caption_duplicates = load_captions(args.captions_file)
		attributes, attribute_duplicates = load_attributes(args.attributes_file)
	except Exception as exc:  # noqa: BLE001
		logging.error("Failed to load inputs: %s", exc)
		return

	logging.info("Images discovered: %d", len(set(captions) | set(attributes)))

	aligned_filenames = validate_inputs(
		captions=captions,
		attributes=attributes,
		caption_duplicates=caption_duplicates,
		attribute_duplicates=attribute_duplicates,
	)

	documents, skipped = merge_documents(aligned_filenames, captions, attributes)

	try:
		save_documents(args.output_file, documents)
	except Exception as exc:  # noqa: BLE001
		logging.error("Failed to write documents: %s", exc)
		return

	elapsed = time.perf_counter() - start_time
	missing_captions = len(set(attributes) - set(captions))
	missing_attributes = len(set(captions) - set(attributes))

	print(f"Documents created: {len(documents)}")
	print(f"Documents skipped: {skipped}")
	print(f"Missing captions: {missing_captions}")
	print(f"Missing attributes: {missing_attributes}")
	print(f"Output path: {args.output_file}")
	print(f"Elapsed runtime: {elapsed:.2f}s")


if __name__ == "__main__":
	main()
