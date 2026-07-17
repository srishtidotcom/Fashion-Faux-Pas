#!/usr/bin/env python3
"""Index canonical fashion documents and image embeddings into local Qdrant.

This script is the final offline indexing step. It validates that the metadata
documents, embedding matrix, and filename list are perfectly aligned before
creating or updating the persistent local Qdrant collection.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_DOCUMENTS_FILE = Path("data/processed/documents.json")
DEFAULT_EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
DEFAULT_FILENAMES_FILE = Path("data/processed/filenames.json")
DEFAULT_QDRANT_PATH = Path("data/qdrant_db")
DEFAULT_COLLECTION_NAME = "fashion_images"
DEFAULT_VECTOR_SIZE = 512
DEFAULT_DISTANCE = models.Distance.COSINE
BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Index fashion documents into local Qdrant.")
	parser.add_argument("--documents-file", type=Path, default=DEFAULT_DOCUMENTS_FILE, help="Path to documents.json.")
	parser.add_argument("--embeddings-file", type=Path, default=DEFAULT_EMBEDDINGS_FILE, help="Path to embeddings.npy.")
	parser.add_argument("--filenames-file", type=Path, default=DEFAULT_FILENAMES_FILE, help="Path to filenames.json.")
	parser.add_argument(
		"--qdrant-path",
		type=Path,
		default=DEFAULT_QDRANT_PATH,
		help="Local Qdrant storage directory.",
	)
	parser.add_argument(
		"--collection-name",
		type=str,
		default=DEFAULT_COLLECTION_NAME,
		help="Qdrant collection name.",
	)
	parser.add_argument(
		"--vector-size",
		type=int,
		default=DEFAULT_VECTOR_SIZE,
		help="Expected embedding dimension.",
	)
	return parser.parse_args()


def setup_logging() -> None:
	"""Configure application logging."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_documents(path: Path) -> List[Dict[str, Any]]:
	"""Load the canonical document list from disk."""
	logging.info("Loading documents...")
	if not path.exists():
		raise FileNotFoundError(f"Missing documents file: {path}")

	raw_text = path.read_text(encoding="utf-8")
	if not raw_text.strip():
		raise ValueError(f"Empty documents file: {path}")

	try:
		data = json.loads(raw_text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

	if not isinstance(data, list):
		raise ValueError(f"Expected a JSON list in {path}")
	return data


def load_filenames(path: Path) -> List[str]:
	"""Load the filename ordering used by the embedding matrix."""
	logging.info("Loading filenames...")
	if not path.exists():
		raise FileNotFoundError(f"Missing filenames file: {path}")

	raw_text = path.read_text(encoding="utf-8")
	if not raw_text.strip():
		raise ValueError(f"Empty filenames file: {path}")

	try:
		data = json.loads(raw_text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

	if not isinstance(data, list):
		raise ValueError(f"Expected a JSON list in {path}")

	filenames = [str(item) for item in data]
	if any(not filename for filename in filenames):
		raise ValueError(f"Filenames file contains empty entries: {path}")
	return filenames


def load_embeddings(path: Path) -> np.ndarray:
	"""Load the normalized embedding matrix from disk."""
	logging.info("Loading embeddings...")
	if not path.exists():
		raise FileNotFoundError(f"Missing embeddings file: {path}")

	try:
		embeddings = np.load(path, mmap_mode="r")
	except Exception as exc:  # noqa: BLE001
		raise ValueError(f"Failed to load embeddings from {path}: {exc}") from exc

	if embeddings.ndim != 2:
		raise ValueError(f"Expected a 2D embedding matrix in {path}, got shape {embeddings.shape}")
	if embeddings.dtype != np.float32:
		embeddings = np.asarray(embeddings, dtype=np.float32)

	return embeddings


def validate_alignment(
	documents: Sequence[Dict[str, Any]],
	embeddings: np.ndarray,
	filenames: Sequence[str],
) -> List[Dict[str, Any]]:
	"""Validate that documents, embeddings, and filenames are perfectly aligned."""
	if len(documents) != len(embeddings) or len(documents) != len(filenames):
		raise ValueError(
			"Input length mismatch: "
			f"documents={len(documents)}, embeddings={len(embeddings)}, filenames={len(filenames)}"
		)

	filename_set = set()
	aligned_documents: List[Dict[str, Any]] = []
	for index, (document, filename) in enumerate(zip(documents, filenames, strict=True)):
		if not isinstance(document, dict):
			raise ValueError(f"Document at index {index} is not an object")
		if filename in filename_set:
			raise ValueError(f"Duplicate filename detected in filenames.json: {filename}")
		filename_set.add(filename)

		document_filename = str(document.get("filename", ""))
		if not document_filename:
			raise ValueError(f"Missing filename in document at index {index}")
		if document_filename != filename:
			raise ValueError(
				f"Filename mismatch at index {index}: document={document_filename!r}, filenames={filename!r}"
			)
		aligned_documents.append(document)

	return aligned_documents


def get_collection_config(vector_size: int) -> models.VectorParams:
	"""Build the collection vector configuration."""
	return models.VectorParams(size=vector_size, distance=DEFAULT_DISTANCE)


def create_or_recreate_collection(
	client: QdrantClient,
	collection_name: str,
	vector_size: int,
) -> None:
	"""Create the collection or recreate it if the vector schema differs."""
	logging.info("Creating collection...")
	vector_config = get_collection_config(vector_size)

	if not client.collection_exists(collection_name):
		client.create_collection(collection_name=collection_name, vectors_config=vector_config)
		logging.info("Created collection: %s", collection_name)
		return

	info = client.get_collection(collection_name)
	current_vectors = info.config.params.vectors
	if isinstance(current_vectors, dict):
		raise ValueError("Named vector collections are not supported by this indexing script")

	if current_vectors.size != vector_size or current_vectors.distance != DEFAULT_DISTANCE:
		logging.warning(
			"Collection schema mismatch detected; recreating %s (size=%s, distance=%s)",
			collection_name,
			current_vectors.size,
			current_vectors.distance,
		)
		client.recreate_collection(collection_name=collection_name, vectors_config=vector_config)
		logging.info("Recreated collection: %s", collection_name)
	else:
		logging.info("Reusing existing collection: %s", collection_name)


def build_points(
	documents: Sequence[Dict[str, Any]],
	embeddings: np.ndarray,
	start_index: int,
	end_index: int,
) -> List[models.PointStruct]:
	"""Build Qdrant point structs for a batch."""
	points: List[models.PointStruct] = []
	for local_index in range(start_index, end_index):
		document = documents[local_index]
		payload = {
			"filename": document.get("filename"),
			"caption": document.get("caption"),
			"scene": document.get("scene"),
			"style": document.get("style"),
			"pose": document.get("pose"),
			"objects": document.get("objects", []),
			"garments": document.get("garments", []),
		}
		points.append(
			models.PointStruct(
				id=local_index,
				vector=embeddings[local_index].tolist(),
				payload=payload,
			)
		)
	return points


def upload_batches(
	client: QdrantClient,
	collection_name: str,
	documents: Sequence[Dict[str, Any]],
	embeddings: np.ndarray,
) -> None:
	"""Upload points to Qdrant in deterministic batches."""
	logging.info("Uploading vectors...")
	total = len(documents)
	progress = tqdm(total=total, desc="Uploading vectors", unit="vector")

	for start_index in range(0, total, BATCH_SIZE):
		end_index = min(start_index + BATCH_SIZE, total)
		try:
			points = build_points(documents, embeddings, start_index, end_index)
			client.upsert(collection_name=collection_name, points=points, wait=True)
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError(f"Failed to upload batch {start_index}:{end_index} to Qdrant: {exc}") from exc
		progress.update(end_index - start_index)

	progress.close()


def verify_index(client: QdrantClient, collection_name: str, expected_count: int) -> Tuple[int, int, models.Distance]:
	"""Verify collection metadata and point count after upload."""
	logging.info("Verifying collection...")
	info = client.get_collection(collection_name)
	vector_params = info.config.params.vectors
	if isinstance(vector_params, dict):
		raise ValueError("Named vector collections are not supported by this indexing script")

	stored_count = int(info.points_count)
	if stored_count != expected_count:
		raise RuntimeError(
			f"Stored vector count mismatch for {collection_name}: expected {expected_count}, got {stored_count}"
		)

	return int(vector_params.size), vector_params.distance, stored_count


def main() -> None:
	"""Run the Qdrant indexing pipeline."""
	setup_logging()
	args = parse_args()
	start_time = time.perf_counter()

	documents = load_documents(args.documents_file)
	embeddings = load_embeddings(args.embeddings_file)
	filenames = load_filenames(args.filenames_file)

	aligned_documents = validate_alignment(documents, embeddings, filenames)
	logging.info("Validated %d records.", len(aligned_documents))

	client = QdrantClient(path=str(args.qdrant_path))
	create_or_recreate_collection(client, args.collection_name, args.vector_size)

	upload_batches(client, args.collection_name, aligned_documents, embeddings)

	vector_size, distance, stored_count = verify_index(client, args.collection_name, len(aligned_documents))

	elapsed = time.perf_counter() - start_time
	print(f"Collection: {args.collection_name}")
	print(f"Vector Dimension: {vector_size}")
	print(f"Distance Metric: {distance.name}")
	print(f"Stored Vectors: {stored_count}")
	print(f"Database Location:\n{args.qdrant_path}")
	print(f"Elapsed Time: {elapsed:.2f} s")
	print("Indexing completed successfully.")


if __name__ == "__main__":
	main()
