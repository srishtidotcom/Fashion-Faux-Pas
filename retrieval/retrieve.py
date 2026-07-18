#!/usr/bin/env python3
"""Semantic ANN retrieval for fashion queries.

This module embeds QueryDocument.embedding_text with FashionCLIP, queries the
local Qdrant collection, and returns typed semantic candidates. It does not
perform reranking, query parsing, or evaluation.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoModel, AutoProcessor

try:
	from retrieval.parse_query import QueryDocument, parse_query
except (ModuleNotFoundError, ImportError):  # pragma: no cover
	# When executed as `python retrieval/retrieve.py`, the parent package is not
	# installed, so fall back to a sibling-module import from the same directory.
	from parse_query import QueryDocument, parse_query

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_NAME = "patrickjohncyh/fashion-clip"
DEFAULT_QDRANT_PATH = Path("data/qdrant_db")
DEFAULT_COLLECTION_NAME = "fashion_images"
DEFAULT_TOP_K = 100
DEFAULT_DISPLAY_TOP_K = 10


@dataclass(slots=True)
class RetrievedCandidate:
	"""Typed semantic retrieval result from Qdrant."""

	id: int
	filename: str
	score: float
	caption: str
	scene: Optional[str]
	style: Optional[str]
	pose: Optional[str]
	objects: List[str]
	garments: List[Dict[str, Any]]
	vector_score: float
	payload: Dict[str, Any]
	attribute_score: float = 0.0
	final_score: float = 0.0
	score_breakdown: Optional[Dict[str, float]] = None


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Run semantic retrieval over local Qdrant.")
	parser.add_argument("--query", type=str, default=None, help="Optional free-form query string.")
	parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of semantic candidates to return.")
	parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face FashionCLIP model.")
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


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
	"""Normalize a vector or matrix to unit length."""
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


@lru_cache(maxsize=1)
def _load_fashionclip_cached(model_name: str, device_name: str) -> Tuple[Any, Any, torch.device]:
	"""Load FashionCLIP once per process."""
	device = detect_device(device_name)
	logging.info("Loading FashionCLIP...")
	processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
	model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=get_torch_dtype(device))
	model.to(device)
	model.eval()
	return processor, model, device


def load_fashionclip(model_name: str = DEFAULT_MODEL_NAME, device: str = "auto") -> Tuple[Any, Any, torch.device]:
	"""Return a cached FashionCLIP processor, model, and resolved device."""
	return _load_fashionclip_cached(model_name, device)


@lru_cache(maxsize=1)
def _load_qdrant_cached(qdrant_path: str) -> QdrantClient:
	"""Create a cached local Qdrant client."""
	logging.info("Connecting to Qdrant...")
	return QdrantClient(path=qdrant_path)


def load_qdrant(qdrant_path: Path = DEFAULT_QDRANT_PATH) -> QdrantClient:
	"""Return a cached local Qdrant client."""
	return _load_qdrant_cached(str(qdrant_path))


def close_qdrant(qdrant_path: Path = DEFAULT_QDRANT_PATH) -> None:
	"""Close the cached local Qdrant client and clear the cache."""
	if _load_qdrant_cached.cache_info().currsize > 0:
		client = _load_qdrant_cached(str(qdrant_path))
		client.close()
	_load_qdrant_cached.cache_clear()


def embed_query(processor: Any, model: Any, device: torch.device, embedding_text: str) -> np.ndarray:
	"""Embed the query text with FashionCLIP and L2-normalize the result."""
	logging.info("Embedding query...")
	inputs = processor(text=[embedding_text], return_tensors="pt", padding=True, truncation=True)
	inputs = {key: value.to(device) for key, value in inputs.items()}

	with torch.inference_mode():
		if hasattr(model, "get_text_features"):
			embedding = model.get_text_features(**inputs)
		else:
			outputs = model(**inputs)
			if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
				embedding = outputs.text_embeds
			elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
				embedding = outputs.pooler_output
			elif isinstance(outputs, tuple) and outputs:
				embedding = outputs[0]
			else:
				raise RuntimeError("Model did not return text embeddings")

	vector = embedding[0].detach().float().cpu().numpy()
	return normalize_embedding(vector)


def search_qdrant(
	client: QdrantClient,
	collection_name: str,
	query_vector: np.ndarray,
	top_k: int,
) -> List[models.ScoredPoint]:
	"""Run ANN search against Qdrant and return scored points."""
	logging.info("Searching Qdrant...")
	response = client.query_points(
		collection_name=collection_name,
		query=query_vector.tolist(),
		limit=top_k,
		with_payload=True,
		with_vectors=False,
	)
	return list(response.points)


def convert_hit(hit: models.ScoredPoint) -> RetrievedCandidate:
	"""Convert a Qdrant scored point into a typed retrieval candidate."""
	payload = dict(hit.payload or {})
	vector_score = float(hit.score)
	return RetrievedCandidate(
		id=int(hit.id),
		filename=str(payload.get("filename", "")),
		score=vector_score,
		caption=str(payload.get("caption", "")),
		scene=payload.get("scene"),
		style=payload.get("style"),
		pose=payload.get("pose"),
		objects=list(payload.get("objects", [])) if isinstance(payload.get("objects"), list) else [],
		garments=list(payload.get("garments", [])) if isinstance(payload.get("garments"), list) else [],
		vector_score=vector_score,
		attribute_score=0.0,
		final_score=vector_score,
		score_breakdown=None,
		payload=payload,
	)


def retrieve(
	query_document: QueryDocument,
	top_k: int = DEFAULT_TOP_K,
	model_name: str = DEFAULT_MODEL_NAME,
	qdrant_path: Path = DEFAULT_QDRANT_PATH,
	collection_name: str = DEFAULT_COLLECTION_NAME,
	device: str = "auto",
) -> List[RetrievedCandidate]:
	"""Retrieve the top semantic candidates for a validated query document."""
	try:
		if not isinstance(query_document, QueryDocument):
			logging.warning("Invalid QueryDocument supplied to retrieve().")
			return []
		if not query_document.embedding_text.strip():
			return []

		processor, model, resolved_device = load_fashionclip(model_name=model_name, device=device)
		client = load_qdrant(qdrant_path)
		query_vector = embed_query(processor, model, resolved_device, query_document.embedding_text)
		hits = search_qdrant(client, collection_name, query_vector, top_k)
		candidates = [convert_hit(hit) for hit in hits]
		logging.info("Retrieved %d candidates.", len(candidates))
		return candidates
	except Exception as exc:  # noqa: BLE001
		logging.warning("Retrieval failed: %s", exc)
		return []


def display_candidates(candidates: Sequence[RetrievedCandidate], top_n: int = DEFAULT_DISPLAY_TOP_K) -> None:
	"""Pretty-print the top semantic candidates for debugging and CLI use."""
	visible_candidates = list(candidates[:top_n])
	if not visible_candidates:
		print("No results found.")
		return

	print("======================================================")
	for rank, candidate in enumerate(visible_candidates, start=1):
		print(f"Rank {rank}")
		print()
		print("Filename")
		print(candidate.filename)
		print()
		print("Score")
		print(f"{candidate.score:.3f}")
		print()
		print("Scene")
		print(candidate.scene or "null")
		print()
		print("Style")
		print(candidate.style or "null")
		print()
		print("Caption")
		print(f'"{candidate.caption}"')
		if rank != len(visible_candidates):
			print("------------------------------------------------------")
	print("======================================================")


def _prompt_query() -> str:
	"""Read a raw query from stdin."""
	try:
		return input("Enter query: ")
	except EOFError:
		return ""


def main() -> None:
	"""Prompt for a query, run semantic retrieval, and print the top results."""
	setup_logging()
	args = parse_args()
	client: Optional[QdrantClient] = None

	try:
		raw_query = args.query if args.query is not None else _prompt_query()
		query_document = parse_query(
			raw_query,
			model_name=args.model_name,
			device=args.device,
		)

		# Rerank hook goes here later.
		candidates = retrieve(
			query_document=query_document,
			top_k=args.top_k,
			model_name=args.model_name,
			qdrant_path=args.qdrant_path,
			collection_name=args.collection_name,
			device=args.device,
		)

		display_candidates(candidates, top_n=DEFAULT_DISPLAY_TOP_K)
		logging.info("Done.")
	finally:
		close_qdrant(args.qdrant_path)


if __name__ == "__main__":
	main()
