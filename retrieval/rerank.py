#!/usr/bin/env python3
"""Deterministic reranking for multimodal fashion retrieval.

The reranker consumes a validated QueryDocument and semantically retrieved
RetrievedCandidate objects, then combines vector similarity with structured
attribute matching. It does not parse queries, compute embeddings, or query
Qdrant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
	from retrieval.parse_query import GarmentQuery, QueryDocument
	from retrieval.retrieve import RetrievedCandidate
except (ModuleNotFoundError, ImportError):  # pragma: no cover
	from parse_query import GarmentQuery, QueryDocument
	from retrieve import RetrievedCandidate

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_ALPHA = 0.75
DEFAULT_BETA = 0.25


@dataclass(slots=True)
class ScoreBreakdown:
	"""Component-wise attribute scores used to build the final hybrid score."""

	garment: Optional[float] = None
	scene: Optional[float] = None
	style: Optional[float] = None
	pose: Optional[float] = None
	objects: Optional[float] = None

	def to_dict(self) -> Dict[str, Optional[float]]:
		"""Convert the score breakdown into a serializable dictionary."""
		return {
			"garment": self.garment,
			"scene": self.scene,
			"style": self.style,
			"pose": self.pose,
			"objects": self.objects,
		}


def _normalize_text(value: Any) -> str:
	"""Normalize arbitrary structured values for comparison."""
	if value is None:
		return ""
	text = str(value).strip().lower()
	return " ".join(text.split())


def _normalize_list(values: Any) -> List[str]:
	"""Normalize a potential list of strings into lowercase, de-duplicated text."""
	if not isinstance(values, list):
		return []

	normalized: List[str] = []
	seen = set()
	for value in values:
		text = _normalize_text(value)
		if not text or text in seen:
			continue
		seen.add(text)
		normalized.append(text)
	return normalized


def _normalize_garment_query(value: Any) -> Optional[GarmentQuery]:
	"""Safely convert a loose value into a GarmentQuery."""
	if isinstance(value, GarmentQuery):
		return value
	if isinstance(value, dict):
		try:
			return GarmentQuery.model_validate(value)
		except Exception:  # noqa: BLE001
			return None
	return None


def _normalize_garment_payload(value: Any) -> Optional[GarmentQuery]:
	"""Safely convert a candidate garment payload into a GarmentQuery."""
	if isinstance(value, GarmentQuery):
		return value
	if isinstance(value, dict):
		try:
			return GarmentQuery.model_validate(value)
		except Exception:  # noqa: BLE001
			return None
	return None


def _garment_matches(query_garment: GarmentQuery, image_garment: GarmentQuery) -> bool:
	"""Check whether one image garment can satisfy one query garment."""
	query_type = _normalize_text(query_garment.type)
	image_type = _normalize_text(image_garment.type)
	if not query_type or not image_type or query_type != image_type:
		return False

	query_color = _normalize_text(query_garment.color)
	image_color = _normalize_text(image_garment.color)
	if query_color and image_color and query_color != image_color:
		return False

	query_pattern = _normalize_text(query_garment.pattern)
	image_pattern = _normalize_text(image_garment.pattern)
	if query_pattern and image_pattern and query_pattern != image_pattern:
		return False

	return True


def garment_score(query_document: QueryDocument, candidate: RetrievedCandidate) -> Optional[float]:
	"""Compute greedy one-to-one garment recall over the available garments."""
	query_garments = [garment for garment in query_document.garments if _normalize_text(garment.type)]
	if not query_garments:
		return None

	image_garments = []
	for value in candidate.garments:
		garment = _normalize_garment_payload(value)
		if garment is not None and _normalize_text(garment.type):
			image_garments.append(garment)

	if not image_garments:
		return 0.0

	matched = 0
	used_indices = set()
	for query_garment in query_garments:
		for index, image_garment in enumerate(image_garments):
			if index in used_indices:
				continue
			if _garment_matches(query_garment, image_garment):
				used_indices.add(index)
				matched += 1
				break

	return matched / len(query_garments)


def scene_score(query_document: QueryDocument, candidate: RetrievedCandidate) -> Optional[float]:
	"""Return 1.0 for an exact normalized scene match, else 0.0."""
	query_scene = _normalize_text(query_document.scene)
	if not query_scene:
		return None
	return 1.0 if query_scene == _normalize_text(candidate.scene) else 0.0


def style_score(query_document: QueryDocument, candidate: RetrievedCandidate) -> Optional[float]:
	"""Return 1.0 for an exact normalized style match, else 0.0."""
	query_style = _normalize_text(query_document.style)
	if not query_style:
		return None
	return 1.0 if query_style == _normalize_text(candidate.style) else 0.0


def pose_score(query_document: QueryDocument, candidate: RetrievedCandidate) -> Optional[float]:
	"""Return 1.0 for an exact normalized pose match, else 0.0."""
	query_pose = _normalize_text(query_document.pose)
	if not query_pose:
		return None
	return 1.0 if query_pose == _normalize_text(candidate.pose) else 0.0


def object_score(query_document: QueryDocument, candidate: RetrievedCandidate) -> Optional[float]:
	"""Compute Jaccard similarity between query and candidate objects."""
	query_objects = set(_normalize_list(query_document.objects))
	if not query_objects:
		return None

	candidate_objects = set(_normalize_list(candidate.objects))
	if not candidate_objects:
		return 0.0

	intersection = len(query_objects & candidate_objects)
	union = len(query_objects | candidate_objects)
	if union == 0:
		return None
	return intersection / union


def attribute_score(query_document: QueryDocument, candidate: RetrievedCandidate) -> Tuple[float, ScoreBreakdown]:
	"""Average the available attribute scores without penalizing unspecified fields."""
	garment = garment_score(query_document, candidate)
	scene = scene_score(query_document, candidate)
	style = style_score(query_document, candidate)
	pose = pose_score(query_document, candidate)
	objects = object_score(query_document, candidate)

	breakdown = ScoreBreakdown(garment=garment, scene=scene, style=style, pose=pose, objects=objects)
	available_scores = [score for score in (garment, scene, style, pose, objects) if score is not None]
	if not available_scores:
		return 0.0, breakdown
	return float(sum(available_scores) / len(available_scores)), breakdown


def hybrid_score(vector_score: float, attribute_score_value: float, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> float:
	"""Combine vector similarity and attribute score into a single final score."""
	return alpha * float(vector_score) + beta * float(attribute_score_value)


def _update_candidate_scores(
	candidate: RetrievedCandidate,
	attribute_score_value: float,
	breakdown: ScoreBreakdown,
	alpha: float,
	beta: float,
) -> RetrievedCandidate:
	"""Populate candidate score fields in place for downstream consumers."""
	final_score = hybrid_score(candidate.vector_score, attribute_score_value, alpha=alpha, beta=beta)
	candidate.attribute_score = attribute_score_value
	candidate.final_score = final_score
	candidate.score_breakdown = breakdown.to_dict()
	candidate.score = final_score
	return candidate


def rerank(
	query_document: QueryDocument,
	candidates: Sequence[RetrievedCandidate],
	alpha: float = DEFAULT_ALPHA,
	beta: float = DEFAULT_BETA,
) -> List[RetrievedCandidate]:
	"""Rerank semantic candidates with deterministic structured attribute matching."""
	if not isinstance(query_document, QueryDocument):
		return []

	ranked_candidates: List[RetrievedCandidate] = []
	for candidate in candidates:
		if not isinstance(candidate, RetrievedCandidate):
			continue
		attribute_score_value, breakdown = attribute_score(query_document, candidate)
		ranked_candidates.append(_update_candidate_scores(candidate, attribute_score_value, breakdown, alpha, beta))

	ranked_candidates.sort(key=lambda item: (item.score, item.vector_score, item.id), reverse=True)
	return ranked_candidates

