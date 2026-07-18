#!/usr/bin/env python3
"""Build an initial retrieval evaluation dataset from indexed fashion metadata.

The script reads ``data/processed/documents.json``, generates deterministic
natural-language queries from the existing metadata, computes the relevant
filenames for each query by exact normalized matching, and writes the resulting
dataset to ``data/processed/evaluation_queries.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_FILE = PROJECT_ROOT / "data/processed/documents.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data/processed/evaluation_queries.json"
DEFAULT_SEED = 20260718
MIN_RELEVANT = 2
MAX_RELEVANT = 40
TARGET_COUNTS = {
	"garment": 20,
	"scene": 15,
	"style": 15,
	"object": 10,
	"multi": 40,
}
MAX_SIMPLE_POOL = 64
MAX_MULTI_GARMENTS = 20
MAX_MULTI_SCENES = 10
MAX_MULTI_STYLES = 10
MAX_MULTI_OBJECTS = 12

OBJECT_QUERY_TERMS = (
	"backpack",
	"handbag",
	"purse",
	"bag",
	"laptop",
	"tablet",
	"phone",
	"book",
	"notebook",
	"camera",
	"sunglasses",
	"cup of coffee",
	"coffee mug",
	"mug",
	"chair",
	"desk",
	"watch",
	"bottle",
	"pen",
	"umbrella",
)


@dataclass(frozen=True, slots=True)
class GarmentCondition:
	"""Normalized garment attributes used for exact matching."""

	type: Optional[str] = None
	color: Optional[str] = None
	pattern: Optional[str] = None


@dataclass(frozen=True, slots=True)
class DocumentRecord:
	"""Normalized document metadata loaded from ``documents.json``."""

	filename: str
	scene: Optional[str]
	style: Optional[str]
	pose: Optional[str]
	objects: Tuple[str, ...]
	garments: Tuple[GarmentCondition, ...]


@dataclass(frozen=True, slots=True)
class QueryCandidate:
	"""A generated query and the structured constraints used to match it."""

	category: str
	query: str
	garment: Optional[GarmentCondition] = None
	scene: Optional[str] = None
	style: Optional[str] = None
	objects: Tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Build a retrieval evaluation dataset from documents.json.")
	parser.add_argument("--documents-file", type=Path, default=DEFAULT_DOCUMENTS_FILE, help="Path to documents.json.")
	parser.add_argument(
		"--output-file",
		type=Path,
		default=DEFAULT_OUTPUT_FILE,
		help="Path to write evaluation_queries.json.",
	)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic shuffle seed.")
	return parser.parse_args()


def setup_logging() -> None:
	"""Configure logging for a concise command-line experience."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize_text(value: str) -> str:
	"""Normalize text for exact matching."""
	text = unicodedata.normalize("NFKC", value).lower().strip()
	text = text.translate(str.maketrans({character: " " for character in string.punctuation}))
	text = re.sub(r"\s+", " ", text).strip()
	return text


def normalize_optional_text(value: Any) -> Optional[str]:
	"""Normalize an optional text field and treat empty markers as missing."""
	if value is None:
		return None
	text = normalize_text(str(value))
	if not text or text in {"none", "null"}:
		return None
	return text


def normalize_string_list(value: Any) -> Tuple[str, ...]:
	"""Normalize a list of strings while preserving order and removing duplicates."""
	if not isinstance(value, list):
		return ()
	normalized: List[str] = []
	seen = set()
	for item in value:
		if not isinstance(item, str):
			continue
		text = normalize_text(item)
		if not text or text in seen:
			continue
		seen.add(text)
		normalized.append(text)
	return tuple(normalized)


def normalize_garments(value: Any) -> Tuple[GarmentCondition, ...]:
	"""Normalize garment dictionaries into typed garment conditions."""
	if not isinstance(value, list):
		return ()
	garments: List[GarmentCondition] = []
	for item in value:
		if not isinstance(item, dict):
			continue
		garment = GarmentCondition(
			type=normalize_optional_text(item.get("type")),
			color=normalize_optional_text(item.get("color")),
			pattern=normalize_optional_text(item.get("pattern")),
		)
		if garment.type is None and garment.color is None and garment.pattern is None:
			continue
		garments.append(garment)
	return tuple(garments)


def load_documents(documents_file: Path) -> List[DocumentRecord]:
	"""Load and normalize the indexed documents."""
	if not documents_file.exists():
		raise FileNotFoundError(f"Missing documents file: {documents_file}")

	raw_text = documents_file.read_text(encoding="utf-8")
	if not raw_text.strip():
		raise ValueError(f"Empty JSON file: {documents_file}")

	try:
		payload = json.loads(raw_text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Malformed JSON in {documents_file}: {exc}") from exc

	if not isinstance(payload, list):
		raise ValueError("documents.json must contain a JSON list.")

	documents: List[DocumentRecord] = []
	for index, item in enumerate(payload):
		if not isinstance(item, dict):
			logging.warning("Skipping invalid document at index %d: not an object", index)
			continue

		filename = Path(str(item.get("filename", "")).strip()).name
		if not filename:
			logging.warning("Skipping invalid document at index %d: missing filename", index)
			continue

		documents.append(
			DocumentRecord(
				filename=filename,
				scene=normalize_optional_text(item.get("scene")),
				style=normalize_optional_text(item.get("style")),
				pose=normalize_optional_text(item.get("pose")),
				objects=normalize_string_list(item.get("objects", [])),
				garments=normalize_garments(item.get("garments", [])),
			),
		)

	return documents


def build_counters(documents: Sequence[DocumentRecord]) -> Dict[str, Counter]:
	"""Build frequency counters used for candidate selection."""
	counters: Dict[str, Counter] = {
		"scene": Counter(),
		"style": Counter(),
		"object": Counter(),
		"garment_type": Counter(),
		"garment_type_color": Counter(),
		"garment_type_pattern": Counter(),
		"garment_type_color_pattern": Counter(),
	}

	for document in documents:
		if document.scene:
			counters["scene"][document.scene] += 1
		if document.style:
			counters["style"][document.style] += 1
		for object_term in document.objects:
			counters["object"][object_term] += 1
		for garment in document.garments:
			if garment.type is None:
				continue
			counters["garment_type"][garment.type] += 1
			if garment.color:
				counters["garment_type_color"][(garment.type, garment.color)] += 1
			if garment.pattern:
				counters["garment_type_pattern"][(garment.type, garment.pattern)] += 1
			if garment.color and garment.pattern:
				counters["garment_type_color_pattern"][(garment.type, garment.color, garment.pattern)] += 1

	return counters


def ranked_items(
	counter: Counter,
	*,
	limit: Optional[int] = None,
	minimum: Optional[int] = None,
	maximum: Optional[int] = None,
) -> List[Tuple[Any, int]]:
	"""Return counter items sorted by frequency and filtered by count bounds."""
	items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
	selected: List[Tuple[Any, int]] = []
	for key, count in items:
		if minimum is not None and count < minimum:
			continue
		if maximum is not None and count > maximum:
			continue
		selected.append((key, count))
		if limit is not None and len(selected) >= limit:
			break
	return selected


def render_garment_phrase(garment: GarmentCondition) -> str:
	"""Render a garment condition into a natural-language phrase."""
	parts: List[str] = []
	if garment.color:
		parts.append(garment.color)
	if garment.pattern:
		parts.append(garment.pattern)
	if garment.type:
		parts.append(garment.type)
	return " ".join(parts)


def render_object_phrase(term: str) -> str:
	"""Render an object term into a natural-language object query."""
	if "sunglasses" in term:
		return f"wearing {term}"
	if any(keyword in term for keyword in ("backpack", "handbag", "purse", "bag")):
		return f"carrying {term}"
	if any(keyword in term for keyword in ("laptop", "tablet", "phone", "camera", "book", "notebook", "mug", "cup")):
		return f"holding {term}"
	return f"person with {term}"


def garment_matches(document_garment: GarmentCondition, garment_condition: GarmentCondition) -> bool:
	"""Check whether a document garment satisfies a garment query condition."""
	if garment_condition.type and document_garment.type != garment_condition.type:
		return False
	if garment_condition.color and document_garment.color != garment_condition.color:
		return False
	if garment_condition.pattern and document_garment.pattern != garment_condition.pattern:
		return False
	return True


def document_matches(document: DocumentRecord, candidate: QueryCandidate) -> bool:
	"""Check whether a document satisfies all constraints in a query candidate."""
	if candidate.scene and document.scene != candidate.scene:
		return False
	if candidate.style and document.style != candidate.style:
		return False
	if candidate.objects:
		document_objects = set(document.objects)
		if any(object_term not in document_objects for object_term in candidate.objects):
			return False
	if candidate.garment is not None:
		if not any(garment_matches(document_garment, candidate.garment) for document_garment in document.garments):
			return False
	return True


def compute_relevant_filenames(documents: Sequence[DocumentRecord], candidate: QueryCandidate) -> List[str]:
	"""Compute the sorted list of relevant filenames for a query candidate."""
	relevant = [document.filename for document in documents if document_matches(document, candidate)]
	return sorted(relevant)


def candidate_is_valid(relevant: Sequence[str]) -> bool:
	"""Return ``True`` when a candidate has an acceptable number of matches."""
	return MIN_RELEVANT <= len(relevant) <= MAX_RELEVANT


def dedupe_candidates(candidates: Sequence[QueryCandidate]) -> List[QueryCandidate]:
	"""Remove duplicate queries using the evaluation dataset normalization rule."""
	unique: List[QueryCandidate] = []
	seen_queries = set()
	for candidate in candidates:
		query_key = candidate.query.lower().strip()
		if query_key in seen_queries:
			continue
		seen_queries.add(query_key)
		unique.append(candidate)
	return unique


def select_query_friendly_objects(counter: Counter, *, limit: int) -> List[str]:
	"""Select query-friendly object terms that have a useful relevance window."""
	selected: List[str] = []
	seen = set()
	for term in OBJECT_QUERY_TERMS:
		if term in seen:
			continue
		count = counter.get(term, 0)
		if MIN_RELEVANT <= count <= MAX_RELEVANT:
			selected.append(term)
			seen.add(term)
		if len(selected) >= limit:
			return selected

	for term, count in ranked_items(counter, minimum=MIN_RELEVANT, maximum=MAX_RELEVANT):
		if not isinstance(term, str):
			continue
		if term in seen:
			continue
		if len(term.split()) > 4:
			continue
		if any(keyword in term for keyword in ("woman", "man", "girl", "boy", "person", "people", "shirt", "dress", "jacket", "pants", "jeans", "skirt", "coat", "sweater", "blouse", "suit", "shoes", "boots", "sneakers", "hoodie", "tie", "hat", "scarf", "glasses", "sunglasses")):
			continue
		selected.append(term)
		seen.add(term)
		if len(selected) >= limit:
			break

	return selected


def build_garment_candidates(counters: Dict[str, Counter]) -> List[QueryCandidate]:
	"""Build garment-focused query candidates from observed garment combinations."""
	candidates: List[QueryCandidate] = []

	for (garment_type, color), _count in ranked_items(counters["garment_type_color"], minimum=MIN_RELEVANT, maximum=MAX_RELEVANT, limit=MAX_SIMPLE_POOL):
		condition = GarmentCondition(type=garment_type, color=color)
		candidates.append(QueryCandidate(category="garment", query=render_garment_phrase(condition), garment=condition))

	for (garment_type, pattern), _count in ranked_items(counters["garment_type_pattern"], minimum=MIN_RELEVANT, maximum=MAX_RELEVANT, limit=MAX_SIMPLE_POOL):
		condition = GarmentCondition(type=garment_type, pattern=pattern)
		candidates.append(QueryCandidate(category="garment", query=render_garment_phrase(condition), garment=condition))

	for (garment_type, color, pattern), _count in ranked_items(counters["garment_type_color_pattern"], minimum=MIN_RELEVANT, maximum=MAX_RELEVANT, limit=MAX_SIMPLE_POOL):
		condition = GarmentCondition(type=garment_type, color=color, pattern=pattern)
		candidates.append(QueryCandidate(category="garment", query=render_garment_phrase(condition), garment=condition))

	for garment_type, _count in ranked_items(counters["garment_type"], minimum=MIN_RELEVANT, maximum=MAX_RELEVANT, limit=MAX_SIMPLE_POOL):
		condition = GarmentCondition(type=garment_type)
		candidates.append(QueryCandidate(category="garment", query=render_garment_phrase(condition), garment=condition))

	return dedupe_candidates(candidates)


def build_scene_candidates(counters: Dict[str, Counter]) -> List[QueryCandidate]:
	"""Build scene-focused query candidates from observed scene values."""
	candidates: List[QueryCandidate] = []
	for scene, _count in ranked_items(counters["scene"], minimum=MIN_RELEVANT, maximum=MAX_RELEVANT, limit=MAX_SIMPLE_POOL):
		candidates.append(QueryCandidate(category="scene", query=scene, scene=scene))
	return dedupe_candidates(candidates)


def build_style_candidates(counters: Dict[str, Counter]) -> List[QueryCandidate]:
	"""Build style-focused query candidates from observed style values."""
	candidates: List[QueryCandidate] = []
	for style, _count in ranked_items(counters["style"], minimum=MIN_RELEVANT, maximum=MAX_RELEVANT, limit=MAX_SIMPLE_POOL):
		candidates.append(QueryCandidate(category="style", query=style, style=style))
	return dedupe_candidates(candidates)


def build_object_candidates(counters: Dict[str, Counter]) -> List[QueryCandidate]:
	"""Build object-focused query candidates from query-friendly object terms."""
	terms = select_query_friendly_objects(counters["object"], limit=MAX_SIMPLE_POOL)
	candidates = [QueryCandidate(category="object", query=render_object_phrase(term), objects=(term,)) for term in terms]
	return dedupe_candidates(candidates)


def build_multi_attribute_candidates(
	garment_candidates: Sequence[QueryCandidate],
	scene_candidates: Sequence[QueryCandidate],
	style_candidates: Sequence[QueryCandidate],
	object_candidates: Sequence[QueryCandidate],
) -> List[QueryCandidate]:
	"""Build multi-attribute query candidates using deterministic templates."""
	selected_garments = [candidate.garment for candidate in garment_candidates if candidate.garment is not None][:MAX_MULTI_GARMENTS]
	selected_scenes = [candidate.scene for candidate in scene_candidates if candidate.scene is not None][:MAX_MULTI_SCENES]
	selected_styles = [candidate.style for candidate in style_candidates if candidate.style is not None][:MAX_MULTI_STYLES]
	selected_objects = [candidate.objects[0] for candidate in object_candidates if candidate.objects][:MAX_MULTI_OBJECTS]

	candidates: List[QueryCandidate] = []

	for garment in selected_garments:
		garment_phrase = render_garment_phrase(garment)
		for scene in selected_scenes:
			candidates.append(
				QueryCandidate(
					category="multi",
					query=f"{garment_phrase} in {scene}",
					garment=garment,
					scene=scene,
				),
			)
		for style in selected_styles:
			candidates.append(
				QueryCandidate(
					category="multi",
					query=f"{style} {garment_phrase}",
					garment=garment,
					style=style,
				),
			)
		for object_term in selected_objects:
			candidates.append(
				QueryCandidate(
					category="multi",
					query=f"{garment_phrase} with {object_term}",
					garment=garment,
					objects=(object_term,),
				),
			)

	for garment in selected_garments:
		garment_phrase = render_garment_phrase(garment)
		for scene in selected_scenes[: max(1, len(selected_scenes) // 2)]:
			for style in selected_styles[: max(1, len(selected_styles) // 2)]:
				candidates.append(
					QueryCandidate(
						category="multi",
						query=f"{style} {garment_phrase} in {scene}",
						garment=garment,
						scene=scene,
						style=style,
					),
				)

	for garment in selected_garments[: max(1, len(selected_garments) // 2)]:
		garment_phrase = render_garment_phrase(garment)
		for scene in selected_scenes[: max(1, len(selected_scenes) // 2)]:
			for object_term in selected_objects[: max(1, len(selected_objects) // 2)]:
				candidates.append(
					QueryCandidate(
						category="multi",
						query=f"{garment_phrase} in {scene} with {object_term}",
						garment=garment,
						scene=scene,
						objects=(object_term,),
					),
				)

	for garment in selected_garments[: max(1, len(selected_garments) // 2)]:
		garment_phrase = render_garment_phrase(garment)
		for style in selected_styles[: max(1, len(selected_styles) // 2)]:
			for object_term in selected_objects[: max(1, len(selected_objects) // 2)]:
				candidates.append(
					QueryCandidate(
						category="multi",
						query=f"{style} {garment_phrase} with {object_term}",
						garment=garment,
						style=style,
						objects=(object_term,),
					),
				)

	for garment in selected_garments[: max(1, len(selected_garments) // 3)]:
		garment_phrase = render_garment_phrase(garment)
		for scene in selected_scenes[: max(1, len(selected_scenes) // 3)]:
			for style in selected_styles[: max(1, len(selected_styles) // 3)]:
				for object_term in selected_objects[: max(1, len(selected_objects) // 3)]:
					candidates.append(
						QueryCandidate(
							category="multi",
							query=f"{style} {garment_phrase} in {scene} with {object_term}",
							garment=garment,
							scene=scene,
							style=style,
							objects=(object_term,),
						),
					)

	return dedupe_candidates(candidates)


def build_candidates(counters: Dict[str, Counter]) -> Dict[str, List[QueryCandidate]]:
	"""Build candidate pools for every query category."""
	garment_candidates = build_garment_candidates(counters)
	scene_candidates = build_scene_candidates(counters)
	style_candidates = build_style_candidates(counters)
	object_candidates = build_object_candidates(counters)
	multi_candidates = build_multi_attribute_candidates(garment_candidates, scene_candidates, style_candidates, object_candidates)
	return {
		"garment": garment_candidates,
		"scene": scene_candidates,
		"style": style_candidates,
		"object": object_candidates,
		"multi": multi_candidates,
	}


def validate_candidates(documents: Sequence[DocumentRecord], candidates: Sequence[QueryCandidate]) -> List[Dict[str, Any]]:
	"""Compute relevance lists, enforce quality filters, and remove duplicates."""
	validated: List[Dict[str, Any]] = []
	seen_queries = set()
	seen_relevant_lists = set()

	for candidate in candidates:
		query_key = candidate.query.lower().strip()
		if query_key in seen_queries:
			continue

		relevant = compute_relevant_filenames(documents, candidate)
		if not candidate_is_valid(relevant):
			continue

		relevant_key = tuple(relevant)
		if relevant_key in seen_relevant_lists:
			continue

		seen_queries.add(query_key)
		seen_relevant_lists.add(relevant_key)
		validated.append({"query": candidate.query, "relevant": relevant, "category": candidate.category})

	return validated


def select_final_dataset(validated: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
	"""Select a balanced final dataset from validated per-category candidates."""
	final_dataset: List[Dict[str, Any]] = []
	counts = {category: 0 for category in TARGET_COUNTS}
	seen_queries = set()
	seen_relevant_lists = set()

	for category in ("garment", "scene", "style", "object", "multi"):
		target = TARGET_COUNTS[category]
		for item in validated.get(category, []):
			query_key = item["query"].lower().strip()
			relevant_key = tuple(item["relevant"])
			if query_key in seen_queries or relevant_key in seen_relevant_lists:
				continue
			seen_queries.add(query_key)
			seen_relevant_lists.add(relevant_key)
			final_dataset.append({"query": item["query"], "relevant": item["relevant"]})
			counts[category] += 1
			if counts[category] >= target:
				break

	return final_dataset, counts


def shuffle_dataset(dataset: List[Dict[str, Any]], seed: int) -> None:
	"""Shuffle the final dataset deterministically."""
	random.Random(seed).shuffle(dataset)


def save_dataset(output_file: Path, dataset: Sequence[Dict[str, Any]]) -> None:
	"""Persist the evaluation dataset to disk as formatted JSON."""
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(json.dumps(list(dataset), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_statistics(documents_count: int, counts: Dict[str, int], output_file: Path) -> None:
	"""Print generation statistics in the requested summary format."""
	print(f"Loaded documents: {documents_count}")
	print()
	print(f"Generated garment queries: {counts.get('garment', 0)}")
	print()
	print(f"Generated scene queries: {counts.get('scene', 0)}")
	print()
	print(f"Generated style queries: {counts.get('style', 0)}")
	print()
	print(f"Generated object queries: {counts.get('object', 0)}")
	print()
	print(f"Generated multi-attribute queries: {counts.get('multi', 0)}")
	print()
	total = sum(counts.get(category, 0) for category in TARGET_COUNTS)
	print(f"Total evaluation queries: {total}")
	print()
	print("Saved to:")
	print()
	try:
		print(str(output_file.relative_to(PROJECT_ROOT)))
	except ValueError:
		print(str(output_file))


def main() -> None:
	"""Run the dataset-building pipeline."""
	setup_logging()
	args = parse_args()
	documents = load_documents(args.documents_file)
	counters = build_counters(documents)
	candidate_pools = build_candidates(counters)
	validated_pools = {
		category: validate_candidates(documents, candidates)
		for category, candidates in candidate_pools.items()
	}
	dataset, counts = select_final_dataset(validated_pools)
	if len(dataset) < sum(TARGET_COUNTS.values()):
		logging.warning(
			"Final dataset contains %d queries, below the nominal target of %d.",
			len(dataset),
			sum(TARGET_COUNTS.values()),
		)
	shuffle_dataset(dataset, args.seed)
	save_dataset(args.output_file, dataset)
	print_statistics(len(documents), counts, args.output_file)


if __name__ == "__main__":
	main()
