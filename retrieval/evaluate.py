#!/usr/bin/env python3
"""Evaluation pipeline for multimodal fashion retrieval.

This module orchestrates the existing query parsing, semantic retrieval, and
deterministic reranking stages. It does not build datasets, compute embeddings,
query Qdrant directly, or train models.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
	from retrieval.parse_query import QueryDocument, parse_query
	from retrieval.retrieve import RetrievedCandidate, retrieve
	from retrieval.rerank import rerank
except (ModuleNotFoundError, ImportError):  # pragma: no cover
	from parse_query import QueryDocument, parse_query
	from retrieve import RetrievedCandidate, retrieve
	from rerank import rerank

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_DATASET_FILE = Path("data/processed/evaluation_queries.json")
DEFAULT_DOCUMENTS_FILE = Path("data/processed/documents.json")
DEFAULT_OUTPUT_FILE = Path("data/processed/evaluation_summary.json")
DEFAULT_TOP_K = 100
DEFAULT_QUERY_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_RETRIEVE_MODEL_NAME = "patrickjohncyh/fashion-clip"
DEFAULT_DEVICE = "auto"
DEFAULT_ALPHA = 0.75
DEFAULT_BETA = 0.25


@dataclass(slots=True)
class EvaluationQuery:
	"""A single labeled query from the evaluation dataset."""

	query: str
	relevant: List[str]


@dataclass(slots=True)
class EvaluationMetrics:
	"""Metric bundle for semantic or hybrid rankings."""

	hit_at_1: float
	hit_at_5: float
	hit_at_10: float
	precision_at_5: float
	recall_at_5: float
	mrr: float
	ndcg_at_5: float
	precision_at_10: float
	recall_at_10: float
	ndcg_at_10: float


@dataclass(slots=True)
class PerQueryEvaluation:
	"""Per-query evaluation outputs for debugging and inspection."""

	query: str
	relevant: List[str]
	semantic_top_10: List[str]
	hybrid_top_10: List[str]
	semantic_ranking: List[str]
	hybrid_ranking: List[str]
	semantic_metrics: EvaluationMetrics
	hybrid_metrics: EvaluationMetrics


@dataclass(slots=True)
class EvaluationSummary:
	"""Aggregate evaluation summary across the full dataset."""

	dataset_size: int
	queries_evaluated: int
	semantic: EvaluationMetrics
	hybrid: EvaluationMetrics
	improvement: Dict[str, float] = field(default_factory=dict)
	per_query_results: List[PerQueryEvaluation] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Evaluate retrieval quality for fashion search.")
	parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE, help="Path to evaluation JSON file.")
	parser.add_argument(
		"--documents-file",
		type=Path,
		default=DEFAULT_DOCUMENTS_FILE,
		help="Path to documents.json for indexed image count.",
	)
	parser.add_argument(
		"--output-file",
		type=Path,
		default=DEFAULT_OUTPUT_FILE,
		help="Path to save evaluation_summary.json.",
	)
	parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of semantic candidates to retrieve.")
	parser.add_argument(
		"--query-model-name",
		type=str,
		default=DEFAULT_QUERY_MODEL_NAME,
		help="Model used by parse_query.",
	)
	parser.add_argument(
		"--retrieve-model-name",
		type=str,
		default=DEFAULT_RETRIEVE_MODEL_NAME,
		help="FashionCLIP model used by retrieve().",
	)
	parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["auto", "cuda", "mps", "cpu"])
	parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Vector similarity weight.")
	parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Attribute score weight.")
	return parser.parse_args()


def setup_logging() -> None:
	"""Configure application logging."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _normalize_relevant_list(values: Any) -> List[str]:
	"""Normalize the ground-truth filename list."""
	if not isinstance(values, list):
		return []
	normalized: List[str] = []
	seen = set()
	for value in values:
		filename = Path(str(value)).name.strip()
		if not filename or filename in seen:
			continue
		seen.add(filename)
		normalized.append(filename)
	return normalized


def load_dataset(dataset_file: Path) -> List[EvaluationQuery]:
	"""Load evaluation queries from JSON."""
	if not dataset_file.exists():
		raise FileNotFoundError(f"Missing evaluation dataset: {dataset_file}")

	raw_text = dataset_file.read_text(encoding="utf-8")
	if not raw_text.strip():
		return []

	try:
		data = json.loads(raw_text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Malformed evaluation dataset JSON: {exc}") from exc

	if not isinstance(data, list):
		raise ValueError("Evaluation dataset must be a JSON list.")

	queries: List[EvaluationQuery] = []
	seen_queries = set()
	for index, item in enumerate(data):
		if not isinstance(item, dict):
			logging.warning("Skipping invalid dataset row at index %d: not an object", index)
			continue
		query = str(item.get("query", "")).strip()
		relevant = _normalize_relevant_list(item.get("relevant"))
		if not query or not relevant:
			logging.warning("Skipping invalid dataset row at index %d: missing query or relevant list", index)
			continue
		if query in seen_queries:
			logging.warning("Skipping duplicate evaluation query at index %d: %s", index, query)
			continue
		seen_queries.add(query)
		queries.append(EvaluationQuery(query=query, relevant=relevant))

	return queries


def load_indexed_count(documents_file: Path) -> int:
	"""Load the number of indexed images from documents.json."""
	if not documents_file.exists():
		raise FileNotFoundError(f"Missing indexed documents file: {documents_file}")

	raw_text = documents_file.read_text(encoding="utf-8")
	if not raw_text.strip():
		return 0

	try:
		data = json.loads(raw_text)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Malformed documents JSON: {exc}") from exc

	if not isinstance(data, list):
		raise ValueError("documents.json must contain a JSON list.")
	return len(data)


def precision_at_k(ranked_filenames: Sequence[str], relevant: Sequence[str], k: int) -> float:
	"""Compute Precision@K."""
	if k <= 0:
		return 0.0
	relevant_set = set(relevant)
	if not relevant_set:
		return 0.0
	predicted = _unique_prefix(ranked_filenames, k)
	return len([filename for filename in predicted if filename in relevant_set]) / float(k)


def recall_at_k(ranked_filenames: Sequence[str], relevant: Sequence[str], k: int) -> float:
	"""Compute Recall@K."""
	relevant_set = set(relevant)
	if not relevant_set:
		return 0.0
	predicted = _unique_prefix(ranked_filenames, k)
	return len([filename for filename in predicted if filename in relevant_set]) / float(len(relevant_set))


def hit_rate_at_k(ranked_filenames: Sequence[str], relevant: Sequence[str], k: int) -> float:
	"""Compute Hit Rate@K."""
	relevant_set = set(relevant)
	if not relevant_set:
		return 0.0
	return 1.0 if any(filename in relevant_set for filename in _unique_prefix(ranked_filenames, k)) else 0.0


def reciprocal_rank(ranked_filenames: Sequence[str], relevant: Sequence[str]) -> float:
	"""Compute reciprocal rank for a single ranked list."""
	relevant_set = set(relevant)
	if not relevant_set:
		return 0.0
	for index, filename in enumerate(_unique_prefix(ranked_filenames, len(ranked_filenames)), start=1):
		if filename in relevant_set:
			return 1.0 / float(index)
	return 0.0


def mrr(ranked_filenames: Sequence[str], relevant: Sequence[str]) -> float:
	"""Compute mean reciprocal rank for a single query."""
	return reciprocal_rank(ranked_filenames, relevant)


def dcg(ranked_filenames: Sequence[str], relevant: Sequence[str], k: int) -> float:
	"""Compute discounted cumulative gain at K."""
	if k <= 0:
		return 0.0
	relevant_set = set(relevant)
	score = 0.0
	for index, filename in enumerate(_unique_prefix(ranked_filenames, k), start=1):
		if filename in relevant_set:
			score += 1.0 / math.log2(index + 1)
	return score


def ndcg(ranked_filenames: Sequence[str], relevant: Sequence[str], k: int) -> float:
	"""Compute normalized DCG at K."""
	relevant_set = set(relevant)
	if not relevant_set or k <= 0:
		return 0.0
	actual_dcg = dcg(ranked_filenames, relevant, k)
	ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))
	if ideal_dcg == 0.0:
		return 0.0
	return actual_dcg / ideal_dcg


def _build_metrics(ranked_filenames: Sequence[str], relevant: Sequence[str]) -> EvaluationMetrics:
	"""Compute the full metric bundle for one ranked list."""
	return EvaluationMetrics(
		hit_at_1=hit_rate_at_k(ranked_filenames, relevant, 1),
		hit_at_5=hit_rate_at_k(ranked_filenames, relevant, 5),
		hit_at_10=hit_rate_at_k(ranked_filenames, relevant, 10),
		precision_at_5=precision_at_k(ranked_filenames, relevant, 5),
		recall_at_5=recall_at_k(ranked_filenames, relevant, 5),
		mrr=mrr(ranked_filenames, relevant),
		ndcg_at_5=ndcg(ranked_filenames, relevant, 5),
		precision_at_10=precision_at_k(ranked_filenames, relevant, 10),
		recall_at_10=recall_at_k(ranked_filenames, relevant, 10),
		ndcg_at_10=ndcg(ranked_filenames, relevant, 10),
	)


def _unique_prefix(ranked_filenames: Sequence[str], k: int) -> List[str]:
	"""Return the first k unique filenames while preserving rank order."""
	unique: List[str] = []
	seen = set()
	for filename in ranked_filenames:
		if filename in seen:
			continue
		seen.add(filename)
		unique.append(filename)
		if len(unique) >= k:
			break
	return unique


def evaluate_query(
	query_entry: EvaluationQuery,
	query_model_name: str,
	retrieve_model_name: str,
	top_k: int,
	device: str,
	alpha: float,
	beta: float,
) -> PerQueryEvaluation:
	"""Run the full pipeline for a single evaluation query."""
	query_document = parse_query(
		query_entry.query,
		model_name=query_model_name,
		device=device,
	)

	semantic_candidates = retrieve(
		query_document=query_document,
		top_k=top_k,
		model_name=retrieve_model_name,
		device=device,
	)
	semantic_ranking = [candidate.filename for candidate in semantic_candidates]
	semantic_top_10 = semantic_ranking[:10]
	semantic_metrics = _build_metrics(semantic_ranking, query_entry.relevant)

	hybrid_candidates = rerank(query_document, copy.deepcopy(semantic_candidates), alpha=alpha, beta=beta)
	hybrid_ranking = [candidate.filename for candidate in hybrid_candidates]
	hybrid_top_10 = hybrid_ranking[:10]
	hybrid_metrics = _build_metrics(hybrid_ranking, query_entry.relevant)

	return PerQueryEvaluation(
		query=query_entry.query,
		relevant=list(query_entry.relevant),
		semantic_top_10=semantic_top_10,
		hybrid_top_10=hybrid_top_10,
		semantic_ranking=semantic_ranking,
		hybrid_ranking=hybrid_ranking,
		semantic_metrics=semantic_metrics,
		hybrid_metrics=hybrid_metrics,
	)


def _mean(values: Sequence[float]) -> float:
	"""Compute the arithmetic mean of a sequence."""
	if not values:
		return 0.0
	return float(sum(values) / len(values))


def _aggregate_metrics(items: Sequence[EvaluationMetrics]) -> EvaluationMetrics:
	"""Average a sequence of metric bundles."""
	if not items:
		return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	return EvaluationMetrics(
		hit_at_1=_mean([item.hit_at_1 for item in items]),
		hit_at_5=_mean([item.hit_at_5 for item in items]),
		hit_at_10=_mean([item.hit_at_10 for item in items]),
		precision_at_5=_mean([item.precision_at_5 for item in items]),
		recall_at_5=_mean([item.recall_at_5 for item in items]),
		mrr=_mean([item.mrr for item in items]),
		ndcg_at_5=_mean([item.ndcg_at_5 for item in items]),
		precision_at_10=_mean([item.precision_at_10 for item in items]),
		recall_at_10=_mean([item.recall_at_10 for item in items]),
		ndcg_at_10=_mean([item.ndcg_at_10 for item in items]),
	)


def evaluate_dataset(
	dataset: Sequence[EvaluationQuery],
	indexed_count: int,
	query_model_name: str,
	retrieve_model_name: str,
	top_k: int,
	device: str,
	alpha: float,
	beta: float,
) -> EvaluationSummary:
	"""Evaluate the full dataset and compute aggregate metrics."""
	results: List[PerQueryEvaluation] = []
	loop_start = time.perf_counter()
	for index, query_entry in enumerate(dataset, start=1):
		logging.info("Evaluating query %d/%d", index, len(dataset))
		query_start = time.perf_counter()
		results.append(
			evaluate_query(
				query_entry=query_entry,
				query_model_name=query_model_name,
				retrieve_model_name=retrieve_model_name,
				top_k=top_k,
				device=device,
				alpha=alpha,
				beta=beta,
			)
		)
		logging.info(
			"Completed query %d/%d in %.2fs",
			index,
			len(dataset),
			time.perf_counter() - query_start,
		)
	logging.info("Finished evaluating %d queries in %.2fs", len(results), time.perf_counter() - loop_start)

	semantic_metrics = _aggregate_metrics([result.semantic_metrics for result in results])
	hybrid_metrics = _aggregate_metrics([result.hybrid_metrics for result in results])
	improvement = {
		"hit_at_1": _percent_change(semantic_metrics.hit_at_1, hybrid_metrics.hit_at_1),
		"hit_at_5": _percent_change(semantic_metrics.hit_at_5, hybrid_metrics.hit_at_5),
		"hit_at_10": _percent_change(semantic_metrics.hit_at_10, hybrid_metrics.hit_at_10),
		"precision_at_5": _percent_change(semantic_metrics.precision_at_5, hybrid_metrics.precision_at_5),
		"recall_at_5": _percent_change(semantic_metrics.recall_at_5, hybrid_metrics.recall_at_5),
		"mrr": _percent_change(semantic_metrics.mrr, hybrid_metrics.mrr),
		"ndcg_at_5": _percent_change(semantic_metrics.ndcg_at_5, hybrid_metrics.ndcg_at_5),
		"precision_at_10": _percent_change(semantic_metrics.precision_at_10, hybrid_metrics.precision_at_10),
		"recall_at_10": _percent_change(semantic_metrics.recall_at_10, hybrid_metrics.recall_at_10),
		"ndcg_at_10": _percent_change(semantic_metrics.ndcg_at_10, hybrid_metrics.ndcg_at_10),
	}

	return EvaluationSummary(
		dataset_size=indexed_count,
		queries_evaluated=len(results),
		semantic=semantic_metrics,
		hybrid=hybrid_metrics,
		improvement=improvement,
		per_query_results=results,
	)


def _percent_change(baseline: float, improved: float) -> float:
	"""Compute relative percentage change, handling zero baselines safely."""
	if baseline == 0.0:
		return 0.0 if improved == 0.0 else 100.0
	return ((improved - baseline) / baseline) * 100.0


def print_summary(summary: EvaluationSummary) -> None:
	"""Print per-query analysis and the aggregate evaluation summary."""
	print("=" * 52)
	print()
	print("Evaluation Summary")
	print()
	print(f"Dataset Size")
	print(f"{summary.dataset_size} indexed images")
	print()
	print(f"Evaluation Queries")
	print(f"{summary.queries_evaluated}")
	print()
	print("Semantic Retrieval")
	_print_metric_block(summary.semantic)
	print()
	print("Hybrid Retrieval")
	_print_metric_block(summary.hybrid)
	print()
	print("Improvement")
	for label, key in (
		("Hit@1", "hit_at_1"),
		("Hit@5", "hit_at_5"),
		("Hit@10", "hit_at_10"),
		("Precision@5", "precision_at_5"),
		("Recall@5", "recall_at_5"),
		("MRR", "mrr"),
		("nDCG@5", "ndcg_at_5"),
		("Precision@10", "precision_at_10"),
		("Recall@10", "recall_at_10"),
		("nDCG@10", "ndcg_at_10"),
	):
		print(f"{label:<12} {summary.improvement[key]:+.1f}%")
	print()
	print("=" * 52)
	print()

	for index, result in enumerate(summary.per_query_results, start=1):
		print(f"Query {index}")
		print(result.query)
		print()
		print("Relevant images")
		print(", ".join(result.relevant))
		print()
		print("Semantic Top-10")
		print(", ".join(result.semantic_top_10))
		print()
		print("Hybrid Top-10")
		print(", ".join(result.hybrid_top_10))
		print()
		print("Semantic Metrics")
		_print_metric_block(result.semantic_metrics)
		print()
		print("Hybrid Metrics")
		_print_metric_block(result.hybrid_metrics)
		print("-" * 60)


def save_results(summary: EvaluationSummary, output_file: Path) -> None:
	"""Persist the evaluation summary and per-query metrics to JSON."""
	logging.info("Saving evaluation results to %s", output_file)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"dataset_size": summary.dataset_size,
		"queries_evaluated": summary.queries_evaluated,
		"semantic": asdict(summary.semantic),
		"hybrid": asdict(summary.hybrid),
		"improvement": summary.improvement,
		"per_query_results": [
			{
				"query": item.query,
				"relevant": item.relevant,
				"semantic_top_10": item.semantic_top_10,
				"hybrid_top_10": item.hybrid_top_10,
				"semantic_ranking": item.semantic_ranking,
				"hybrid_ranking": item.hybrid_ranking,
				"semantic_metrics": asdict(item.semantic_metrics),
				"hybrid_metrics": asdict(item.hybrid_metrics),
			}
			for item in summary.per_query_results
		],
	}
	with output_file.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=False)
	logging.info("Saved evaluation results.")


def _print_metric_block(metrics: EvaluationMetrics) -> None:
	"""Print a metric bundle in the requested aligned format."""
	print(f"{'Hit@1':<12}{metrics.hit_at_1:.2f}")
	print(f"{'Hit@5':<12}{metrics.hit_at_5:.2f}")
	print(f"{'Hit@10':<12}{metrics.hit_at_10:.2f}")
	print(f"{'Precision@5':<12}{metrics.precision_at_5:.2f}")
	print(f"{'Recall@5':<12}{metrics.recall_at_5:.2f}")
	print(f"{'MRR':<12}{metrics.mrr:.2f}")
	print(f"{'nDCG@5':<12}{metrics.ndcg_at_5:.2f}")
	print(f"{'Precision@10':<12}{metrics.precision_at_10:.2f}")
	print(f"{'Recall@10':<12}{metrics.recall_at_10:.2f}")
	print(f"{'nDCG@10':<12}{metrics.ndcg_at_10:.2f}")


def main() -> None:
	"""Run evaluation over the configured dataset and print the summary."""
	setup_logging()
	args = parse_args()
	if args.alpha < 0 or args.beta < 0:
		raise ValueError("alpha and beta must be non-negative")
	weight_total = args.alpha + args.beta
	if weight_total <= 0:
		raise ValueError("alpha + beta must be greater than zero")
	alpha = args.alpha / weight_total
	beta = args.beta / weight_total
	dataset = load_dataset(args.dataset_file)
	indexed_count = load_indexed_count(args.documents_file)
	if not dataset:
		logging.warning("No evaluation queries found in %s", args.dataset_file)
		return

	summary = evaluate_dataset(
		dataset=dataset,
		indexed_count=indexed_count,
		query_model_name=args.query_model_name,
		retrieve_model_name=args.retrieve_model_name,
		top_k=args.top_k,
		device=args.device,
		alpha=alpha,
		beta=beta,
	)
	print_summary(summary)
	save_results(summary, args.output_file)


if __name__ == "__main__":
	main()
