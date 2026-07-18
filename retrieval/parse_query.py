#!/usr/bin/env python3
"""Convert a natural-language fashion query into a canonical QueryDocument.

This module performs query understanding only. It does not embed queries,
query Qdrant, retrieve candidates, or rerank results.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_NUM_BEAMS = 1

VALID_SCENES = {
	"office",
	"street",
	"park",
	"home",
	"cafe",
	"mall",
	"beach",
	"airport",
	"station",
	"restaurant",
	"indoor",
	"outdoor",
}

VALID_STYLES = {
	"formal",
	"casual",
	"business formal",
	"business casual",
	"streetwear",
	"party",
	"athleisure",
}

PERSON_TERMS = {
	"woman",
	"women",
	"man",
	"men",
	"person",
	"people",
	"girl",
	"boy",
	"lady",
	"guy",
	"ladies",
	"gentleman",
	"gentlemen",
	"female",
	"male",
	"model",
	"adult",
	"human",
}

GARMENT_TERMS = {
	"shirt",
	"tshirt",
	"tee",
	"top",
	"blouse",
	"dress",
	"skirt",
	"pants",
	"trousers",
	"jeans",
	"shorts",
	"blazer",
	"jacket",
	"coat",
	"hoodie",
	"sweater",
	"cardigan",
	"suit",
	"suit jacket",
	"shirt dress",
	"tank top",
	"tank",
	"polo",
	"sweatshirt",
	"vest",
	"leggings",
	"jumpsuit",
	"romper",
	"overalls",
	"scarf",
	"hat",
	"cap",
	"beanie",
	"gloves",
	"shoes",
	"sneakers",
	"boots",
	"heels",
	"sandals",
	"bag",
	"handbag",
	"purse",
	"backpack",
	"belt",
}

OBJECT_EXCLUSION_TERMS = PERSON_TERMS | GARMENT_TERMS

SCENE_ALIASES: Sequence[Tuple[str, str]] = (
	("office", "office"),
	("street", "street"),
	("studio", "studio"),
	("store", "store"),
	("shop", "shop"),
	("home", "home"),
	("indoor", "indoor"),
	("outdoor", "outdoor"),
	("beach", "beach"),
	("park", "park"),
	("restaurant", "restaurant"),
	("cafe", "cafe"),
	("gym", "gym"),
	("bedroom", "bedroom"),
	("kitchen", "kitchen"),
	("bathroom", "bathroom"),
	("runway", "runway"),
	("stage", "stage"),
)


class GarmentQuery(BaseModel):
	"""A structured garment mention extracted from a query."""

	model_config = ConfigDict(extra="forbid")

	type: Optional[str] = None
	color: Optional[str] = None
	pattern: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		"""Convert the garment query into a JSON-serializable dictionary."""
		return self.model_dump()

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "GarmentQuery":
		"""Validate and construct a garment query from a dictionary."""
		return cls.model_validate(payload)

	@model_validator(mode="before")
	@classmethod
	def _normalize_null_fields(cls, data: Any) -> Any:
		"""Accept null garment fields emitted by the LLM."""
		if data is None:
			return {}
		if isinstance(data, dict):
			return {
				"type": data.get("type"),
				"color": data.get("color"),
				"pattern": data.get("pattern"),
			}
		return data


class QueryDocument(BaseModel):
	"""Canonical query representation aligned with indexed ImageDocument objects."""

	model_config = ConfigDict(extra="forbid")

	raw_query: str
	scene: Optional[str] = None
	style: Optional[str] = None
	pose: Optional[str] = None
	objects: List[str] = Field(default_factory=list)
	garments: List[GarmentQuery] = Field(default_factory=list)
	embedding_text: str = ""

	def to_dict(self) -> Dict[str, Any]:
		"""Convert the query document into a JSON-serializable dictionary."""
		data = self.model_dump()
		data["garments"] = [garment.to_dict() for garment in self.garments]
		return data

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "QueryDocument":
		"""Validate and construct a query document from a dictionary."""
		return cls.model_validate(payload)

	@model_validator(mode="before")
	@classmethod
	def _normalize_null_arrays(cls, data: Any) -> Any:
		"""Convert null list fields into empty lists before Pydantic validation."""
		if not isinstance(data, dict):
			return data
		normalized = dict(data)
		if normalized.get("objects") is None:
			normalized["objects"] = []
		if normalized.get("garments") is None:
			normalized["garments"] = []
		return normalized


def _normalize_text(value: str) -> str:
	"""Normalize text for rule-based filtering."""
	text = unicodedata.normalize("NFKC", value).lower().strip()
	text = text.translate(str.maketrans({punctuation: " " for punctuation in string.punctuation}))
	text = re.sub(r"\s+", " ", text).strip()
	return text


class QueryExtractionRecord(BaseModel):
	"""Internal validation model for LLM output."""

	model_config = ConfigDict(extra="forbid")

	scene: Optional[str] = None
	style: Optional[str] = None
	pose: Optional[str] = None
	objects: List[str] = Field(default_factory=list)
	garments: List[GarmentQuery] = Field(default_factory=list)

	@model_validator(mode="before")
	@classmethod
	def _normalize_null_arrays(cls, data: Any) -> Any:
		"""Convert null list fields into empty lists before validation."""
		if not isinstance(data, dict):
			return data
		normalized = dict(data)
		if normalized.get("objects") is None:
			normalized["objects"] = []
		if normalized.get("garments") is None:
			normalized["garments"] = []
		return normalized


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Parse a fashion retrieval query into structured metadata.")
	parser.add_argument("--query", type=str, default=None, help="Optional query to parse non-interactively.")
	parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model name.")
	parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retry count for invalid JSON.")
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=DEFAULT_MAX_NEW_TOKENS,
		help="Maximum tokens to generate.",
	)
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


@lru_cache(maxsize=1)
def _load_model_cached(model_name: str, device_name: str) -> Tuple[Any, Any, torch.device]:
	"""Load the tokenizer and causal LLM once per process."""
	device = detect_device(device_name)
	logging.info("Loading model: %s", model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		trust_remote_code=True,
		torch_dtype=get_torch_dtype(device),
	)
	model.to(device)
	model.eval()
	return tokenizer, model, device


def load_model(model_name: str, device: torch.device) -> Tuple[Any, Any, torch.device]:
	"""Load the tokenizer and model, caching the result for reuse."""
	return _load_model_cached(model_name, str(device))


def normalize_query(query: str) -> str:
	"""Normalize free-form query text for stable prompting and embedding."""
	return _normalize_text(query)


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
	"""Normalize an optional categorical label."""
	if value is None:
		return None
	text = _normalize_text(str(value))
	return text or None


def normalize_scene_label(value: Optional[str]) -> Optional[str]:
	"""Normalize scene labels to concise canonical forms when possible."""
	text = normalize_optional_text(value)
	if text is None:
		return None

	for needle, canonical in SCENE_ALIASES:
		if needle in text:
			return canonical if canonical in VALID_SCENES else None
	return text if text in VALID_SCENES else None


def normalize_style_label(value: Optional[str]) -> Optional[str]:
	"""Normalize style labels and enforce the closed vocabulary."""
	text = normalize_optional_text(value)
	if text is None:
		return None
	return text if text in VALID_STYLES else None


def _phrase_in_text(phrase: str, text: str) -> bool:
	"""Check whether a phrase is present as a normalized substring."""
	return bool(phrase) and phrase in text


def _is_object_allowed(object_text: str, garment_types: Sequence[str]) -> bool:
	"""Keep only non-garment, non-person object mentions using exact word-boundary matching."""
	normalized = _normalize_text(object_text)
	if not normalized:
		return False
		
	# Convert current garment types to lower case for comparison
	normalized_garment_types = {g.lower() for g in garment_types}
	if normalized in normalized_garment_types:
		return False

	# Tokenize into individual words to prevent bad substring matches (e.g., dropping "laptop" because of "cap")
	words = set(normalized.split())

	# Ensure none of the individual words are restricted words
	if not words.isdisjoint(PERSON_TERMS):
		return False
	if not words.isdisjoint(GARMENT_TERMS):
		return False

	return True


def _is_explicit_garment_type(garment_type: str, query_text: str) -> bool:
	"""Keep only garment types that are explicitly stated in the query."""
	normalized = _normalize_text(garment_type)
	if not normalized:
		return False
	if normalized in PERSON_TERMS:
		return False
	if normalized not in GARMENT_TERMS:
		return False
	return _phrase_in_text(normalized, query_text)


def normalize_string_list(values: Sequence[Any]) -> List[str]:
	"""Normalize a list of strings while removing duplicates."""
	normalized: List[str] = []
	seen = set()
	for value in values:
		if not isinstance(value, str):
			continue
		text = normalize_optional_text(value)
		if text is None or text in seen:
			continue
		seen.add(text)
		normalized.append(text)
	return normalized


def normalize_garments(values: Sequence[Any], query_text: str) -> List[GarmentQuery]:
	"""Normalize garment records and remove duplicates."""
	garments: List[GarmentQuery] = []
	seen = set()
	for value in values:
		if isinstance(value, GarmentQuery):
			garment = value
		elif isinstance(value, dict):
			garment = GarmentQuery.model_validate(value)
		else:
			continue

		normalized = GarmentQuery(
			type=normalize_optional_text(garment.type),
			color=normalize_optional_text(garment.color),
			pattern=normalize_optional_text(garment.pattern),
		)
		if normalized.type is None or not _is_explicit_garment_type(normalized.type, query_text):
			continue

		key = (normalized.type, normalized.color, normalized.pattern)
		if key in seen:
			continue
		seen.add(key)
		garments.append(normalized)
	return garments


def build_prompt(normalized_query: str, previous_error: Optional[str] = None, previous_output: Optional[str] = None) -> str:
	"""Build the extraction prompt for the LLM."""
	system_prompt = (
		"You are a strict fashion query understanding engine. "
		"Return exactly one JSON object with this schema:\n"
		'{"scene": string or null, "style": string or null, "pose": string or null, '
		'"objects": [string, ...], "garments": [{"type": string or null, "color": string or null, '
		'"pattern": string or null}]}\n\n'
		"Rules:\n"
		"- Return JSON only. No markdown, no code fences, no commentary.\n"
		"- Never invent attributes.\n"
		"- garments must be empty unless the query explicitly names a physical garment such as shirt, blazer, coat, pants, skirt, shoes, or bag.\n"
		"- Use null when a value is unclear.\n"
		f"- scene must be one of: {', '.join(sorted(VALID_SCENES))}.\n"
		f"- style must be one of: {', '.join(sorted(VALID_STYLES))}.\n"
		"- Keep scene, style, and pose concise and lowercase.\n"
		"- Remove duplicate objects and garments.\n\n"
		"Example:\n"
		"Query: \"something nice for a job interview\"\n"
		"JSON: {\"scene\": null, \"style\": \"business formal\", \"pose\": null, \"objects\": [], \"garments\": []}"
	)

	user_lines = [f'Query: "{normalized_query}"', "Return JSON only."]
	if previous_error:
		user_lines = [
			"The previous answer was invalid.",
			f"Error: {previous_error}",
			*( [f"Previous output: {previous_output}"] if previous_output else [] ),
			*user_lines,
		]
	return system_prompt + "\n\n" + "\n".join(user_lines)


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
	"""Wrap a prompt in a chat template when the tokenizer supports it."""
	if hasattr(tokenizer, "apply_chat_template"):
		messages = [
			{"role": "system", "content": prompt.split("\n\n", 1)[0]},
			{"role": "user", "content": prompt.split("\n\n", 1)[-1]},
		]
		return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	return prompt


def generate_json(
	tokenizer: Any,
	model: Any,
	device: torch.device,
	normalized_query: str,
	max_new_tokens: int,
	num_beams: int,
	max_retries: int,
) -> Optional[str]:
	"""Generate a JSON-only response with retry-on-parse-failure behavior."""
	previous_error: Optional[str] = None
	previous_output: Optional[str] = None

	for _attempt in range(1, max_retries + 1):
		prompt = build_prompt(normalized_query, previous_error=previous_error, previous_output=previous_output)
		formatted_prompt = _apply_chat_template(tokenizer, prompt)
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
		response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

		try:
			parse_response(response_text)
			return response_text
		except Exception as exc:  # noqa: BLE001
			previous_error = str(exc)
			previous_output = response_text
			logging.warning("Invalid JSON response; retrying: %s", exc)

	return None


def _extract_json_blob(text: str) -> str:
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

	for index in range(start, len(cleaned)):
		character = cleaned[index]
		if in_string:
			if escape:
				escape = False
			elif character == "\\":
				escape = True
			elif character == '"':
				in_string = False
			continue

		if character == '"':
			in_string = True
		elif character == "{":
			depth += 1
		elif character == "}":
			depth -= 1
			if depth == 0:
				return cleaned[start : index + 1]

	raise ValueError("Unbalanced JSON object in response.")


def parse_response(response_text: str) -> QueryExtractionRecord:
	"""Parse and validate the raw LLM response."""
	logging.info("Validating JSON...")
	json_blob = _extract_json_blob(response_text)
	payload = json.loads(json_blob)
	if hasattr(QueryExtractionRecord, "model_validate"):
		return QueryExtractionRecord.model_validate(payload)
	return QueryExtractionRecord.parse_obj(payload)  # type: ignore[attr-defined]


def validate_query_document(raw_query: str, embedding_text: str, response_text: Optional[str]) -> QueryDocument:
	"""Convert model output into a validated QueryDocument or an empty fallback."""
	query_text = _normalize_text(raw_query)
	if response_text is None:
		return QueryDocument(
			raw_query=raw_query,
			scene=None,
			style=None,
			pose=None,
			objects=[],
			garments=[],
			embedding_text=embedding_text,
		)

	try:
		record = parse_response(response_text)
	except (ValidationError, ValueError, json.JSONDecodeError) as exc:
		logging.warning("Falling back to empty query document: %s", exc)
		return QueryDocument(
			raw_query=raw_query,
			scene=None,
			style=None,
			pose=None,
			objects=[],
			garments=[],
			embedding_text=embedding_text,
		)

	document = QueryDocument(
		raw_query=raw_query,
		scene=normalize_scene_label(record.scene),
		style=normalize_style_label(record.style),
		pose=normalize_optional_text(record.pose),
		objects=[],
		garments=normalize_garments(record.garments, query_text),
		embedding_text=embedding_text,
	)
	garment_types = [garment.type for garment in document.garments if garment.type]
	filtered_objects: List[str] = []
	seen_objects = set()
	for obj in normalize_string_list(record.objects):
		if not _is_object_allowed(obj, garment_types):
			continue
		if obj in seen_objects:
			continue
		seen_objects.add(obj)
		filtered_objects.append(obj)
	document.objects = filtered_objects
	return document


def parse_query(
	raw_query: str,
	model_name: str = DEFAULT_MODEL_NAME,
	device: str = "auto",
	max_retries: int = DEFAULT_MAX_RETRIES,
	max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
	num_beams: int = DEFAULT_NUM_BEAMS,
) -> QueryDocument:
	"""Parse a raw natural-language query into a validated QueryDocument."""
	normalized_query = normalize_query(raw_query)
	if not normalized_query:
		return QueryDocument(
			raw_query=raw_query,
			scene=None,
			style=None,
			pose=None,
			objects=[],
			garments=[],
			embedding_text=normalized_query,
		)

	tokenizer, model, resolved_device = load_model(model_name, detect_device(device))
	response_text = generate_json(
		tokenizer=tokenizer,
		model=model,
		device=resolved_device,
		normalized_query=normalized_query,
		max_new_tokens=max_new_tokens,
		num_beams=num_beams,
		max_retries=max_retries,
	)
	return validate_query_document(raw_query=raw_query, embedding_text=normalized_query, response_text=response_text)


def _read_query(args: argparse.Namespace) -> str:
	"""Read the query either from CLI or from stdin."""
	if args.query is not None:
		return args.query
	try:
		return input("Enter query: ")
	except EOFError:
		return ""


def main() -> None:
	"""Run the query parser and print the validated QueryDocument as JSON."""
	setup_logging()
	args = parse_args()

	logging.info("Loading model...")
	tokenizer, model, device = load_model(args.model_name, detect_device(args.device))

	raw_query = _read_query(args)
	normalized_query = normalize_query(raw_query)

	if not normalized_query:
		document = QueryDocument(
			raw_query=raw_query,
			scene=None,
			style=None,
			pose=None,
			objects=[],
			garments=[],
			embedding_text=normalized_query,
		)
		print(json.dumps(document.to_dict(), indent=2, ensure_ascii=False))
		logging.info("Done.")
		return

	logging.info("Parsing query...")
	response_text = generate_json(
		tokenizer=tokenizer,
		model=model,
		device=device,
		normalized_query=normalized_query,
		max_new_tokens=args.max_new_tokens,
		num_beams=args.num_beams,
		max_retries=args.max_retries,
	)

	document = validate_query_document(raw_query=raw_query, embedding_text=normalized_query, response_text=response_text)
	print(json.dumps(document.to_dict(), indent=2, ensure_ascii=False))
	logging.info("Done.")


if __name__ == "__main__":
	main()
