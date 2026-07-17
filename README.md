# Fashion-Faux-Pas

Fashion-aware, composition-correct multimodal image retrieval.

> Given a natural language query like *"a red tie and a white shirt in a formal setting"*, return images where the colors are bound to the **correct** garments — not just images where those words appear somewhere.

Vanilla CLIP retrieval gets the vocabulary right and the binding wrong. This repo is the fix.

---

## Why this exists

Global image/text embeddings (CLIP, FashionCLIP-alone, etc.) collapse a scene into a single vector. That's fine for "dog on a beach," and it's *not* fine for fashion, because fashion queries are compositional: `garment → attribute` pairs matter, and a single vector has no notion of which color modifies which object. Two images — *white shirt / red tie* and *red shirt / white tie* — can sit almost on top of each other in CLIP space. A retrieval system judged on compositional queries will fail silently and confidently.

At the same time, some queries carry **no explicit attributes at all** (*"casual weekend outfit for a city walk"*) and can only be solved by a model that generalizes zero-shot from a style concept to the garments it implies. That rules out a purely structured/attribute-only system too.

Fashion-Faux-Pas is built around **not picking one** — it combines a domain-tuned embedding (for zero-shot generalization) with an LLM-parsed structured attribute schema (for binding correctness), and reranks one against the other.

Full design rationale, alternatives considered, and evaluation methodology are in the design document submitted alongside this repo (not yet checked in here — add it under e.g. `docs/design_document.pdf` if you want it version-controlled alongside the code).

---

## Architecture

Two independent pipelines, one shared schema — an offline indexer and an online retriever.

```
                    OFFLINE — INDEXER (batch, once per image)
                    ─────────────────────────────────────────
                    Raw image
                         │
                         ▼
                    Florence-2  ──▶  dense caption
                         │
                         ▼
                    LLM attribute parser (strict JSON schema)
                    {scene, style, pose, objects[], garments:[{type, color}]}
                         │
                         ▼
                    FashionCLIP image encoder  ──▶  768-d vector
                         │
                         ▼
                    Qdrant upsert: vector + attribute payload (one point)


                    ONLINE — RETRIEVER (per query, low latency)
                    ─────────────────────────────────────────
                    User query (natural language)
                         │
                         ▼
                    LLM query parser (same JSON schema)
                         │
                         ├──▶ FashionCLIP text encoder ──▶ query vector
                         │
                         ▼
                    Stage 1 — Qdrant ANN (HNSW) recall  ──▶  top-500 candidates
                         │
                         ▼
                    Stage 2 — hybrid rerank
                    score = α · cosine_similarity + β · attribute_overlap
                         │
                         ▼
                    Top-k images
```

Why this shape, briefly (see the design doc for the full comparison table of alternatives — vanilla CLIP, FashionCLIP-alone, caption-then-embed, and grounded detection):

| Component | Choice | Because |
|---|---|---|
| Captioner | Florence-2 | Denser, more literal captions on multi-garment scenes than BLIP-2; better input for the JSON parser |
| Attribute extraction | LLM, constrained to a fixed JSON schema, validated before indexing | Manufactures `garment → color` binding without training or running a detector; schema validation stops corrupt payloads reaching the index |
| Embedding backbone | FashionCLIP | Same interface as CLIP, better fashion vocabulary, zero added architectural cost |
| Vector store | Qdrant | Vector + structured payload on one point (no join, no second DB); native HNSW; native payload filtering, which the geo/weather extension needs |
| Ranking | Two-stage: cheap ANN recall → bounded hybrid rerank | Keeps the expensive, informative signal (attribute overlap) off the full corpus — this is what keeps the design scalable to 1M images without touching the ranking logic |

Grounded detection (GroundingDINO / OWL-ViT) is the structurally "most correct" fix for compositionality but is disproportionate engineering overhead for a 500–1k image, single-object-class problem. It's flagged as a Phase 2 upgrade — see [Future Work](#future-work).

---

## Repository structure

```
Fashion-Faux-Pas/
├── data/
│   ├── raw/                        # source images
│   ├── processed/
│   │   ├── captions.json           # Florence-2 output, keyed by filename
│   │   ├── attributes.json         # LLM-parsed {scene, style, pose, garments[]} per image
│   │   ├── documents.json          # caption + attributes + filename merged into one indexable record
│   │   ├── embeddings.npy          # FashionCLIP image vectors
│   │   ├── embedding_metadata.json # embedding <-> filename/index mapping
│   │   └── filenames.json          # ordering reference for embeddings.npy
│   ├── qdrant_db/                  # live Qdrant local storage (collection: fashion_images)
│   ├── qdrant_db_sanity/           # scratch collection for sanity checks
│   └── qdrant_db_test_probe/       # scratch collection for probing/debugging
├── indexing/
│   ├── caption_images.py           # Florence-2 → data/processed/captions.json
│   ├── parse_caption.py            # LLM, schema-constrained → data/processed/attributes.json
│   ├── build_document.py           # merges caption + attributes + filename → documents.json
│   ├── extract_embeddings.py       # FashionCLIP, batch → embeddings.npy + embedding_metadata.json
│   ├── extract_embedding.py        # single-image variant (debugging/ad hoc use)
│   └── index_qdrant.py             # upserts vectors + document payload into Qdrant
├── retrieval/                       # query parsing, ANN recall, hybrid rerank — in progress
├── scripts/
│   └── rename_images.py            # normalizes raw filenames (e.g. 000001.jpg) before indexing
├── configs/                         # reserved for model/α-β/Qdrant connection config — not yet populated
└── requirements.txt
```

Indexing is fully wired end to end: `caption_images.py → parse_caption.py → build_document.py → extract_embeddings.py → index_qdrant.py`, each stage reading the previous stage's output from `data/processed/`. `retrieval/` and `configs/` are scaffolded directories — the query-side pipeline (query parsing, ANN recall, hybrid α/β rerank) is the active work-in-progress; commands below reflect the current indexing pipeline and the intended retrieval interface.

The three `qdrant_db*` directories aren't redundant: `qdrant_db` is the real collection built from `data/processed/`, while `qdrant_db_sanity` and `qdrant_db_test_probe` are disposable scratch collections for validating the indexing/search logic without touching the real index. Worth `.gitignore`-ing all three (or at least the two scratch ones) rather than committing local Qdrant storage.

---

## Setup

```bash
git clone https://github.com/<you>/fashion-faux-pas.git
cd fashion-faux-pas
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You'll need:
- Qdrant running with local persistence (this repo uses `data/qdrant_db` as on-disk storage, not a separate server process — see `index_qdrant.py` for the client init)
- API access for the LLM used in caption parsing
- ~500–1,000 fashion images in `data/raw/`, with variation across environment (office / street / park / home), clothing type (formal / casual / outerwear), and color

If you don't have a dataset on hand, sample from [Fashionpedia](https://fashionpedia.github.io/home/) or pull permissively-licensed lifestyle photography from the Pexels/Unsplash APIs.

---

## Usage

### 1. Prepare raw images

```bash
python scripts/rename_images.py --input data/raw
```

Normalizes filenames (e.g. `000001.jpg`, `000002.jpg`, ...) so every downstream stage can join on filename cleanly.

### 2. Index the dataset (offline, run once)

```bash
python indexing/caption_images.py    --images data/raw --out data/processed/captions.json
python indexing/parse_caption.py     --captions data/processed/captions.json --out data/processed/attributes.json
python indexing/build_document.py    --captions data/processed/captions.json --attributes data/processed/attributes.json --out data/processed/documents.json
python indexing/extract_embeddings.py --images data/raw --out data/processed/embeddings.npy
python indexing/index_qdrant.py      --documents data/processed/documents.json --embeddings data/processed/embeddings.npy --collection fashion_images
```

`extract_embedding.py` (singular) is the single-image variant used for debugging one file at a time — not part of the batch run.

Use `qdrant_db_sanity` or `qdrant_db_test_probe` as the `--collection`/storage target when validating changes to the pipeline, so the real `qdrant_db` collection isn't disturbed.

### 3. Query *(retrieval pipeline — in progress)*

The query-side pipeline is being built out in `retrieval/`. The intended interface:

```bash
python retrieval/retrieve.py --query "a red tie and a white shirt in a formal setting" --top_k 5
```

This will run: query parsing (same schema as indexing) → FashionCLIP text embedding → Qdrant ANN recall (top-500) → hybrid rerank (`score = α·cosine_similarity + β·attribute_overlap`) → top-k.

### 4. Evaluate against the baseline

Planned: a `queries.json` covering the five evaluation categories (attribute-specific, contextual/place, complex semantic, style inference, compositional) and an `evaluate.py` reporting Precision@5, Recall@5, and MRR for **FashionCLIP-only** retrieval vs. the **hybrid reranked** pipeline — the comparison that actually answers "did this beat vanilla CLIP," not just "does it return images."

---

## How it behaves on the five evaluation query types

| Query type | What resolves it | What breaks without it |
|---|---|---|
| Attribute-specific ("bright yellow raincoat") | Attribute overlap on `garment.type=raincoat, garment.color=yellow` | CLIP conflates "yellow" with any yellow object in frame |
| Contextual/place ("business attire in a modern office") | `scene=office` + `style=business`, reinforced by embedding similarity | Scene tokens can leak from background objects rather than the person's attire |
| Complex semantic ("blue shirt on a park bench") | Joint match across garment + scene + pose — no single field suffices | Embedding-only retrieval can return "blue jeans in a park" or "shirt on an office bench" |
| Style inference ("casual weekend outfit for a city walk") | FashionCLIP zero-shot embedding — nothing to parse | The one case where leaning on structured attributes actively hurts |
| Compositional ("red tie, white shirt, formal") | Attribute overlap with correct garment→color binding | The documented CLIP failure mode this whole reranker exists to fix |

---

## Scalability

The design is built so scale only touches Stage 1 (ANN recall) — never the rerank stage or the indexing pipeline, both of which are already per-item and embarrassingly parallel.

- **Indexing**: captioning, parsing, and embedding are independent per image and batch/parallelize trivially — 1M images is a compute-cost question, not an architecture question.
- **ANN recall**: Qdrant's HNSW index has near-logarithmic query cost; a tuned index (`m`, `ef_construct`, `ef_search`) keeps top-500 recall in low tens of milliseconds at 1M × 768-d. Quantization (scalar/PQ) is available if memory becomes a constraint.
- **Rerank stays cheap by construction**: it only ever scores the top-500 candidates from Stage 1, regardless of corpus size.
- **Horizontal scaling**: Qdrant supports native sharding/replication if a single node's memory or QPS budget is exceeded — an infra change, not a retrieval-logic change.

## Zero-shot capability

Two complementary mechanisms cover two different failure modes:
- **Unseen style/vibe** ("beach-ready look") has no schema field to parse into — retrieval falls back almost entirely on FashionCLIP's pretrained embedding space.
- **Unseen but explicit attribute combinations** ("mustard corduroy jacket") are handled by the LLM parser doing open-vocabulary extraction — there's no closed label space to fall outside of.

The α/β blend isn't only a precision knob — when `attribute_overlap` is near zero because nothing parsed cleanly, the embedding term carries the ranking on its own, so the system degrades gracefully rather than failing on out-of-schema queries.

---

## Future work

**Locations and weather** (additive, not a redesign):
- Add two payload fields at indexing time — `geo_context` (visual cues: architecture, signage, foliage, or EXIF/GPS where available) and `weather_context` (wet pavement, coats, snow, overcast light).
- Extend the same LLM parsing prompt with both fields; Florence-2 captions already tend to surface this detail, so the parser can lift it directly.
- Fold both fields into the existing `attribute_overlap` term — the rerank formula doesn't change shape, only its input schema.
- Qdrant's native payload filtering can pre-filter by geo/weather before the ANN stage when a query is unambiguous ("in Tokyo"), improving precision and shrinking what the reranker has to score.

**Precision**:
- Grid-search α/β against a held-out, human-labeled query→relevant-image set (Precision@5, Recall@5, MRR — not estimated).
- Audit the attribute parser directly: since the whole compositionality fix rests on caption→JSON accuracy, sampling parser output for missed garments or mis-bound colors is likely higher-leverage than swapping any single model.
- Use parser/embedding disagreement as a trigger to route the disputed subset through grounded detection (Approach D) as a Phase 2 attribute source — keeping the expensive detector off the common case.

---

## License

MIT (or update to match your submission requirements).