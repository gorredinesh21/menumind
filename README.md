# MenuMind — Semantic Menu Search + Grounded RAG, in pure Go

**Live:** https://menumind-yzzxrxetcq-uc.a.run.app

MenuMind searches ~800 Bangalore dishes **the way people think**: "something warm and
comforting", "high protein under ₹300", "late night spicy indulgence". It is a
dependency-free Go implementation of a modern retrieval stack:

- **BM25** (hand-written, classic Robertson k1/b) for lexical relevance,
- **vector similarity** over bge-small-en-v1.5 embeddings (384-d, cosine),
- fused with **Reciprocal Rank Fusion**, then filtered (veg / price / rating / cuisine),
- and an **Ask** mode that streams an LLM answer grounded *strictly* in the retrieved
  dishes, with `[D1]`-style citations — never inventing menu items.

Every result card shows its **score breakdown** (BM25 rank, semantic rank, cosine) —
retrieval you can audit, not a black box.

```
"something warm and comforting"
  ├─ BM25 channel ──────► top-50 ranked docs          ┐
  ├─ bge-small vector ──► cosine top-50 (0.15 floor)  ┘→ RRF fuse → filters → hits
  └─ Ask mode: top-8 context → LLM answers ONLY from context, streamed with [D#] cites
```

## Why Go for GenAI plumbing

The expensive parts here are I/O-shaped: embedding 800 dishes is 800 network calls —
MenuMind warms them with a **bounded concurrent worker pool** (`errgroup.SetLimit`) in
~26 s, caches to a committed `data/embeddings.json`, and boots instantly with vectors
in memory. Query-time embedding is an on-demand call with a bounded LRU. No Python, no
GIL, one static binary.

## Architecture

```
main.go               boot: index, committed vector cache, graceful shutdown
internal/corpus/      synthetic Bangalore corpus — 36 restaurants, ~800 tagged dishes
internal/search/      BM25 + cosine + RRF fusion + filters (pure Go, no deps)
internal/embed/       bge client (1D/2D mean-pool decode), query LRU, disk cache
internal/ask/         grounded streaming generation (OpenAI-compatible SSE parsing)
internal/server/      HTTP + SSE; /stats, /healthz
cmd/warm/             one-shot concurrent corpus embedding → data/embeddings.json
web/                  vanilla-JS UI (embed.FS)
```

**Graceful degradation is a feature:** no API key → BM25-only search still works and
Ask returns a deterministic retrieval summary; embedding endpoint down mid-query →
that query falls back to lexical with a logged warning; LLM stream fails → the UI shows
retrieved dishes plus an explicit error line. The product never white-screens.

## Run

```bash
export HF_TOKEN=hf_...        # optional — enables semantic + Ask
go run .                      # http://localhost:8080

go test ./...                 # BM25, cosine, RRF, filters, decoders, cache
make docker                   # distroless image
```

Re-warm embeddings after touching the corpus: `go run ./cmd/warm -workers 10`.

## API

| Route | What |
|---|---|
| `GET /api/search?q=&veg=1&max_price=&min_rating=&cuisine=&k=` | hybrid retrieval, instant |
| `POST /api/ask` | `{question, veg_only, max_price, …}` → SSE: `sources`, `delta`*n, `done` |
| `GET /stats` | corpus + vector counts |
| `GET /healthz` | liveness + channel status |

## Interview talking points

- **Why RRF over score summation?** BM25 (unbounded) and cosine ([-1,1]) are
  incomparable scales; RRF uses only ranks, so no calibration needed — the standard
  trick in production hybrid search.
- **Where would this break at scale?** Corpus → Postgres/OpenSearch; vectors →
  pgvector or an ANN index (my brute-force cosine is O(n·d), fine at 800, wrong at 1M);
  query embeddings → local model or batched service.
- **How do you stop hallucinated menu items?** The system prompt constrains to the
  numbered context; retrieval happens server-side first; the UI renders the exact
  sources next to the answer so claims are auditable.

## Stack

Go 1.27 · net/http + embed.FS · golang.org/x/sync (bounded pools) ·
bge-small-en-v1.5 + Llama-3.1-8B-Instruct via OpenAI-compatible endpoints ·
vanilla JS · Google Cloud Run.

---
Built by [Dinesh Gorre](https://github.com/gorredinesh21) · see also
[OrderPilot](https://github.com/gorredinesh21/orderpilot) — Go + Agentic AI food-ordering copilot.
