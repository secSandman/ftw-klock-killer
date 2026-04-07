# /optimize-retrieval

Analyze the current RAG retrieval quality for a given query and suggest improvements.

## What this command does

1. Loads the metadata index for the target repo (if present)
2. Runs both plain BM25 and metadata-augmented retrieval on the query
3. Shows side-by-side comparison: which chunks each strategy picks, scores, why
4. Identifies if the query hits a known failure mode (large-function dominance, vocabulary mismatch, missing symbol routing)
5. Suggests concrete fixes: build/rebuild metadata index, add summaries, tune field weights, or switch to embedding retrieval

## Usage

```
/optimize-retrieval --repo <path> --question "<your question>" [--top-k 5]
```

## Example

```
/optimize-retrieval --repo doom/ --question "how does sound propagation work" --top-k 5
```

## Failure modes diagnosed

| Mode | Symptom | Fix |
|------|---------|-----|
| Large-function dominance | P_LookForPlayers (1600 lines) wins every query | Build metadata index (size penalty kicks in) |
| Vocabulary mismatch | Zero-scoring chunks for domain-specific terms | Add T1 summaries (`--summaries` flag) |
| Missing symbol routing | "A_Chase" query doesn't find A_Chase directly | Metadata index enables exact name boost ×5 |
| No metadata index | Falls back to plain BM25 for all queries | `python rag/build_metadata.py --repo <path>` |

## Implementation notes

This command reads:
- `rag/retriever.py` — RAGRetriever with metadata support
- `rag/metadata_index.py` — MetadataRetriever and field weights
- `.kloc/metadata_index.json` — pre-built index (if present)

Run `python rag/build_metadata.py --repo <path>` first to build the index.
