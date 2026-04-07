# /analyze-chunks

Inspect the chunk distribution for a C/Python file or repo and identify RAG quality issues.

## What this command does

1. Chunks the target file/repo via `Chunker`
2. Reports size distribution (line counts, token counts per chunk)
3. Flags problematic chunks:
   - Giant chunks (>200 lines) — will dominate BM25 without metadata penalty
   - Tiny chunks (<5 lines) — too little context to retrieve meaningfully
   - Unnamed chunks (`chunk_type=module`) — fallback chunks, no name boost possible
4. Shows per-chunk type breakdown (function / struct / class / module)
5. Reports call-graph density: avg callees per function (higher = richer metadata)

## Usage

```
/analyze-chunks --file <path/to/file.c>
/analyze-chunks --repo <repo_path>
```

## Output example

```
  p_enemy.c — 107 chunks
  ┌─────────────────┬──────┬──────────┬──────────┐
  │ type            │ count│ avg lines│ max lines│
  ├─────────────────┼──────┼──────────┼──────────┤
  │ function        │   97 │    16.4  │  1,623   │
  │ struct          │    3 │     8.1  │     14   │
  │ module          │    7 │     4.2  │     12   │
  └─────────────────┴──────┴──────────┴──────────┘
  ⚠  3 giant chunks (>200 lines) — will skew BM25 without metadata index
  ✓  Metadata index present: 97 functions indexed, 12 with summaries
```

## When to run this

- After adding a new C/Python file to the pipeline
- When retrieval results seem irrelevant
- Before building a metadata index (to understand expected index size)

## Implementation notes

Reads chunks from `rag/chunker.py` → language parsers.
Metadata status from `.kloc/metadata_index.json`.
No external dependencies.
