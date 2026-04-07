# /build-metadata

Build or rebuild the function metadata index for a repo.

## What this command does

Runs `rag/build_metadata.py` to:
1. Chunk all C/Python files
2. Extract function signatures, callee names, complexity tiers
3. Build call graph (caller ↔ callee backfill)
4. Optionally generate T1 Haiku one-line summaries for complex functions
5. Save to `{repo}/.kloc/metadata_index.json`

The index is automatically picked up by `RAGRetriever` and `MetadataRetriever`
on the next retrieval call — no other wiring needed.

## Usage

```bash
# Basic (signatures + call graph, no LLM)
python rag/build_metadata.py --repo doom/

# With T1 Haiku summaries for complex functions (~$0.002 per 100 functions)
python rag/build_metadata.py --repo doom/ --summaries

# Single file only
python rag/build_metadata.py --repo doom/ --file doom/src/p_enemy.c

# Dry run: print index without saving
python rag/build_metadata.py --repo doom/ --dry-run
```

Or via kloc.py (once wired):
```bash
python kloc.py build-metadata --repo doom/
python kloc.py build-metadata --repo doom/ --summaries
```

## When to rebuild

- After adding new source files to the repo
- After renaming functions (stale call-graph entries are harmless but waste slots)
- After enabling `--summaries` for the first time
- TER drops or retrieval quality degrades unexpectedly

## Output location

```
{repo}/.kloc/metadata_index.json
```

## Field weights used by retriever

| Field         | Weight | Description |
|---------------|--------|-------------|
| function name | ×5     | Exact/partial name match |
| summary       | ×3     | T1 Haiku one-liner (if generated) |
| callees       | ×2     | Function calls share domain vocabulary |
| signature     | ×2     | Parameter names carry intent |
| body text     | ×1     | Baseline BM25 length-normalised |

Large functions (>100 lines) get an additional `÷ log(line_count)` size penalty
to prevent vocabulary-mass dominance.
