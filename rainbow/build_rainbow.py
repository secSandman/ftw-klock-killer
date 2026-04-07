"""
build_rainbow.py — Pre-compute hash rainbow tables for stdlib patterns.
Run once: python rainbow/build_rainbow.py
Outputs: data/rainbow_stdlib_python.jsonl  data/rainbow_stdlib_c.jsonl
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PYTHON_PATTERNS = [
    "from __future__ import annotations",
    "import os", "import sys", "import re", "import json", "import time",
    "import logging", "import hashlib", "import math", "import ast",
    "from pathlib import Path", "from typing import Any, Dict, List, Optional",
    "from dataclasses import dataclass", "from collections import defaultdict",
    "import argparse", "import subprocess", "import shutil", "import tempfile",
    "import unittest", "import pytest",
    "logger = logging.getLogger(__name__)",
    "if __name__ == '__main__':",
    "def __init__(self):", "def __repr__(self):", "def __str__(self):",
]

C_PATTERNS = [
    "#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>",
    "#include <stdint.h>", "#include <stdbool.h>", "#include <math.h>",
    "#include <assert.h>", "#include <ctype.h>", "#include <errno.h>",
    "#include <limits.h>", "#include <time.h>", "#include <unistd.h>",
    "int main(int argc, char *argv[])", "int main(void)",
    "NULL", "EOF", "true", "false",
    "typedef struct {", "} ;",
    "malloc(sizeof(",  "free(",
    "printf(", "fprintf(stderr,",
    "memset(", "memcpy(", "strlen(",
]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(patterns: list, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for pattern in patterns:
            # "compressed" = just the pattern itself for now
            # In production: run through GRUG's synonym + caveman pass
            compressed = pattern.replace("__future__", "__fut__").replace("annotations", "ann")
            rec = {"hash": sha256(pattern), "original": pattern, "compressed": compressed}
            f.write(json.dumps(rec) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = Path(__file__).parent.parent / "data"
    py_n = build(PYTHON_PATTERNS, base / "rainbow_stdlib_python.jsonl")
    c_n  = build(C_PATTERNS,      base / "rainbow_stdlib_c.jsonl")
    print(f"Python rainbow: {py_n} entries -> data/rainbow_stdlib_python.jsonl")
    print(f"C rainbow:      {c_n} entries -> data/rainbow_stdlib_c.jsonl")
