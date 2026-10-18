"""Deterministic parsing, train-only vocabulary, and disjoint time windows."""
from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

SOURCE = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
PARSER_VERSION = "bgl-content-mask-v1"


def read_bgl(path: Path, limit: int = 300_000) -> list[dict]:
    """Stream a zip member without extracting archive-controlled paths."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith("bgl.log")]
            if len(names) != 1:
                raise ValueError("Expected exactly one BGL.log in the archive.")
            with archive.open(names[0]) as binary:
                return read_lines(io.TextIOWrapper(binary, encoding="utf-8", errors="replace"), limit)
    with path.open(encoding="utf-8", errors="replace") as handle:
        return read_lines(handle, limit)


def read_lines(lines, limit: int) -> list[dict]:
    records = []
    for index, raw in enumerate(lines):
        if index >= limit:
            break
        parts = raw.rstrip().split(maxsplit=9)
        if len(parts) < 9 or not parts[1].isdigit():
            raise ValueError(f"Malformed BGL record at line {index + 1}.")
        # Remove the alert-category label AND all metadata from model input.
        # Severity is deliberately excluded too: the research question is event content/order.
        content = parts[9] if len(parts) == 10 else "<empty>"
        records.append({"line": index + 1, "time": parts[4], "label": int(parts[0] != "-"),
                        "content": content, "template": normalize(content)})
    return records


def normalize(content: str) -> str:
    content = content.lower().strip()
    content = re.sub(r"0x[0-9a-f]+", "<hex>", content)
    content = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", content)
    content = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>", content)
    content = re.sub(r"\b[a-z]*\d+[a-z0-9_.:/-]*\b", "<num>", content)
    content = re.sub(r"\s+", " ", content)
    return content


def fit_vocab(records: list[dict], min_count: int = 2) -> dict[str, int]:
    counts = Counter(r["template"] for r in records)
    return {"<UNK>": 0, **{key: i + 1 for i, key in enumerate(sorted(k for k, n in counts.items() if n >= min_count))}}


def encode(records: list[dict], vocab: dict[str, int]) -> np.ndarray:
    return np.array([vocab.get(r["template"], 0) for r in records], dtype=np.int64)


def windows(records: list[dict], vocab: dict[str, int], size: int) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    count = len(records) // size
    if count == 0:
        raise ValueError(f"At least {size} records are required per split.")
    groups = [records[i * size:(i + 1) * size] for i in range(count)]
    x = encode(records[:count * size], vocab).reshape(count, size)
    y = np.array([int(any(r["label"] for r in group)) for group in groups])
    return x, y, groups


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    # Split raw records FIRST. Windowing happens separately inside each split.
    n = len(records)
    return {"train": records[:int(n * .6)], "validation": records[int(n * .6):int(n * .8)],
            "test": records[int(n * .8):]}


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".part")
    urllib.request.urlretrieve(SOURCE, pending)
    if not zipfile.is_zipfile(pending):
        pending.unlink(missing_ok=True)
        raise ValueError("Download was not a valid zip archive.")
    pending.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
