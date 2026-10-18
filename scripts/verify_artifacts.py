"""Verify included artifacts against the release manifest (no model execution)."""
import hashlib
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--artifacts',type=Path)
args=parser.parse_args()
artifact_root = args.artifacts or (root/'artifacts'/'validated' if (root/'artifacts'/'validated'/'report.json').exists() else root/'artifacts'/'portfolio')
manifest = json.loads((artifact_root / 'manifest.json').read_text(encoding='utf-8'))
for name, expected in manifest['files'].items():
    path = artifact_root / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Artifact mismatch: {name}')
print(f"Verified {len(manifest['files'])} artifact checksums.")
