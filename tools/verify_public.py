#!/usr/bin/env python3
"""Verify every delivered public file against the snapshot catalogue."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
entries=json.loads((root/'SHA256.json').read_text())
for name,expected in entries.items():
 p=root/name
 if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
  raise SystemExit('Missing or altered file: '+name)
print(f'PASS: {len(entries)} public files verified')
