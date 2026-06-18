#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# Benign-observed ranges for the two SDP origin fields
SESS_ID_MIN = 1_767_520_664
SESS_ID_MAX = 1_767_542_828
VERSION_MIN = 232
VERSION_MAX = 999_749

# The exact literal string produced by SIPTorch
TARGET_PATTERN = re.compile(r"o=mhandley 29739 7272939( IN IP4)")


def fix_record(rec: dict[str, Any]) -> bool:
    """Mutate rec['buffers'] in-place.  Returns True if a change was made."""
    buf = rec.get("buffers")
    if not isinstance(buf, str):
        return False
    if "29739 7272939" not in buf:
        return False

    # Derive a reproducible seed from stable per-record fields.
    seed = (rec.get("ip_id", 0) * 1_000_003) ^ (rec.get("flowstart_time", 0) * 999_983)
    rng = random.Random(seed)

    sess_id = rng.randint(SESS_ID_MIN, SESS_ID_MAX)
    version = rng.randint(VERSION_MIN, VERSION_MAX)

    new_buf = TARGET_PATTERN.sub(
        rf"o=mhandley {sess_id} {version}\1",
        buf,
    )
    if new_buf == buf:
        return False
    rec["buffers"] = new_buf
    return True


def process_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Load, fix, and optionally write a dataset file.

    Returns (total_records, fixed_records).
    """
    with path.open("r", encoding="utf-8") as f:
        raw: Any = json.load(f)

    is_wrapped = isinstance(raw, dict)
    records: list[dict] = raw.get("dataset", []) if is_wrapped else raw

    fixed = sum(fix_record(rec) for rec in records)

    if not dry_run and fixed:
        with path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
            f.write("\n")

    return len(records), fixed


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "files",
        nargs="*",
        default=glob.glob("sip-dataset/attack/*.json"),
        help="Attack JSON files to fix (default: sip-dataset/attack/*.json)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count fixes without writing files",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    total_recs = total_fixed = 0
    for p_str in sorted(args.files):
        path = Path(p_str)
        if not path.exists():
            print(f"  SKIP (not found): {path}", file=sys.stderr)
            continue
        if path.suffix.lower() != ".json" or path.name.endswith(".bak"):
            continue
        recs, fixed = process_file(path, dry_run=args.dry_run)
        prefix = "[DRY-RUN] " if args.dry_run else ""
        print(f"  {prefix}{path}: {fixed}/{recs} records fixed")
        total_recs += recs
        total_fixed += fixed

    print(f"\nTotal: {total_fixed}/{total_recs} records fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
