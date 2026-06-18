#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

CLIENT_IP = "172.18.0.4"
SERVER_IP = "172.18.0.2"

# Match the SDP connection field in the body (after the SIP header blank line).
# The pattern is always on its own line.
SDP_C_LINE = re.compile(r"(c=IN IP4 )172\.18\.0\.2")

# Match the From header to check whether this is client-originated.
FROM_CLIENT_RE = re.compile(
    r"^From:.*sip:[^@]+@" + re.escape(CLIENT_IP),
    re.IGNORECASE | re.MULTILINE,
)


def fix_record(rec: dict[str, Any]) -> bool:
    """Mutate rec['buffers'] in-place.  Returns True if a change was made."""
    buf = rec.get("buffers")
    if not isinstance(buf, str):
        return False
    if SERVER_IP not in buf:
        return False
    if "c=IN IP4 " + SERVER_IP not in buf:
        return False
    # Only fix client-originated records.
    if not FROM_CLIENT_RE.search(buf):
        return False

    new_buf = SDP_C_LINE.sub(r"\g<1>" + CLIENT_IP, buf)
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
