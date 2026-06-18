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
SERVER_SIP_PORT = "5060"


def _via_src_port(buf: str) -> str | None:
    """Return the source port from the first Via sent-by field, or None."""
    m = re.search(
        r"Via:\s*SIP/2\.0/\w+\s+" + re.escape(CLIENT_IP) + r":(\d+)",
        buf,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def fix_record(buf: str) -> tuple[str, bool]:
    """Return (fixed_buf, changed) for one SIP buffer string."""
    if not isinstance(buf, str):
        return buf, False

    lines = buf.split("\r\n")

    # Check that From contains the client IP — only then is this a mismatch.
    from_has_client = any(
        line.lower().startswith("from:") and CLIENT_IP in line
        for line in lines
    )
    if not from_has_client:
        return buf, False

    # Find the source port from Via to use in the corrected Contact.
    src_port = _via_src_port(buf)

    changed = False
    new_lines: list[str] = []
    for line in lines:
        if line.lower().startswith("contact:") and SERVER_IP in line:
            new_line = line.replace(SERVER_IP, CLIENT_IP)
            # Also update the port: :5060 → :<src_port> inside the URI.
            # Guard: only do this when src_port is known and the Contact value
            # currently references the old server port 5060 immediately after
            # the IP we just replaced.
            if src_port and f"{CLIENT_IP}:{SERVER_SIP_PORT}" in new_line:
                new_line = new_line.replace(
                    f"{CLIENT_IP}:{SERVER_SIP_PORT}",
                    f"{CLIENT_IP}:{src_port}",
                )
            if new_line != line:
                changed = True
            line = new_line
        new_lines.append(line)

    return "\r\n".join(new_lines), changed


def process_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Load, fix, and optionally write a dataset file.

    Returns (total_records, fixed_records).
    """
    with path.open("r", encoding="utf-8") as f:
        raw: Any = json.load(f)

    is_wrapped = isinstance(raw, dict)
    records: list[dict] = raw.get("dataset", []) if is_wrapped else raw

    fixed = 0
    for rec in records:
        buf = rec.get("buffers")
        new_buf, changed = fix_record(buf)
        if changed:
            rec["buffers"] = new_buf
            fixed += 1

    if not dry_run:
        out = raw  # mutated in-place above
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
            f.write("\n")

    return len(records), fixed


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        if path.suffix.lower() not in (".json",) or path.name.endswith(".bak"):
            continue
        recs, fixed = process_file(path, dry_run=args.dry_run)
        status = "[DRY-RUN] " if args.dry_run else ""
        print(f"  {status}{path}: {fixed}/{recs} records fixed")
        total_recs += recs
        total_fixed += fixed

    print(f"\nTotal: {total_fixed}/{total_recs} records fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
