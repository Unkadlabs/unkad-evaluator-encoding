"""Local Phase 3 analysis over downloaded activation caches.

    uv run python scripts/analyze_caches.py --root runs/modal/pilot_410m \
        --pairs A_s0:B_s0 A_s0:c0 B_s0:c0

Works on any directory containing hiddens/<name>/ caches (Mac or Modal runs).
Prints the summary table and writes <root>/analysis/report.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eae.analysis import compare_branches, summarize  # noqa: E402
from eae.caching import load_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="colon-separated cache-name pairs, e.g. A_s0:B_s0")
    args = ap.parse_args()

    root = Path(args.root)
    report = {}
    for pair in args.pairs:
        a, b = pair.split(":")
        ca = load_cache(root / "hiddens" / a)
        cb = load_cache(root / "hiddens" / b)
        cmp = compare_branches(ca, cb)
        report[f"{a}_vs_{b}"] = {"summary": summarize(cmp), **cmp}
        print(f"{a} vs {b}: "
              f"{ {k: round(v, 3) for k, v in report[f'{a}_vs_{b}']['summary'].items()} }")

    out = root / "analysis" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
