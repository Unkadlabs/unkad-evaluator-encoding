#!/bin/bash
# M1-M3 pilot: 410M drop-mode pilot (4 branches), then penalize-mode
# comparison (A, B) sharing the SAME C0 checkpoint. Local Mac, $0.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== pilot 1/2: drop mode (4 branches) ==="
uv run python -m eae.run --config configs/pilot_410m.yaml

echo "=== copying shared C0 into penalize run dir ==="
mkdir -p runs/pilot_410m_pen
rm -rf runs/pilot_410m_pen/c0
cp -R runs/pilot_410m/c0 runs/pilot_410m_pen/c0
cp runs/pilot_410m/c0_verify.json runs/pilot_410m_pen/c0_verify.json

echo "=== pilot 2/2: penalize mode (A, B) ==="
uv run python -m eae.run --config configs/pilot_410m_pen.yaml --stage data
uv run python -m eae.run --config configs/pilot_410m_pen.yaml --stage phase2
uv run python -m eae.run --config configs/pilot_410m_pen.yaml --stage cache
uv run python -m eae.run --config configs/pilot_410m_pen.yaml --stage analyze

echo "=== PILOT COMPLETE ==="
