#!/bin/bash
# ============================================================
#  QSC2025 Full Benchmark Suite
# ============================================================
set -e
source ~/QARPdemo/venv/bin/activate
cd ~/QARPdemo/qsc2025
mkdir -p results

echo "=== QSC2025 Full Benchmark ==="
echo "Starting: $(date)"
echo ""

# Run benchmark suite (all algorithms, all problem sizes)
python -u benchmark_suite.py \
    --input request.json request_12q.json request_advantage.json \
    --algorithms all \
    --layers 6 2>&1 | tee results/benchmark_output.txt

echo ""
echo "Benchmark complete: $(date)"

# Also run the advantage demo
echo ""
echo "=== Quantum Advantage Demonstration ==="
python -u run_optimization.py \
    --input request_advantage.json \
    --output results/advantage_result.json \
    --shots 10000 \
    --max_iter 500

echo ""
echo "All results saved in results/ directory"
ls -la results/
