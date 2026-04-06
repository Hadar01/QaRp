#!/bin/bash
#SBATCH --job-name=qsc_request
#SBATCH -N 1
#SBATCH -p Batch
#SBATCH --time=00:30:00
#SBATCH --output=results/qsc_request-%j.log
#SBATCH --error=results/qsc_request-%j.err

# ── Environment setup ────────────────────────────────────────
export LD_LIBRARY_PATH=/home/share/developer/boost1.90.0/lib:/usr/local/lib
export BOOST_ROOT=/home/share/developer/boost-1.90.0
source ~/QARPdemo/venv/bin/activate
cd ~/QARPdemo/qsc2025

mkdir -p results

echo "Starting QSC2025 optimization: request.json"
echo "Backend: qulacs, Nodes: 1"
echo "Start time: $(date)"

# ── Run optimization ─────────────────────────────────────────
python -u run_optimization.py \
    --input request.json \
    --backend qulacs \
    --output results/request_result.json \
    --shots 10000 \
    --max_iter 500

echo "Completed: $(date)"
echo "Exit code: $?"
