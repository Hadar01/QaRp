#!/bin/bash
#SBATCH --job-name=qsc_request_12q
#SBATCH -N 1
#SBATCH -p Batch
#SBATCH --time=01:00:00
#SBATCH --output=results/qsc_request_12q-%j.log
#SBATCH --error=results/qsc_request_12q-%j.err

# ── Environment setup ────────────────────────────────────────
export LD_LIBRARY_PATH=/home/share/developer/boost1.90.0/lib:/usr/local/lib
export BOOST_ROOT=/home/share/developer/boost-1.90.0
source ~/QARPdemo/venv/bin/activate
cd ~/QARPdemo/qsc2025

mkdir -p results

echo "Starting QSC2025 optimization: request_12q.json"
echo "Backend: qulacs_mpi, Nodes: 1"
echo "Start time: $(date)"

# ── Run optimization ─────────────────────────────────────────
mpirun -N 1 -npernode 1 -n 1 python -u run_optimization.py \
    --input request_12q.json \
    --backend qulacs_mpi \
    --output results/request_12q_result.json \
    --shots 10000 \
    --max_iter 500

echo "Completed: $(date)"
echo "Exit code: $?"
