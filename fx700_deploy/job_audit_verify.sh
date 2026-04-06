#!/bin/bash
#SBATCH --job-name=qsc_audit_verify
#SBATCH -N 1
#SBATCH -p Batch
#SBATCH --time=01:00:00
#SBATCH --output=results/audit_verify-%j.log
#SBATCH --error=results/audit_verify-%j.err

# ══════════════════════════════════════════════════════════════
#  QSC2025 — Red Team Audit Verification on FX700
#  Runs: tests (28/28), ILP baseline, benchmark with advantage
# ══════════════════════════════════════════════════════════════

# ── Environment setup ────────────────────────────────────────
export LD_LIBRARY_PATH=/home/share/developer/boost1.90.0/lib:/usr/local/lib
export BOOST_ROOT=/home/share/developer/boost-1.90.0
export PYTHONIOENCODING=utf-8
source ~/QARPdemo/venv/bin/activate
cd ~/QARPdemo/qsc2025

mkdir -p results

echo "═══════════════════════════════════════════════════"
echo "  QSC2025 Red Team Audit Verification"
echo "  Host: $(hostname)"
echo "  Start: $(date)"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Unit Tests (28 tests including 4 new audit tests)
echo ""
echo "▶ Step 1: Unit Tests (should be 28/28)"
python -u tests/tests.py
TEST_EXIT=$?
echo "  Exit code: $TEST_EXIT"

# ── Step 2: Audit Fixes Fair Benchmark (ILP + SA + RQAOA)
echo ""
echo "▶ Step 2: Audit Fair Benchmark (6q advantage problem)"
python -u audit_fixes.py -i data/request_advantage.json
AUDIT_EXIT=$?
echo "  Exit code: $AUDIT_EXIT"

# ── Step 3: Full Benchmark with ILP comparison
echo ""
echo "▶ Step 3: Benchmark Suite with ILP baseline"
python -u benchmark_suite.py \
    --input data/request_advantage.json \
    --algorithms exact,rqaoa \
    --layers 3
BENCH_EXIT=$?
echo "  Exit code: $BENCH_EXIT"

# ── Step 4: 12-qubit benchmark (verify fixed data)
echo ""
echo "▶ Step 4: 12-qubit Benchmark (fixed RET-LA demand)"
python -u benchmark_suite.py \
    --input data/request_12q.json \
    --algorithms rqaoa \
    --layers 2
BENCH12_EXIT=$?
echo "  Exit code: $BENCH12_EXIT"

# ── Summary
echo ""
echo "═══════════════════════════════════════════════════"
echo "  RESULTS SUMMARY"
echo "  Tests:      $([ $TEST_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Audit:      $([ $AUDIT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Benchmark:  $([ $BENCH_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  12q Bench:  $([ $BENCH12_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Completed: $(date)"
echo "═══════════════════════════════════════════════════"
