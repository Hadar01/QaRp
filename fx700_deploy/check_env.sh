#!/bin/bash
# ============================================================
#  FX700 Environment Verification for QSC2025
# ============================================================
set -e
echo "=== QSC2025 FX700 Environment Check ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Check Python
echo "--- Python ---"
python3 --version 2>/dev/null || { echo "ERROR: python3 not found"; exit 1; }

# Check QARP
echo ""
echo "--- QARP SDK ---"
python3 -c "import qarp; print(f'QARP version: {qarp.__version__}')" 2>/dev/null \
    || echo "WARNING: qarp not importable. Check: source ~/QARPdemo/venv/bin/activate"

# Check Qulacs
echo ""
echo "--- Qulacs ---"
python3 -c "import qulacs; print(f'Qulacs: OK (version check N/A)')" 2>/dev/null \
    || echo "ERROR: qulacs not installed"

# Check MPI
echo ""
echo "--- MPI ---"
which mpirun 2>/dev/null && echo "mpirun: $(mpirun --version 2>&1 | head -1)" \
    || echo "WARNING: mpirun not in PATH (MPI backend unavailable)"

# Check Boost
echo ""
echo "--- Boost ---"
if [ -d "/home/share/developer/boost1.90.0" ]; then
    echo "Boost 1.90.0: Found"
else
    echo "WARNING: Boost not at expected path"
fi

# Check SLURM
echo ""
echo "--- SLURM ---"
which sbatch 2>/dev/null && echo "SLURM: $(sbatch --version)" \
    || echo "WARNING: sbatch not found (cannot submit batch jobs)"

# Check Python dependencies
echo ""
echo "--- Python Dependencies ---"
python3 -c "
deps = ['numpy', 'scipy', 'qulacs', 'fastapi', 'pydantic']
for d in deps:
    try:
        __import__(d)
        print(f'  {d}: OK')
    except ImportError:
        print(f'  {d}: MISSING')
"

# Run quick QARP test
echo ""
echo "--- Quick QARP Integration Test ---"
python3 -c "
import sys
sys.path.insert(0, '.')
from qarp_backend import detect_environment
env = detect_environment()
for k, v in env.items():
    print(f'  {k}: {v}')
if env['qarp_available'] and not env.get('qarp_mock', False):
    print('  STATUS: Real QARP available - ready for quantum simulation!')
elif env.get('qarp_mock', False):
    print('  STATUS: QARP Mock mode - local testing only')
else:
    print('  STATUS: No QARP - will use local Qulacs fallback')
"

echo ""
echo "=== Environment check complete ==="
