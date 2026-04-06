# QSC2025 FX700 Deployment Package

## Quick Start

1. **Upload project to FX700:**
   ```bash
   scp -r . fx700:~/QARPdemo/qsc2025/
   ```

2. **Verify environment:**
   ```bash
   ssh fx700
   cd ~/QARPdemo/qsc2025
   bash fx700_deploy/check_env.sh
   ```

3. **Run benchmark:**
   ```bash
   bash fx700_deploy/run_benchmarks.sh
   ```

4. **Submit individual jobs via SLURM:**
   ```bash
   sbatch fx700_deploy/job_request.sh
   sbatch fx700_deploy/job_request_12q.sh
   sbatch fx700_deploy/job_request_advantage.sh
   ```

5. **Check results:**
   ```bash
   cat results/benchmark_results.json
   cat results/advantage_result.json
   ```

## Important Notes

- Use `-p Batch` for SLURM (IntrHPC is suspended during QSC2025)
- First pytket-tenet run takes 2+ hours (Julia JIT precompilation)
- Submit tenet jobs ONE AT A TIME (Julia environment conflicts)
- MPI Qulacs requires: C_COMPILER=mpicc CXX_COMPILER=mpic++ USE_MPI=Yes
- Backend priority: qulacs_mpi > qulacs > qiskit_aer > local_sim

## File Manifest

- `check_env.sh` — Environment verification
- `job_*.sh` — SLURM job scripts per problem
- `run_benchmarks.sh` — Full benchmark suite runner
- `../run_optimization.py` — CLI entry point
- `../benchmark_suite.py` — Comprehensive benchmark
- `../main.py` — FastAPI server
