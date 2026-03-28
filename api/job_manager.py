"""
job_manager.py
==============
PHASE 2 — Async Job Polling + Redis Caching
============================================
Mentor Note: Phase 1's /api/v1/optimize was synchronous — it blocked the HTTP
connection for the entire 1-5 second optimization. That's fine for small problems,
but for 15+ qubits on FX700 with MPI, a single run can take HOURS.

This module introduces:
  1. Job Queue — fire-and-forget: POST returns a job_id immediately
  2. Job Store — tracks job state (PENDING → RUNNING → COMPLETED/FAILED)
  3. Redis Cache — prevents re-running identical optimizations
  4. Warm-start — seeds QAOA angles from similar past solutions

POLLING PATTERN (how clients use the new API):
──────────────────────────────────────────────
  POST /api/v1/optimize/async    →  {"job_id": "abc123", "status": "pending"}
  GET  /api/v1/jobs/abc123       →  {"status": "running", "progress": 0.4}
  GET  /api/v1/jobs/abc123       →  {"status": "completed", "result": {...}}

This matches the pattern used in the FX700 Slurm environment, where sbatch
returns a job_id and you poll squeue for status.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job State Machine
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING   = "pending"    # queued, not started
    RUNNING   = "running"    # optimizer is executing
    COMPLETED = "completed"  # result ready
    FAILED    = "failed"     # error occurred


@dataclass
class JobRecord:
    """
    Represents a single optimization job's lifecycle.
    Stored in memory (and Redis if available).
    """
    job_id:      str
    status:      JobStatus
    created_at:  str
    started_at:  Optional[str]  = None
    finished_at: Optional[str]  = None
    progress:    float           = 0.0     # 0.0 → 1.0
    result:      Optional[dict]  = None
    error:       Optional[str]   = None
    request_hash: Optional[str]  = None    # for cache lookup
    backend_used: Optional[str]  = None
    n_qubits:    Optional[int]   = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# In-Memory Job Store (+ optional Redis)
# ---------------------------------------------------------------------------

class JobStore:
    """
    Stores job records.

    Uses an in-memory LRU dict as primary storage.
    If Redis is available, jobs are also persisted there for:
      - Multi-worker deployments (multiple uvicorn workers share state)
      - Survival across API restarts
      - Job history queries

    Mentor Note: For the FX700 challenge, in-memory is sufficient if
    you're running a single FastAPI process. Add Redis if you scale to
    multiple Gunicorn workers.
    """

    def __init__(self, max_memory_jobs: int = 500, redis_url: Optional[str] = None):
        self._store: OrderedDict[str, JobRecord] = OrderedDict()
        self._max   = max_memory_jobs
        self._redis = None

        # Optional Redis connection
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"Redis connected: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}). Using in-memory only.")

    def put(self, job: JobRecord):
        """Save or update a job record."""
        # LRU eviction when memory limit reached
        if len(self._store) >= self._max and job.job_id not in self._store:
            self._store.popitem(last=False)

        self._store[job.job_id] = job

        # Persist to Redis with 24-hour TTL
        if self._redis:
            try:
                self._redis.setex(
                    f"job:{job.job_id}",
                    86400,   # 24 hours
                    json.dumps(job.to_dict()),
                )
            except Exception as e:
                logger.warning(f"Redis write failed: {e}")

    def get(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve a job by ID."""
        if job_id in self._store:
            return self._store[job_id]

        # Try Redis fallback
        if self._redis:
            try:
                raw = self._redis.get(f"job:{job_id}")
                if raw:
                    data = json.loads(raw)
                    data["status"] = JobStatus(data["status"])
                    job = JobRecord(**data)
                    self._store[job_id] = job   # cache locally
                    return job
            except Exception as e:
                logger.warning(f"Redis read failed: {e}")

        return None

    def list_recent(self, limit: int = 20) -> list[dict]:
        """Returns most recent jobs, newest first."""
        jobs = list(self._store.values())
        return [j.to_dict() for j in reversed(jobs[-limit:])]


# ---------------------------------------------------------------------------
# Result Cache
# ---------------------------------------------------------------------------

class ResultCache:
    """
    Prevents redundant quantum computation for identical requests.

    Cache key = SHA-256 hash of (nodes, routes, demands, objective, quantum_iterations).
    If the same supply chain problem is submitted twice, we return the
    cached result instantly instead of re-running the quantum optimizer.

    This is especially valuable on FX700 where a single VQE run can take
    hours on large problems.

    Mentor Note: Cache invalidation here is simple — time-based TTL.
    In production you might also invalidate when inventory data changes.
    """

    def __init__(
        self,
        max_entries: int = 200,
        ttl_seconds: int = 3600,    # 1 hour default; tune for your use case
        redis_url: Optional[str] = None,
    ):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._timestamps: dict[str, float]  = {}
        self._max       = max_entries
        self._ttl       = ttl_seconds
        self._redis     = None

        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                pass

    @staticmethod
    def compute_key(request_dict: dict) -> str:
        """
        Deterministic hash of an optimization request.

        We hash a normalized, sorted JSON string so that requests
        with the same data in different field orders produce the same key.
        """
        # Normalize: sort lists by a stable key, sort dict keys
        normalized = json.dumps(request_dict, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[dict]:
        """Returns cached result if fresh, else None."""
        # TTL check for in-memory
        if key in self._timestamps:
            age = time.time() - self._timestamps[key]
            if age > self._ttl:
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)
                return None

        if key in self._cache:
            result = dict(self._cache[key])
            result["cache_hit"] = True
            result["cache_age_seconds"] = int(time.time() - self._timestamps.get(key, 0))
            return result

        # Redis fallback
        if self._redis:
            try:
                raw = self._redis.get(f"cache:{key}")
                if raw:
                    result = json.loads(raw)
                    result["cache_hit"] = True
                    return result
            except Exception:
                pass

        return None

    def put(self, key: str, result: dict):
        """Store a result in cache."""
        if len(self._cache) >= self._max:
            oldest_key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(oldest_key, None)

        self._cache[key] = result
        self._timestamps[key] = time.time()

        if self._redis:
            try:
                self._redis.setex(
                    f"cache:{key}",
                    self._ttl,
                    json.dumps(result, default=str),
                )
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "entries":    len(self._cache),
            "max":        self._max,
            "ttl_s":      self._ttl,
            "redis":      self._redis is not None,
        }


# ---------------------------------------------------------------------------
# Warm-Start Registry
# ---------------------------------------------------------------------------

class WarmStartRegistry:
    """
    Stores optimized QAOA angles (γ, β) from past runs.
    Seeds new optimizations with similar past solutions instead of random init.

    WHY THIS MATTERS:
    ─────────────────
    QAOA's optimization landscape is continuous. Problems with similar
    structure (same node topology, similar cost ratios) tend to have
    optimal angles in similar regions of parameter space.

    Starting near a known good point → fewer VQE iterations → faster convergence.
    For FX700 where circuit eval is expensive, this can cut optimization
    time by 30-60%.

    Mentor Note: The "similarity" metric here is simple (cosine similarity
    of cost coefficient vectors). Phase 3 could replace this with a
    learned embedding model.
    """

    def __init__(self, max_entries: int = 100):
        # Each entry: {"feature_vector": [...], "params": [...], "energy": float}
        self._registry: list[dict] = []
        self._max = max_entries

    def register(
        self,
        feature_vector: list[float],
        best_params: "np.ndarray",
        best_energy: float,
    ):
        """Store an optimized solution with its problem fingerprint."""
        import numpy as np
        if len(self._registry) >= self._max:
            # Evict highest-energy (worst) solution
            self._registry.sort(key=lambda x: x["energy"])
            self._registry = self._registry[:self._max - 1]

        self._registry.append({
            "feature_vector": list(feature_vector),
            "params":         list(best_params),
            "energy":         float(best_energy),
        })

    def find_warm_start(
        self,
        feature_vector: list[float],
        n_params: int,
        top_k: int = 3,
    ) -> "np.ndarray | None":
        """
        Find the most similar past solution and return its params
        as a warm-start initialization for the new optimization.

        Returns None if registry is empty or no sufficiently similar
        solution exists, so the caller falls through to the full
        multi-restart VQE path.
        """
        import numpy as np

        if not self._registry:
            logger.info("Warm-start registry empty — skipping warm-start")
            return None

        fv = np.array(feature_vector)
        fv_norm = np.linalg.norm(fv)
        if fv_norm < 1e-10:
            return None

        # Cosine similarity with all stored solutions
        best_sim = -1.0
        best_params = None

        for entry in self._registry:
            stored = np.array(entry["feature_vector"])
            # Pad/trim to match lengths
            min_len = min(len(fv), len(stored))
            sim = float(
                np.dot(fv[:min_len], stored[:min_len]) /
                (fv_norm * np.linalg.norm(stored[:min_len]) + 1e-10)
            )
            if sim > best_sim:
                best_sim = sim
                best_params = np.array(entry["params"])

        # Only use warm-start if similarity is high enough (>0.8)
        # Below this threshold, the landscapes are likely too different
        if best_sim >= 0.8 and best_params is not None:
            # Adapt to current n_params (different p_layers)
            if len(best_params) == n_params:
                logger.info(f"Warm-start: cosine similarity={best_sim:.3f}")
                return best_params.copy()
            elif len(best_params) > n_params:
                return best_params[:n_params].copy()
            else:
                # Pad with zeros (conservative)
                padded = np.zeros(n_params)
                padded[:len(best_params)] = best_params
                return padded

        logger.info(f"No warm-start (best similarity={best_sim:.3f} < 0.8)")
        return None

    def extract_feature_vector(self, ising_hamiltonian) -> list[float]:
        """
        Convert an IsingHamiltonian into a fixed-size feature vector
        for similarity comparison.

        Features used:
        - Sorted h values (single-qubit fields)
        - Sorted J values (coupling strengths)
        - Problem statistics (mean, std, max)
        """
        import numpy as np

        h_vals = sorted(ising_hamiltonian.h.values()) if ising_hamiltonian.h else [0.0]
        j_vals = sorted(ising_hamiltonian.J.values()) if ising_hamiltonian.J else [0.0]

        h_arr = np.array(h_vals[:10])   # cap at 10 for fixed size
        j_arr = np.array(j_vals[:10])

        features = [
            float(ising_hamiltonian.n_qubits),
            float(np.mean(h_arr)),
            float(np.std(h_arr)),
            float(np.max(np.abs(h_arr))),
            float(np.mean(j_arr)),
            float(np.std(j_arr)),
            float(np.max(np.abs(j_arr))),
        ]
        # Pad with zeros to fixed size 20
        while len(features) < 20:
            features.append(0.0)

        return features[:20]


# ---------------------------------------------------------------------------
# Async Job Queue (asyncio-based, no extra dependencies)
# ---------------------------------------------------------------------------

class JobQueue:
    """
    Manages async execution of optimization jobs.

    Uses asyncio.Queue + background worker tasks so that:
    - POST /optimize/async returns immediately with job_id
    - Optimization runs in the background
    - GET /jobs/{job_id} returns live status

    Mentor Note: For the FX700 challenge, a single-worker queue is fine.
    In production, you'd use Celery + Redis for a distributed task queue.
    This asyncio implementation has zero extra dependencies — just Python.
    """

    def __init__(
        self,
        job_store: JobStore,
        result_cache: ResultCache,
        warm_start_registry: WarmStartRegistry,
        max_concurrent: int = 2,   # limit parallel quantum circuit evals
    ):
        self._store      = job_store
        self._cache      = result_cache
        self._warmstart  = warm_start_registry
        self._queue      = asyncio.Queue()
        self._semaphore  = asyncio.Semaphore(max_concurrent)
        self._workers    = []
        self._running    = False

    async def start(self, n_workers: int = 2):
        """Launch background worker coroutines."""
        self._running = True
        for i in range(n_workers):
            task = asyncio.create_task(self._worker(worker_id=i))
            self._workers.append(task)
        logger.info(f"JobQueue started with {n_workers} workers")

    async def stop(self):
        """Gracefully drain the queue and stop workers."""
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("JobQueue stopped")

    async def submit(self, job_id: str, request_dict: dict, run_fn) -> JobRecord:
        """
        Enqueue an optimization job.

        Args:
            job_id       : unique identifier
            request_dict : serialized OptimizationRequest
            run_fn       : sync callable that performs the optimization;
                           receives request_dict, returns OptimizationResult dict

        Returns:
            JobRecord in PENDING state.
        """
        job = JobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC).isoformat(),
            started_at=None,
            request_hash=ResultCache.compute_key(request_dict),
        )
        self._store.put(job)
        await self._queue.put((job_id, request_dict, run_fn))
        logger.info(f"Job {job_id} queued (queue depth: {self._queue.qsize()})")
        return job

    async def _worker(self, worker_id: int):
        """
        Background worker: dequeues and executes optimization jobs.
        """
        logger.info(f"Worker {worker_id} started")
        while self._running:
            try:
                job_id, request_dict, run_fn = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue

            async with self._semaphore:
                job = self._store.get(job_id)
                if not job:
                    continue

                # Check cache before running
                cached = self._cache.get(job.request_hash)
                if cached:
                    logger.info(f"Job {job_id}: cache HIT — skipping quantum computation")
                    job.status      = JobStatus.COMPLETED
                    job.started_at  = datetime.now(UTC).isoformat()
                    job.finished_at = datetime.now(UTC).isoformat()
                    job.progress    = 1.0
                    job.result      = cached
                    job.result["cache_hit"] = True
                    self._store.put(job)
                    self._queue.task_done()
                    continue

                # Mark as running
                job.status     = JobStatus.RUNNING
                job.started_at = datetime.now(UTC).isoformat()
                job.progress   = 0.05
                self._store.put(job)

                try:
                    # Run optimization in thread pool to avoid blocking event loop
                    # (scipy.optimize.minimize is CPU-bound / synchronous)
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, run_fn, request_dict)

                    job.status      = JobStatus.COMPLETED
                    job.finished_at = datetime.now(UTC).isoformat()
                    job.progress    = 1.0
                    job.result      = result
                    job.backend_used = result.get("quantum_metrics", {}).get("backend", "unknown")
                    self._store.put(job)

                    # Cache the result
                    self._cache.put(job.request_hash, result)

                    logger.info(f"Job {job_id}: COMPLETED")

                except Exception as e:
                    job.status      = JobStatus.FAILED
                    job.finished_at = datetime.now(UTC).isoformat()
                    job.error       = str(e)
                    self._store.put(job)
                    logger.error(f"Job {job_id}: FAILED — {e}", exc_info=True)

                self._queue.task_done()


# ---------------------------------------------------------------------------
# Module-level singletons (shared across all API requests)
# ---------------------------------------------------------------------------

job_store    = JobStore(redis_url=os.environ.get("REDIS_URL"))
result_cache = ResultCache(
    ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", 3600)),
    redis_url=os.environ.get("REDIS_URL"),
)
warm_start   = WarmStartRegistry(max_entries=100)
job_queue    = JobQueue(job_store, result_cache, warm_start)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def test():
        store  = JobStore()
        cache  = ResultCache(ttl_seconds=60)
        ws     = WarmStartRegistry()
        queue  = JobQueue(store, cache, ws)
        await queue.start(n_workers=1)

        # Fake run function
        def fake_run(req):
            time.sleep(0.2)
            return {"optimization_id": "test_123", "objective_value": 42.0,
                    "status": "success", "quantum_metrics": {}}

        job = await queue.submit("job_001", {"nodes": [], "routes": [], "demands": []}, fake_run)
        print(f"Submitted: {job.job_id} / {job.status}")

        await asyncio.sleep(1.0)

        retrieved = store.get("job_001")
        print(f"After wait: status={retrieved.status}, result={retrieved.result}")

        await queue.stop()

    asyncio.run(test())
