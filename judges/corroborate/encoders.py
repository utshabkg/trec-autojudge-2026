"""Model loading, device selection, and an on-disk score cache.

Three things here are deliberate rather than incidental.

1. `pick_device` proves CUDA with a real matmul. `torch.cuda.is_available()`
   returns True on this driver/wheel combination even when CUDA initialisation
   has actually failed, so only a real operation settles it.
2. The cross-encoder is driven through plain `transformers` rather than
   `sentence_transformers.CrossEncoder`. We need P(entailment) specifically,
   and which activation `CrossEncoder.predict` applies by default has changed
   across versions. Computing it from the logits ourselves is version-proof.
3. Every score is cached on disk keyed by content, never by position. The cache
   is what makes an interrupted run cheap to resume and the ablation variants
   nearly free, and it holds only hashes and floats, never text.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .textutil import cache_key, normalize

log = logging.getLogger(__name__)

# Below this much free VRAM we stay on CPU rather than risk an OOM mid-run.
# These GPUs are shared and another user's job routinely grows to ~37 GiB of the
# 47 GiB, so headroom measured at startup is not headroom five minutes later. The
# threshold is deliberately generous and is only the first line of defence: the
# real protection is the OOM retry in _batched_with_oom_retry.
_MIN_FREE_VRAM_BYTES = 6 * 1024**3


def limit_threads(n: int = 8):
    """Cap every thread pool we can reach, and return a context for BLAS.

    This is not tuning, it is a correctness-of-citizenship issue on a shared
    machine. Defaults here are 64 OpenBLAS threads, 60 OpenMP and 60 torch on a
    120-core box, and the clustering issues very many small matrix products. Each
    one then launches a 60-plus thread parallel region for a few milliseconds of
    work, which drove load average to 123 and made the machine unusable for its
    owner while doing almost no useful computation.

    Returns a context manager, because OpenBLAS reads its environment at import
    and cannot be reconfigured by setting variables afterwards. threadpoolctl
    changes it in place for the duration of a block.
    """
    n = max(1, int(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=n)
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


def pick_device(prefer: Optional[str] = None) -> str:
    """Return 'cuda:N' for the emptiest usable GPU, else 'cpu'.

    `prefer` short-circuits the probe, which is what the tests and the
    container use.
    """
    if prefer:
        return prefer
    env = os.environ.get("CORROBORATE_DEVICE")
    if env:
        return env
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        best, best_free = None, 0
        for i in range(torch.cuda.device_count()):
            try:
                free, _total = torch.cuda.mem_get_info(i)
            except Exception:
                continue
            if free > best_free:
                best, best_free = i, free
        if best is None or best_free < _MIN_FREE_VRAM_BYTES:
            log.warning(
                "No GPU with >= %.1f GiB free, using CPU", _MIN_FREE_VRAM_BYTES / 2**30
            )
            return "cpu"
        # Prove it: is_available() can lie on a driver/wheel mismatch.
        dev = f"cuda:{best}"
        _ = (torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev)).sum().item()
        log.info("Using %s (%.1f GiB free)", dev, best_free / 2**30)
        return dev
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("CUDA probe failed (%s), using CPU", exc)
        return "cpu"


class ScoreCache:
    """Content-addressed sqlite cache for vectors and scalars.

    Safe to point several processes at the same file: WAL mode plus a lock
    around writes. A cache miss is never an error, so a corrupt or unwritable
    cache degrades to recomputation instead of failing the run.
    """

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, dim INT, vec BLOB)"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS ce (k TEXT PRIMARY KEY, score REAL)"
                )
                self._conn.commit()
            except Exception as exc:
                log.warning("Score cache disabled (%s): %s", self.path, exc)
                self._conn = None

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def get_vectors(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        if not self._conn or not keys:
            return {}
        out: Dict[str, np.ndarray] = {}
        try:
            with self._lock:
                for chunk in _chunks(list(keys), 900):
                    q = f"SELECT k, dim, vec FROM emb WHERE k IN ({','.join('?' * len(chunk))})"
                    for k, dim, blob in self._conn.execute(q, chunk):
                        out[k] = np.frombuffer(blob, dtype=np.float32, count=dim)
        except Exception as exc:
            log.warning("Cache read failed, recomputing: %s", exc)
        return out

    def put_vectors(self, items: Sequence[Tuple[str, np.ndarray]]) -> None:
        if not self._conn or not items:
            return
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO emb (k, dim, vec) VALUES (?, ?, ?)",
                    [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items],
                )
                self._conn.commit()
        except Exception as exc:
            log.warning("Cache write failed: %s", exc)

    def get_scores(self, keys: Sequence[str]) -> Dict[str, float]:
        if not self._conn or not keys:
            return {}
        out: Dict[str, float] = {}
        try:
            with self._lock:
                for chunk in _chunks(list(keys), 900):
                    q = f"SELECT k, score FROM ce WHERE k IN ({','.join('?' * len(chunk))})"
                    for k, s in self._conn.execute(q, chunk):
                        out[k] = float(s)
        except Exception as exc:
            log.warning("Cache read failed, recomputing: %s", exc)
        return out

    def put_scores(self, items: Sequence[Tuple[str, float]]) -> None:
        if not self._conn or not items:
            return
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO ce (k, score) VALUES (?, ?)",
                    [(k, float(s)) for k, s in items],
                )
                self._conn.commit()
        except Exception as exc:
            log.warning("Cache write failed: %s", exc)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            finally:
                self._conn = None


def _chunks(xs: List, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


class Encoders:
    """Lazily loaded bi-encoder and cross-encoder with caching.

    Models load on first use, so a judge configured with
    `use_cross_encoder: false` never pays to load one, and the CPU fallback
    path never touches CUDA.
    """

    def __init__(
        self,
        bi_encoder: str,
        cross_encoder: Optional[str] = None,
        *,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        embed_batch_size: int = 256,
        cross_batch_size: int = 128,
        cross_max_length: int = 512,
        local_files_only: bool = True,
    ):
        self.bi_name = bi_encoder
        self.cross_name = cross_encoder
        self.device = pick_device(device)
        self.embed_batch_size = embed_batch_size
        self.cross_batch_size = cross_batch_size
        self.cross_max_length = cross_max_length
        self.local_files_only = local_files_only
        self.cache = ScoreCache(Path(cache_dir) / "corroborate.db" if cache_dir else None)
        self._bi = None
        self._cross = None
        self._cross_tok = None
        self._entail_index: Optional[int] = None
        self._cross_labels: int = 0

    # ------------------------------------------------------------ OOM handling

    def _move_to_cpu(self, why: str) -> None:
        """Give up on the GPU for the rest of the run.

        Loaded models are moved rather than reloaded, and un-halved, because fp16
        on CPU is far slower than fp32.
        """
        log.warning("moving encoders to CPU: %s", why)
        self.device = "cpu"
        try:
            if self._bi is not None:
                self._bi.to("cpu")
            if self._cross is not None:
                self._cross.float().to("cpu")
        except Exception as exc:
            log.warning("could not move models, will reload on CPU: %s", exc)
            self._bi = None
            self._cross = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _batched_with_oom_retry(self, items, run, batch_size: int, what: str):
        """Run `run(chunk)` over `items`, surviving a contended GPU.

        On CUDA OOM the batch size is halved and the whole pass retried, down to
        one item, and then the models move to CPU and it is retried again. This
        matters because the alternative is losing a multi-hour run to another
        user's job growing, and because a run that silently died is worse than a
        slow one.
        """
        import torch

        bs = max(1, batch_size)
        while True:
            try:
                out = []
                for i in range(0, len(items), bs):
                    out.extend(run(items[i : i + bs]))
                return out
            except torch.OutOfMemoryError:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if bs > 1:
                    bs = max(1, bs // 2)
                    log.warning("CUDA OOM in %s, retrying at batch_size=%d", what, bs)
                    continue
                if self.device != "cpu":
                    self._move_to_cpu(f"OOM in {what} even at batch_size=1")
                    bs = max(1, batch_size)
                    continue
                raise

    # ---------------------------------------------------------------- bi-encoder

    def _load_bi(self):
        if self._bi is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading bi-encoder %s on %s", self.bi_name, self.device)
            self._bi = SentenceTransformer(
                self.bi_name, device=self.device, local_files_only=self.local_files_only
            )
        return self._bi

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised embeddings, cached per (model, normalised text).

        Returns a (len(texts), dim) float32 array in input order. Duplicates in
        the input are encoded once.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        norm = [normalize(t) for t in texts]
        keys = [cache_key(self.bi_name, t) for t in norm]
        hit = self.cache.get_vectors(keys)

        # Unique misses only, in first-appearance order so the batch is stable.
        todo: List[str] = []
        todo_keys: List[str] = []
        seen: set[str] = set()
        for k, t in zip(keys, norm):
            if k not in hit and k not in seen:
                seen.add(k)
                todo.append(t)
                todo_keys.append(k)

        if todo:
            def _encode(chunk):
                model = self._load_bi()
                return list(
                    model.encode(
                        chunk,
                        batch_size=len(chunk),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                )

            vecs = np.asarray(
                self._batched_with_oom_retry(
                    todo, _encode, self.embed_batch_size, "embed"
                ),
                dtype=np.float32,
            )
            self.cache.put_vectors(list(zip(todo_keys, vecs)))
            for k, v in zip(todo_keys, vecs):
                hit[k] = v

        dim = len(next(iter(hit.values())))
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, k in enumerate(keys):
            out[i] = hit[k]
        return out

    # ------------------------------------------------------------- cross-encoder

    def _load_cross(self):
        if self._cross is None:
            if not self.cross_name:
                raise ValueError("no cross_encoder configured")
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

            cfg = AutoConfig.from_pretrained(self.cross_name, local_files_only=self.local_files_only)
            self._cross_labels = int(getattr(cfg, "num_labels", 1) or 1)
            id2label = {int(k): str(v).lower() for k, v in (cfg.id2label or {}).items()}
            self._entail_index = next(
                (i for i, lab in id2label.items() if "entail" in lab), None
            )
            log.info(
                "Loading cross-encoder %s on %s (labels=%d, entail_index=%s)",
                self.cross_name, self.device, self._cross_labels, self._entail_index,
            )
            self._cross_tok = AutoTokenizer.from_pretrained(
                self.cross_name, local_files_only=self.local_files_only
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.cross_name, local_files_only=self.local_files_only
            )
            model.eval().to(self.device)
            if self.device.startswith("cuda"):
                model.half()
            self._cross = model
        return self._cross

    def cross(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """Support score in [0, 1] for each (premise, hypothesis) pair.

        For a 3-way NLI head this is P(entailment). For a 1-logit reranker it is
        sigmoid(logit), which is a relevance score rather than entailment. Both
        are monotone "does the premise support the hypothesis" signals, which is
        all the caller needs, but they are not interchangeable in absolute
        terms, so the multilingual path is calibrated separately.
        """
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        norm = [(normalize(a), normalize(b)) for a, b in pairs]
        keys = [cache_key(self.cross_name or "", a, b) for a, b in norm]
        hit = self.cache.get_scores(keys)

        todo: List[Tuple[str, str]] = []
        todo_keys: List[str] = []
        seen: set[str] = set()
        for k, p in zip(keys, norm):
            if k not in hit and k not in seen:
                seen.add(k)
                todo.append(p)
                todo_keys.append(k)

        if todo:
            import torch

            def _score(batch):
                model = self._load_cross()
                with torch.inference_mode():
                    enc = self._cross_tok(
                        [a for a, _ in batch],
                        [b for _, b in batch],
                        padding=True,
                        truncation=True,
                        max_length=self.cross_max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = model(**enc).logits.float()
                    if self._cross_labels == 1:
                        out = torch.sigmoid(logits.squeeze(-1))
                    elif self._entail_index is not None:
                        out = torch.softmax(logits, dim=-1)[:, self._entail_index]
                    else:
                        # Unlabelled multi-class head: fall back to index 1,
                        # which is the entailment slot in every NLI checkpoint
                        # we ship. Logged loudly because it is a guess.
                        log.warning("cross-encoder %s has no entailment label, using index 1",
                                    self.cross_name)
                        out = torch.softmax(logits, dim=-1)[:, min(1, logits.shape[-1] - 1)]
                    return out.detach().cpu().tolist()

            scores: List[float] = self._batched_with_oom_retry(
                todo, _score, self.cross_batch_size, "cross-encode"
            )
            self.cache.put_scores(list(zip(todo_keys, scores)))
            for k, s in zip(todo_keys, scores):
                hit[k] = s

        return np.asarray([hit[k] for k in keys], dtype=np.float32)

    def close(self) -> None:
        self.cache.close()
