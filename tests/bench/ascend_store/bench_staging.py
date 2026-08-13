"""Mooncake backend put/get benchmark with per-stage wall-clock timing.

Drives ``MooncakeBackend.put/get`` directly with tunable workload sizes and
reports the time spent in every internal stage, so you can see where the
staging scheme (or the direct multi-buffer path) spends its time.

Workload model: each key carries ``--members`` scattered NPU buffers of
``--block-mb`` MB each — the same shape block aggregation (or per-block
layerwise) hands to the backend. ``--staging-size`` overrides the
``staging_buffer_size`` from mooncake.json without touching the file.

Requires a real node with vLLM + torch_npu + Mooncake + at least one NPU
(no need for a second machine; the store can be a single-node pool).

Usage:
    MOONCAKE_CONFIG_PATH=/path/to/mooncake.json \
    python tests/bench/ascend_store/bench_staging.py \
        --keys 64 --members 4 --block-mb 1 --iters 10 --staging-size 4MB

    # write-only (fresh keys per round — every put is a real write)
    python tests/bench/ascend_store/bench_staging.py --mode put --keys 64 --iters 10

    # read-only (all keys pre-loaded before the timed loop)
    python tests/bench/ascend_store/bench_staging.py --mode get --keys 64 --iters 10

Notes:
    - Timing is done with mock.patch wrappers around each stage method;
      the real implementations still run, so numbers are real wall-clock.
    - ``torch.npu.synchronize`` is timed as "sync(d2d wait)" — that is where
      the actual D2D staging copy executes, not the memcpy launch itself.
    - Mooncake RDMA time appears as "mooncake_rdma"; C++ internals are not
      visible from Python, use torch_npu profiler for kernel-level detail.
    - Each round uses unique keys (``bench:{round}:{i}``), so every put is a
      real write rather than a no-op overwrite.
"""

import argparse
import contextlib
import os
import time
from collections import defaultdict
from unittest import mock

import torch
from vllm.config import ParallelConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeBackend,
    _parse_global_segment_size,
    _staging_memcpy_available,
)


class StageTimer:
    """Accumulate wall-clock time per named stage and print a summary table."""

    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.maxs = defaultdict(float)

    @contextlib.contextmanager
    def time(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.totals[name] += elapsed
            self.counts[name] += 1
            self.maxs[name] = max(self.maxs[name], elapsed)

    def report(self):
        lines = [f"{'stage':<28}{'calls':>6}{'total(s)':>11}{'avg(ms)':>10}{'max(ms)':>10}"]
        for name, total in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            n = self.counts[name]
            lines.append(
                f"{name:<28}{n:>6}{total:>11.3f}{total / n * 1000:>10.3f}{self.maxs[name] * 1000:>10.3f}"
            )
        return "\n".join(lines)


def wrap(timer, owner, attr, stage):
    """Return a mock.patch timing every call to ``owner.attr`` as ``stage``.

    The original implementation still runs inside the timed wrapper.
    """
    orig = getattr(owner, attr)

    def timed(*args, **kwargs):
        with timer.time(stage):
            return orig(*args, **kwargs)

    return mock.patch.object(owner, attr, timed)


def wrap_sync(timer, stage):
    """Time mooncake_backend's torch.npu.synchronize calls (D2D copy wait)."""
    orig = mooncake_backend.torch.npu.synchronize

    def timed():
        with timer.time(stage):
            return orig()

    return mock.patch.object(mooncake_backend.torch.npu, "synchronize", timed)


class TimedStore:
    """Proxy around the C++ Mooncake store that times the RDMA entry points.

    ``MooncakeDistributedStore`` is a pybind11 object: its methods are
    read-only attributes, so ``mock.patch`` cannot wrap them. Instead the
    backend's ``store`` is swapped for this proxy, which forwards every
    attribute and only wraps the two RDMA calls.
    """

    _TIMED_METHODS = ("batch_put_from_multi_buffers", "batch_get_into_multi_buffers")

    def __init__(self, real, timer):
        self._real = real
        self._timer = timer

    def __getattr__(self, attr):
        real_attr = getattr(self._real, attr)
        if attr in self._TIMED_METHODS:
            stage = "put: mooncake_rdma" if attr == "batch_put_from_multi_buffers" else "get: mooncake_rdma"

            def timed(*args, **kwargs):
                with self._timer.time(stage):
                    return real_attr(*args, **kwargs)

            return timed
        return real_attr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to mooncake.json (falls back to $MOONCAKE_CONFIG_PATH).",
    )
    parser.add_argument("--keys", type=int, default=64, help="Number of keys per round.")
    parser.add_argument("--members", type=int, default=4, help="Scattered buffers per key.")
    parser.add_argument("--block-mb", type=float, default=1.0, help="Bytes per member buffer (MiB).")
    parser.add_argument("--iters", type=int, default=10, help="Number of rounds (each round uses unique keys).")
    parser.add_argument(
        "--mode",
        choices=["put", "get", "both"],
        default="both",
        help=(
            "put: write-only (fresh keys per round); "
            "get: read-only (keys pre-loaded then each round reads a different batch); "
            "both: put then get per round."
        ),
    )
    parser.add_argument(
        "--staging-size",
        default=None,
        help="Override staging_buffer_size (bytes, e.g. '4MB'); None keeps mooncake.json.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that get() returned the exact bytes put() sent.",
    )
    return parser.parse_args()


def alloc_buffers(count, block_bytes):
    """Allocate `count` independent NPU uint8 buffers (scattered addresses)."""
    tensors = [torch.empty(block_bytes, dtype=torch.uint8, device="npu") for _ in range(count)]
    return [buf.data_ptr() for buf in tensors], tensors


def main():
    args = parse_args()
    block_bytes = int(args.block_mb * 1024 * 1024)
    total_members = args.keys * args.members

    if args.config:
        os.environ["MOONCAKE_CONFIG_PATH"] = args.config
    if not os.getenv("MOONCAKE_CONFIG_PATH"):
        raise SystemExit("MOONCAKE_CONFIG_PATH is not set; pass --config or export it.")

    torch.npu.set_device(0)
    backend = MooncakeBackend(ParallelConfig(), lazy_init=False, contribute_memory=True)

    if args.staging_size is not None:
        backend.staging_buffer_size = _parse_global_segment_size(args.staging_size)
        backend._staging_enabled = backend.staging_buffer_size > 0 and _staging_memcpy_available()
        backend._staging_pool = None  # rebuilt lazily on next put/get
    print(
        f"workload: keys={args.keys} members/key={args.members} "
        f"block={block_bytes} B total={total_members * block_bytes / 1024**2:.1f} MiB/key-buffers "
        f"| staging_enabled={backend._staging_enabled} "
        f"staging_buffer_size={backend.staging_buffer_size}"
    )

    print(f"allocating {total_members * 2} NPU buffers ...")
    put_ptrs, put_tensors = alloc_buffers(total_members, block_bytes)
    get_ptrs, get_tensors = alloc_buffers(total_members, block_bytes)
    if args.verify:
        for i, buf in enumerate(put_tensors):
            buf.fill_(i % 251)
        torch.npu.synchronize()
    # Registered memory is required for RDMA access.
    backend.register_buffer(put_ptrs, [block_bytes] * total_members)
    backend.register_buffer(get_ptrs, [block_bytes] * total_members)

    addrs = [put_ptrs[k * args.members:(k + 1) * args.members] for k in range(args.keys)]
    get_addrs = [get_ptrs[k * args.members:(k + 1) * args.members] for k in range(args.keys)]
    sizes = [[block_bytes] * args.members for _ in range(args.keys)]

    timer = StageTimer()

    if args.mode == "get":
        # Pre-load all keys the timed loop will read. Runs on the raw store
        # (no proxy yet) so it is excluded from the report.
        print(f"pre-loading {args.iters * args.keys} keys ...")
        for it in range(args.iters):
            pre_keys = [f"bench:{it}:{i}" for i in range(args.keys)]
            backend.put(pre_keys, addrs, sizes)
        torch.npu.synchronize()
    # C++ store methods are read-only; time RDMA through a forwarding proxy.
    backend.store = TimedStore(backend.store, timer)

    for it in range(args.iters):
        keys = [f"bench:{it}:{i}" for i in range(args.keys)]

        if args.mode in ("put", "both"):
            with (
                wrap(timer, backend, "_stage_put", "put: stage_put(py)"),
                wrap(timer, backend, "_launch_memcpy", "put: memcpy.launch"),
                wrap_sync(timer, "put: sync(d2d wait)"),
            ):
                with timer.time("put: total"):
                    backend.put(keys, addrs, sizes)

        if args.mode in ("get", "both"):
            with (
                wrap(timer, backend, "_stage_get", "get: stage_get(py)"),
                wrap(timer, backend, "_scatter_back_get", "get: scatter_back(py)"),
                wrap(timer, backend, "_launch_memcpy", "get: memcpy.launch"),
                wrap_sync(timer, "get: sync(d2d wait)"),
            ):
                with timer.time("get: total"):
                    res = backend.get(keys, get_addrs, sizes)

            failed = sum(1 for v in res if v != 0) if res else -1
            if failed:
                print(f"round {it}: get failed keys={failed}")

    if args.verify:
        torch.npu.synchronize()
        matched = sum(
            1
            for k in range(args.keys)
            for m in range(args.members)
            if torch.equal(get_tensors[k * args.members + m], put_tensors[k * args.members + m])
        )
        print(f"verify: {matched}/{total_members} member buffers match")

    print("\n" + timer.report())


if __name__ == "__main__":
    main()
