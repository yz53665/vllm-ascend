# Standard
import functools
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.parallel_state import get_global_rank

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB
# Number of contiguous staging buffers pre-allocated per rank. The pool grows
# on demand when a single put/get needs more concurrent buffers.
DEFAULT_STAGING_NUM_BUFFERS = 4
STAGING_COPY_BLOCK_SIZE = 8192


@functools.lru_cache(maxsize=1)
def _mooncake_setup_supports_ssd_offload() -> bool:
    """True when installed Mooncake exposes SSD kwargs on setup() (v0.3.11+)."""
    from mooncake.store import MooncakeDistributedStore  # type: ignore

    setup = MooncakeDistributedStore.setup
    try:
        import inspect

        sig = inspect.signature(setup)
        return "enable_ssd_offload" in sig.parameters
    except (TypeError, ValueError):
        # pybind11 overloaded bindings often reject inspect.signature
        doc = setup.__doc__ or ""
        return "enable_ssd_offload" in doc


def _ssd_setup_kwargs(config: "MooncakeStoreConfig") -> dict[str, object]:
    """Keyword args for store.setup(); empty on old Mooncake or when SSD is off."""
    if not config.enable_ssd_offload:
        return {}
    if not _mooncake_setup_supports_ssd_offload():
        raise RuntimeError(
            "mooncake.json has enable_ssd_offload=true, but the installed "
            "Mooncake does not support enable_ssd_offload/ssd_offload_path in "
            "MooncakeDistributedStore.setup(). Upgrade Mooncake to v0.3.11 or "
            "later (see Mooncake ssd-offload.md Step 3A), or set "
            "enable_ssd_offload to false."
        )
    return {
        "enable_ssd_offload": config.enable_ssd_offload,
        "ssd_offload_path": config.ssd_offload_path,
    }


@functools.lru_cache(maxsize=1)
def _staging_memcpy_available() -> bool:
    """True when the pointer-level Triton memcpy kernel can be used.

    The kernel is not usable on 310P (fallback path there uses torch tensor
    views, which the KV pool cannot build from raw block pointers).
    """
    try:
        from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel  # noqa: F401
        from vllm_ascend.utils import is_310p

        return not is_310p()
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _get_batch_memcpy_kernel():
    from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

    return batch_memcpy_kernel


class _StagingBufferPool:
    """Pool of pre-allocated contiguous NPU staging buffers.

    A single store object key may map to several scattered block slices
    (Mooncake multi-buffer semantics). Staging copies those slices into one
    (or more, when the total exceeds ``buffer_size``) contiguous buffer so
    Mooncake transfers a compact blob instead of many small scattered
    segments. The pool grows on demand and is registered with the transfer
    engine once per buffer (skipped in fabric-memory mode).
    """

    def __init__(self, backend: "MooncakeBackend", buffer_size: int, num_buffers: int = DEFAULT_STAGING_NUM_BUFFERS):
        self._backend = backend
        self.buffer_size = buffer_size
        self._buffers: list[torch.Tensor] = []
        self._free: list[int] = []
        self._lock = threading.Lock()
        with self._lock:
            for _ in range(num_buffers):
                self._add_buffer_unlocked()

    def _add_buffer_unlocked(self) -> None:
        buf = torch.empty(self.buffer_size, dtype=torch.uint8, device="npu")
        self._buffers.append(buf)
        self._free.append(len(self._buffers) - 1)
        if not self._backend._use_fabric_mem:
            local_hostname = get_ip()
            global_te.get_transfer_engine(local_hostname, device_name=None)
            global_te.register_buffer([buf.data_ptr()], [self.buffer_size])

    def acquire(self, count: int) -> list[int]:
        """Acquire ``count`` staging buffer ids, growing the pool if needed."""
        with self._lock:
            while len(self._free) < count:
                self._add_buffer_unlocked()
            acquired = self._free[:count]
            del self._free[:count]
            return acquired

    def release(self, buffer_ids: list[int]) -> None:
        if not buffer_ids:
            return
        with self._lock:
            self._free.extend(buffer_ids)

    def ptr(self, buffer_id: int) -> int:
        return self._buffers[buffer_id].data_ptr()


class MooncakeBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False, contribute_memory: bool = True):
        self.parallel_config = parallel_config
        self.config = MooncakeStoreConfig.load_from_env()
        if self.config.protocol != "ascend":
            raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")

        self.store: Any | None = None
        self.local_seg: str | None = None
        self._use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        self._lazy_init = lazy_init and self._use_fabric_mem
        self._contribute_memory = contribute_memory
        self._store_initialized = False
        self._store_init_lock = threading.Lock()

        # Staging coalescing: copy each key's scattered block slices into
        # contiguous staging buffers before transfer. Enabled by
        # `staging_buffer_size` (bytes) in mooncake.json; independent of
        # block_aggregation so it also covers per-block multi-layer slices.
        self.staging_buffer_size = self.config.staging_buffer_size
        self._staging_enabled = self.staging_buffer_size > 0 and _staging_memcpy_available()
        self._staging_pool: _StagingBufferPool | None = None
        self._staging_pool_lock = threading.Lock()
        if self.staging_buffer_size > 0 and not self._staging_enabled:
            logger.warning(
                "staging_buffer_size=%d requested but staging is disabled "
                "(pointer-level Triton memcpy unavailable on this platform). "
                "Falling back to direct multi-buffer transfer.",
                self.staging_buffer_size,
            )

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    def ensure_initialized(self):
        if self._store_initialized:
            return

        with self._store_init_lock:
            if self._store_initialized:
                return

            logger.info("Initializing Mooncake store. metadata_server=%s", self.config.metadata_server)
            self.store = self._setup_store()
            self._store_initialized = True

    def _setup_store(self):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        store = MooncakeDistributedStore()
        local_hostname = get_ip()
        ssd_kwargs = _ssd_setup_kwargs(self.config)
        # Each rank that contributes memory to the pool uses its own SSD
        # directory to avoid bucket file collisions. Key by the globally unique
        # rank so that DP/TP/PP/CP replicas never share a directory (dense and
        # MoE alike); only ranks that contribute memory need an offload dir.
        if ssd_kwargs and ssd_kwargs.get("ssd_offload_path") and self._contribute_memory:
            global_rank = get_global_rank(self.parallel_config)
            rank_path = os.path.join(str(ssd_kwargs["ssd_offload_path"]), f"rank_{global_rank}")
            try:
                os.makedirs(rank_path, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create per-rank SSD offload directory: {rank_path!r} ({e})")
            ssd_kwargs["ssd_offload_path"] = rank_path
        # ASCEND_ENABLE_USE_FABRIC_MEM: Enable unified memory address direct transmission scheme
        # and only can be used for 800 I/T A3 series.
        # Required supporting hardware versions are as follows:
        if not self._use_fabric_mem:
            transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
            self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size if self._contribute_memory else 0,
                local_buffer_size=self.config.local_buffer_size if self._contribute_memory else 0,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                engine=transfer_engine.get_engine(),
                **ssd_kwargs,
            )
        else:
            self.local_seg = local_hostname
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size if self._contribute_memory else 0,
                local_buffer_size=0,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                **ssd_kwargs,
            )

        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(
                "Initialize mooncake failed. ret=%d, metadata_server=%s. Check mooncake config and network.",
                ret,
                self.config.metadata_server,
            )
            raise RuntimeError(msg)
        if ssd_kwargs:
            logger.info(
                "Mooncake SSD offload enabled (Mode A): path=%s",
                self.config.ssd_offload_path,
            )
        return store

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        torch.npu.set_device(0)
        return cls(parallel_config, contribute_memory=False)

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        if not self._use_fabric_mem:
            local_hostname = get_ip()
            global_te.get_transfer_engine(local_hostname, device_name=None)
            global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MooncakeBackend.exists called before store initialization; treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        self.ensure_initialized()
        assert self.store is not None
        staging = None
        if getattr(self, "_staging_enabled", False):
            try:
                staging = self._stage_put(addrs, sizes)
            except Exception:
                logger.warning(
                    "Staging put failed for %d keys; falling back to direct "
                    "multi-buffer transfer.",
                    len(keys),
                    exc_info=True,
                )
                staging = None
        try:
            config = ReplicateConfig()
            if self.config.preferred_segment:
                config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = self.config.prefer_alloc_in_same_node
            if staging is not None:
                addrs, sizes = staging["addrs"], staging["sizes"]
                # Wait for the staging copies to land before Mooncake reads them.
                torch.npu.synchronize()
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            failed_codes = [int(value) for value in res if value < 0]
            failed_count = len(failed_codes)
            if failed_count:
                error_codes = sorted(set(failed_codes))
                logger.error(
                    "Failed to put %d keys out of %d. error_codes=%s. Check memory and store capacity.",
                    failed_count,
                    len(keys),
                    error_codes,
                )
                logger.debug("Failed to put key details. keys=%s, result=%s", keys, res)
                if self._lazy_init:
                    logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
        except Exception as e:
            logger.error(
                "Failed to put %d keys out of %d. type=%s, error=%s. Check store state and memory.",
                len(keys),
                len(keys),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to put key details. keys=%s", keys)
            if self._lazy_init:
                logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
        finally:
            if staging is not None:
                self._ensure_staging_pool().release(staging["buffer_ids"])

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        if self._lazy_init and not self._store_initialized:
            logger.error(
                "Failed to get %d keys out of %d. Store is not initialized; "
                "call put() first to trigger initialization.",
                len(keys),
                len(keys),
            )
            logger.debug("Failed to get key details. keys=%s", keys)
            return
        assert self.store is not None
        logger.debug(
            "MooncakeBackend.get enter keys=%d sample_keys=%s",
            len(keys),
            keys[:3],
        )
        staging = None
        if getattr(self, "_staging_enabled", False):
            try:
                staging = self._stage_get(addrs, sizes)
            except Exception:
                logger.warning(
                    "Staging get failed for %d keys; falling back to direct "
                    "multi-buffer transfer.",
                    len(keys),
                    exc_info=True,
                )
                staging = None
        try:
            if staging is not None:
                addrs, sizes = staging["addrs"], staging["sizes"]
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
            res_list = list(res)
            failed_codes = [int(value) for value in res_list if value < 0]
            failed_count = len(failed_codes)
            error_codes = sorted(set(failed_codes))
            if failed_count:
                logger.error(
                    "Failed to get %d keys out of %d. error_codes=%s. Check key existence and memory state.",
                    failed_count,
                    len(keys),
                    error_codes,
                )
                logger.debug("Failed to get key details. keys=%s, result=%s", keys, res_list)
            for i, value in enumerate(res_list):
                if value > 0:
                    res_list[i] = 0
            if staging is not None:
                self._scatter_back_get(staging, res_list)
                # Wait for the scatter copies: callers read the KV cache
                # immediately after get() returns.
                torch.npu.synchronize()
            return res_list
        except Exception as e:
            logger.error(
                "Failed to get %d keys out of %d. type=%s, error=%s. Check store state and network.",
                len(keys),
                len(keys),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to get key details. keys=%s", keys)
            return None
        finally:
            if staging is not None:
                self._ensure_staging_pool().release(staging["buffer_ids"])

    def _ensure_staging_pool(self) -> _StagingBufferPool:
        if self._staging_pool is None:
            with self._staging_pool_lock:
                if self._staging_pool is None:
                    self._staging_pool = _StagingBufferPool(self, self.staging_buffer_size)
        return self._staging_pool

    @staticmethod
    def _chunk_members(addrs: list[list[int]], sizes: list[list[int]], buffer_size: int):
        """Split each key's scattered (addr, size) members into staging chunks.

        A key is staged only when it carries more than one buffer. Its members
        are packed into consecutive chunks of at most ``buffer_size`` bytes;
        members are never split across chunks. Keys with a single buffer (or
        none) are passed through untouched.

        Returns a list aligned with ``addrs``; each entry is either None (not
        staged) or a list of ``(members, total_bytes)`` chunk tuples.
        """
        chunks_per_key = []
        for key_addrs, key_sizes in zip(addrs, sizes):
            if len(key_addrs) <= 1:
                chunks_per_key.append(None)
                continue
            chunks: list[tuple[list[tuple[int, int]], int]] = []
            members: list[tuple[int, int]] = []
            total = 0
            for addr, size in zip(key_addrs, key_sizes):
                if members and total + size > buffer_size:
                    chunks.append((members, total))
                    members, total = [], 0
                members.append((addr, size))
                total += size
            if members:
                chunks.append((members, total))
            chunks_per_key.append(chunks)
        return chunks_per_key

    def _stage_put(self, addrs: list[list[int]], sizes: list[list[int]]):
        """Copy scattered member buffers into contiguous staging buffers.

        Returns a dict with the staged ``addrs``/``sizes`` (staging buffer
        pointers, one entry per chunk of at most ``staging_buffer_size``
        bytes) plus the acquired buffer ids, or None when nothing needs
        staging. The caller must wait on the copies before handing the
        buffers to Mooncake and release the ids afterwards.
        """
        chunks_per_key = self._chunk_members(addrs, sizes, self.staging_buffer_size)
        if not any(chunks is not None for chunks in chunks_per_key):
            return None
        pool = self._ensure_staging_pool()
        total_chunks = sum(len(chunks) for chunks in chunks_per_key if chunks is not None)
        buffer_ids = pool.acquire(total_chunks)
        staged_addrs: list[list[int]] = []
        staged_sizes: list[list[int]] = []
        copy_ops: list[tuple[int, int, int]] = []
        buffer_idx = 0
        for key_addrs, key_sizes, chunks in zip(addrs, sizes, chunks_per_key):
            if chunks is None:
                staged_addrs.append(key_addrs)
                staged_sizes.append(key_sizes)
                continue
            key_staged_addrs: list[int] = []
            key_staged_sizes: list[int] = []
            for members, total in chunks:
                ptr = pool.ptr(buffer_ids[buffer_idx])
                offset = 0
                for addr, size in members:
                    copy_ops.append((addr, ptr + offset, size))
                    offset += size
                key_staged_addrs.append(ptr)
                key_staged_sizes.append(total)
                buffer_idx += 1
            staged_addrs.append(key_staged_addrs)
            staged_sizes.append(key_staged_sizes)
        if copy_ops:
            self._launch_memcpy(copy_ops)
        return {
            "addrs": staged_addrs,
            "sizes": staged_sizes,
            "buffer_ids": buffer_ids,
        }

    def _stage_get(self, addrs: list[list[int]], sizes: list[list[int]]):
        """Reserve contiguous staging buffers for a get request.

        Returns the staged ``addrs``/``sizes`` (staging buffer pointers), the
        flattened ``(key_idx, members, total)`` chunks for the scatter-back
        and the acquired buffer ids, or None when nothing needs staging.
        """
        chunks_per_key = self._chunk_members(addrs, sizes, self.staging_buffer_size)
        if not any(chunks is not None for chunks in chunks_per_key):
            return None
        pool = self._ensure_staging_pool()
        total_chunks = sum(len(chunks) for chunks in chunks_per_key if chunks is not None)
        buffer_ids = pool.acquire(total_chunks)
        staged_addrs: list[list[int]] = []
        staged_sizes: list[list[int]] = []
        chunks_flat: list[tuple[int, list[tuple[int, int]], int]] = []
        buffer_idx = 0
        for key_idx, (key_addrs, key_sizes, chunks) in enumerate(zip(addrs, sizes, chunks_per_key)):
            if chunks is None:
                staged_addrs.append(key_addrs)
                staged_sizes.append(key_sizes)
                continue
            key_staged_addrs: list[int] = []
            key_staged_sizes: list[int] = []
            for members, total in chunks:
                key_staged_addrs.append(pool.ptr(buffer_ids[buffer_idx]))
                key_staged_sizes.append(total)
                chunks_flat.append((key_idx, members, total))
                buffer_idx += 1
            staged_addrs.append(key_staged_addrs)
            staged_sizes.append(key_staged_sizes)
        return {
            "addrs": staged_addrs,
            "sizes": staged_sizes,
            "buffer_ids": buffer_ids,
            "chunks": chunks_flat,
        }

    def _scatter_back_get(self, staging: dict, results: list[int] | None) -> None:
        """Copy data fetched into staging buffers back to the scattered block
        addresses for every key that loaded successfully."""
        chunks = staging["chunks"]
        if not chunks:
            return
        copy_ops: list[tuple[int, int, int]] = []
        pool = self._ensure_staging_pool()
        for (key_idx, members, _total), buffer_id in zip(chunks, staging["buffer_ids"]):
            if results is not None and key_idx < len(results) and results[key_idx] != 0:
                continue
            ptr = pool.ptr(buffer_id)
            offset = 0
            for addr, size in members:
                copy_ops.append((ptr + offset, addr, size))
                offset += size
        if copy_ops:
            self._launch_memcpy(copy_ops)

    @staticmethod
    def _launch_memcpy(copy_ops: list[tuple[int, int, int]]) -> None:
        """Launch a batched pointer-level device copy (scattered <-> contiguous)."""
        kernel = _get_batch_memcpy_kernel()
        src_ptrs = torch.tensor([op[0] for op in copy_ops], dtype=torch.int64, device="npu")
        dst_ptrs = torch.tensor([op[1] for op in copy_ops], dtype=torch.int64, device="npu")
        sizes = torch.tensor([op[2] for op in copy_ops], dtype=torch.int64, device="npu")
        kernel[(len(copy_ops),)](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=STAGING_COPY_BLOCK_SIZE)


@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    preferred_segment: bool
    prefer_alloc_in_same_node: bool
    enable_ssd_offload: bool = False
    ssd_offload_path: str = ""
    # Bytes of contiguous staging buffers used to coalesce a key's scattered
    # block slices before transfer. 0 disables staging.
    staging_buffer_size: int = 0

    def __post_init__(self) -> None:
        if not self.enable_ssd_offload:
            return
        if not self.ssd_offload_path:
            raise ValueError(
                "enable_ssd_offload is true but ssd_offload_path is empty. Set ssd_offload_path in mooncake.json."
            )
        if not os.path.isabs(self.ssd_offload_path):
            raise ValueError(f"ssd_offload_path must be an absolute path, got: {self.ssd_offload_path!r}")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        master_server_address = os.getenv("MOONCAKE_MASTER", None)
        global_segment_size_env = os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=_parse_global_segment_size(
                global_segment_size_env
                if global_segment_size_env is not None
                else config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            master_server_address=master_server_address
            if master_server_address is not None
            else config.get("master_server_address"),
            preferred_segment=config.get("preferred_segment", False),
            prefer_alloc_in_same_node=config.get("prefer_alloc_in_same_node", True),
            enable_ssd_offload=bool(config.get("enable_ssd_offload", False)),
            ssd_offload_path=config.get("ssd_offload_path", ""),
            staging_buffer_size=_parse_global_segment_size(config.get("staging_buffer_size", 0)),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)


def _parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: GB, MB, KB, B

    Args:
        value: Input value (int, str, or other convertible types)

    Returns:
        int: Size in bytes

    Raises:
        ValueError: For invalid format, missing number, or negative values
        TypeError: For unsupported input types
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    UNIT_MULTIPLIERS = {
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"

    multiplier = UNIT_MULTIPLIERS[unit]
    return _convert_to_bytes(number_str, multiplier, value)


def _convert_to_bytes(number_str: str, multiplier: int, original_input: str) -> int:
    """
    Convert numeric string to byte count

    Args:
        number_str: Numeric portion of input
        multiplier: Unit conversion factor
        original_input: Original input string (for error messages)

    Returns:
        int: Byte count

    Raises:
        ValueError: For invalid numbers or negative results
    """
    try:
        numeric_value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{original_input}'")
    # Calculate byte count
    try:
        byte_count = int(numeric_value * multiplier)
    except OverflowError:
        raise ValueError(f"Storage size too large: '{original_input}'")
    return byte_count
