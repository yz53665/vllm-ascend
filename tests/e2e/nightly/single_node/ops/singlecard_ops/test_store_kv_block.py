import gc
import time

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op, load_custom_ops_library

torch.set_printoptions(threshold=np.inf)

enable_custom_op()
# ASCEND950 的硬件 profile 未声明 RUNTIME_CUSTOM_OPS，enable_custom_op() 在 A5 上返回 False，
# 不会注册 _C_ascend 算子。本用例只验证单个算子，单独加载扩展即可。
load_custom_ops_library()


def cal_slot(key, key_cache, slot_mapping, block_size):
    key_expect = key_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        block_index = slot // block_size
        block_offset = slot % block_size
        key_expect[block_index][block_offset] = token_key
    return key_expect.npu()


def cal_scatternd(key, key_cache, slot_mapping, block_size):
    key_expect = key_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        key_expect[slot] = token_key

    return key_expect.npu()


@pytest.mark.parametrize("num_tokens", [16])  # 6398
@pytest.mark.parametrize("num_head", [1])  # 512
@pytest.mark.parametrize("block_size", [128])  # 128
@pytest.mark.parametrize("num_blocks", [1773])  # 1599
@pytest.mark.parametrize("count", [1])
def test_scatter(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k = 64
    key = torch.randint(low=0, high=128, size=(num_tokens, num_head, head_size_k), dtype=torch.int8).npu()

    key_cache = torch.randint(
        low=0, high=128, size=(num_blocks * block_size, num_head, head_size_k), dtype=torch.int8
    ).npu()

    slot_list = []
    for i in range(0, num_tokens):
        slot_list.append([2 + i])
        # slot_list.append(6+i)
    assert num_tokens == len(slot_list)
    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_scatternd(key, key_cache, slot_mapping_npu, block_size)
    N = 101
    for i in range(N):
        torch_npu.npu_scatter_nd_update_(key_cache, slot_mapping_npu, key)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1)


@pytest.mark.parametrize("num_tokens", [52])  # 6398
@pytest.mark.parametrize("num_head", [1])  # 512
@pytest.mark.parametrize("block_size", [1])  # 128
@pytest.mark.parametrize("num_blocks", [1773])  # 1599
@pytest.mark.parametrize("count", [1])
def test_myops(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k = 2
    key_cache = torch.randint(low=0, high=128, size=(num_blocks, block_size, num_head, head_size_k), dtype=torch.int8)
    key_cache_npu = key_cache.npu()

    slot_list = []
    for i in range(0, num_tokens):
        slot_list.append(2 + i)

    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key = torch.randint(low=0, high=128, size=(num_tokens, head_size_k), dtype=torch.int8)
    key_npu = key.npu()
    key_expect = cal_slot(key_npu, key_cache_npu, slot_list_np, block_size)

    time.sleep(0.1)

    group_len = torch.empty(num_tokens, dtype=torch.int32).npu()
    group_key_idx = torch.empty(num_tokens, dtype=torch.int32).npu()
    group_key_cache_idx = torch.empty(num_tokens, dtype=torch.int32).npu()
    torch.ops._C_ascend.store_kv_block_metadata(
        slot_mapping_npu, group_len, group_key_idx, group_key_cache_idx, block_size
    )
    torch.ops._C_ascend.store_kv_block(
        key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size
    )

    torch.testing.assert_close(key_expect, key_cache_npu, atol=0.001, rtol=0.1)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [52])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [1])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("count", [1])
def test_myops_fp8_key_float32_scale(num_tokens, num_head, block_size, num_blocks, count):
    """LI C8 with an FP8 indexer cache.

    The indexer K cache is float8_e4m3fn and its dequant scale is float32,
    so store_kv_block has to handle 1-byte fp8 rows and 4-byte fp32 rows.
    """
    head_size_k = 2
    key_cache = torch.rand((num_blocks, block_size, num_head, head_size_k), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_cache = torch.rand((num_blocks, block_size, num_head, 1), dtype=torch.float32)

    slot_list_np = np.array([2 + i for i in range(num_tokens)])
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key = torch.rand((num_tokens, head_size_k), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.rand((num_tokens, 1), dtype=torch.float32)

    key_cache_expect = key_cache.clone()
    key_cache_expect.view(-1, num_head, head_size_k)[2:2 + num_tokens] = key.view(num_tokens, num_head, head_size_k)
    scale_cache_expect = scale_cache.clone()
    scale_cache_expect.view(-1, num_head, 1)[2:2 + num_tokens] = scale.view(num_tokens, num_head, 1)

    key_npu = key.npu()
    key_cache_npu = key_cache.npu()
    scale_npu = scale.npu()
    scale_cache_npu = scale_cache.npu()

    group_len = torch.empty(num_tokens, dtype=torch.int32).npu()
    group_key_idx = torch.empty(num_tokens, dtype=torch.int32).npu()
    group_key_cache_idx = torch.empty(num_tokens, dtype=torch.int32).npu()
    torch.ops._C_ascend.store_kv_block_metadata(
        slot_mapping_npu, group_len, group_key_idx, group_key_cache_idx, block_size
    )
    torch.ops._C_ascend.store_kv_block(
        key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size
    )
    # The fp32 dequant scale goes through the same operator (tilingKey 4 path).
    torch.ops._C_ascend.store_kv_block(
        scale_npu, scale_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size
    )

    torch.testing.assert_close(key_cache_expect.float(), key_cache_npu.float().cpu(), atol=0, rtol=0)
    torch.testing.assert_close(scale_cache_expect, scale_cache_npu.cpu(), atol=0, rtol=0)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
