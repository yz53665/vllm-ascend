# 算子源码分析摘要：FusedLightningIndexerManage

路线：DAV_2201（ASCEND910B / ASCEND910_93）→ DAV_3510（ASCEND950 / SOC_VERSION=ascend950dt_9574）
分析依据：源码逆向（`op_kernel/`、`op_host/`），API 差异依据 `api-diff-guide.md`，cube 依据 `cube-migration-guide.md`。

## 1. 算子功能描述

- **计算逻辑**：Lightning Indexer 的 topk 路由 + MTP miss 归并融合算子。
  1. `PrepareNonOffloadRows`（AIV）：按 `request_state` 把非 offload 请求的计数/路由表清零（`DataCopyPad` 写 int32 计数）。
  2. `LIPreload::Process`（AIC+AIV 混合）：AIC 侧 `LIMatmul::ComputeMm1` 用 cube 计算 index score（Q×K），L0C 回写 GM；AIV 侧 topk 选择、阈值比较、写 `topk_*` 输出。
  3. `ProcessDecode`：`ICachePreLoad` + `ProcessLD`，依赖 `SyncAll()` 在核间对齐。
  4. `MtpUnion`（`fused_lightning_indexer_manage_union.h`）：多 route 的 miss 集合归并、slot 分配、输出写回。
- **算子类别**：attention（indexer 打分 = matmul + topk/reduction），含 cube（Mmad）与 vector 双路径。

## 2. 接口签名（`op_host/fused_lightning_indexer_manage_def.cpp`）

- 输入（全部 REQUIRED，FORMAT_ND，AutoContiguous）：
  `index_weights`(BF16/FP16)、`query_dequant_scale`(FP32)、`query`(BF16/FP16)、`index_key_dequant_scale`(FP32)、
  `index_key_cache`(BF16/FP16)、`index_block_table`(INT32)、`actual_seq_lengths_query`(INT32)、
  `actual_seq_lengths_key`(INT32)、`offload_seq_lengths_key`(INT32)、`num_cache_tokens`(INT32)、
  `request_state`(INT32)、`req_pool_entries`(INT32)、`cache_slots_pool`(INT32)
- 输出（全部 REQUIRED，INT32）：`topk_src_ids`、`topk_dst_slots`、`topk_miss_counts`、`miss_src_ids`、
  `miss_dst_slots`、`miss_counts`、`cache_slots_pool_out`
- 属性：无
- 常量约束（`fused_lightning_indexer_manage_constants.h`）：`TOPK=2048`、`MISS_CAPACITY=32768`、
  `CACHE_BLOCK_SIZE=128`、`HEAD_DIM=128`、`KEY_HEADS=1`、`MAX_ROUTES=14`、`MAX_CACHE_TOKENS=32640`、`AIVS_PER_AIC=2`
- 状态码：`REQUEST_STATE_NON_OFFLOAD=-3`、`FIRST_DECODE=-2`、`STEADY=-1`、`PADDING_SOURCE_ID=-1`

## 3. 支持的数据类型

- 主计算：**BF16 / FP16**（`def.cpp` DataType 组合 `{DT_BF16, DT_FLOAT16}`）
- 中间计算：FP32（cube Mmad 的 L0C、vector topk 分数）
- 索引/计数：INT32
- 无 FP8 / HiFloat8 / INT8

## 4. 计算逻辑伪代码

```
kernel fused_lightning_indexer_manage(...):
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
  #if (__CCE_AICORE__ == 310) || defined(__DAV_310R6__) || (__CCE_AICORE__ == 200)
      // ★ 空实现：A5(310) 在此早退，必须移除该分支
  #else
  INVOKE_LI_MTP_TOPK(...):
      PrepareNonOffloadRows(...)            // AIV
      if ASCEND_IS_AIV: SyncAll()            // isAIVOnly=true（默认）
      op.Init(...); op.Process()
      if ASCEND_IS_AIV:
          SyncAll()
          if blockIdx % AIVS_PER_AIC == 0:
              FinalizeStandardCounts(...) 或 MtpUnion::Process(...)
```

`LIPreload::Process()` → `ProcessMain()` + `ProcessDecode()`：

```
ProcessMain():
  if aiCoreIdx >= usedCoreNum: return          // 整核空闲（AIC 与其 2 个 AIV 同时返回）
  if AIV: AllocEventID(); CrossCoreSetFlag<2,PIPE_MTE2>(syncV1C1) ×2
  else:   AllocEventID()
  for bN2 / gS1 / s2: ProcessBaseBlock()
  if AIV: FreeEventID()
  else:   FreeEventID(); CrossCoreWaitFlag(syncV1C1) ×2     // 排空预灌 2 拍
ProcessBaseBlock():
  AIC: CrossCoreWaitFlag(syncV1C1); ComputeMm1(); CrossCoreSetFlag<2,PIPE_FIX>(syncC1V1)
  AIV: CrossCoreWaitFlag(syncC1V1); ProcessVec(); CrossCoreSetFlag<2,PIPE_MTE2>(syncV1C1)
ProcessDecode():
  if AIV: InitLDBuffers(); ICachePreLoad(); SyncAll(); if isLD: ProcessLD()
```

`ComputeMm1`（cube 低阶）：`KeyNd2Nz/KeyNd2NzForPA`、`QueryNd2Nz`（`LoadData3DParamsV2`+`LoadData`）
→ `LoadQueryToL0a` / `LoadKeyToL0b`（`LoadData2DParams`+`LoadData`）→ `Mmad`（`MmadParams`，`cmatrixInitVal=true`）
→ `Fixp`（`DataCopyCO12DstParams` + `SetFixpipeNz2dFlag` → `DataCopy` L0C→GM）。**未使用高阶 Matmul API**。

分块常量：`M_BASIC_BLOCK=256, D_BASIC_BLOCK=128, S2_BASIC_BLOCK=256`；
L0：`M_BASIC_BLOCK_L0=128, D_BASIC_BLOCK_L0=128, S2_BASIC_BLOCK_L0=128`；`KEY_BUF_NUM=3, QUERY_BUF_NUM=2, L0_BUF_NUM=2`。

## 5. 关键 API 使用

| 类别 | API |
|---|---|
| cube 装载 | `LoadData3DParamsV2` / `IsResetLoad3dConfig`、`LoadData2DParams`、`LoadData` |
| cube 计算 | `Mmad` + `MmadParams` |
| L0C 回写 | `DataCopyCO12DstParams` + `SetFixpipeNz2dFlag(1,1,1)` + `DataCopy` |
| Matmul 布局 | `SetMMLayoutTransform(true/false)`（AllocEventID/FreeEventID） |
| 数据搬运 | `DataCopy`、`DataCopyPad`（非对齐 blockLen，含 `{1, uint16_t(bytes), 0, 0}` 形式）、`DataCacheCleanAndInvalid` |
| 核内事件 | `SetFlag/WaitFlag<HardEvent::{MTE1_MTE2,MTE2_MTE1,MTE1_M,M_MTE1,V_MTE3,MTE3_V,V_MTE2,MTE2_V,V_S,S_MTE3,MTE3_S}>` |
| 跨核同步 | `CrossCoreSetFlag<2, pipe>` / `CrossCoreWaitFlag` / `SyncAll` |
| vector | `Cast`、`Sort/TopK` 类（`lightning_indexer_service_vector.h`）、`Reduce` 类 |

## 6. 边界条件与特殊值

- 哨兵值：`INVALID_IDX`（int32）通过 `InitGlobalMemory` 写入 topk 输出；负无穷用裸位模式
  `negInf = 0xFC00`（FP16）/ `0xFF80`（BF16）经 `InitGlobalMemory` 写入 value 输出。
- 空任务核：`aiCoreIdx >= usedCoreNum` 整核返回（AIC 与其 2 个 AIV 同判定，参与集合保持一致）。
- 空 batch：`curActSeqLenIsZero` 走 `DealActSeqLenIsZero` 分支。
- 无 NaN/Inf 输入假设；输出为整数索引，不含浮点累加。

## 7. ★ API 差异风险扫描结果

### 7.1 精度（Subnormal）

- **是否使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal：否**
  证据：`grep -E "\b(Exp|Ln|Log|Sqrt|Rsqrt|Reciprocal|Div)\s*\(" op_kernel/` → 0 命中。
- **是否存在结构性 subnormal 排除（策略 0）**：N/A（无受影响的超越/除法 API）。
- **输入是否可能包含 subnormal**：对上述 API 无影响；算子对 BF16/FP16 只做 `Cast` 与乘加，A5 对
  `Cast`/`Muls`/`Add` 无 subnormal 裁剪差异（`api-diff-guide.md` §1.2 仅列超越函数与除法）。
- **eps/常量 dtype 可表示性检查**：

| 常量 | 值 | FP16 | BF16 | FP32 | 结论 |
|---|---|---|---|---|---|
| `negInf` | 0xFC00 / 0xFF80（位模式） | 精确（-Inf） | 精确（-Inf） | 精确 | 无风险（位模式不受 subnormal 裁剪影响） |
| `LIMConfig::*`（TOPK 等） | 整数 ≤ 32768 | 精确 | 精确 | 精确 | 无风险 |
| `PADDING_SOURCE_ID` | -1 | 精确 | 精确 | 精确 | 无风险 |
| `0`（`FLOAT_ZERO` 类） | 0 | 精确 | 精确 | 精确 | 无风险 |

- **选定适配策略**：**无需处理**（算子内无 subnormal 敏感 API，且无 eps 常量）。

### 7.2 兼容性

- **是否使用 int4b_t 作为 Mmad 输入：否**（数据类型仅 BF16/FP16/FP32）
- **是否使用 4:2 稀疏：否**（grep `Sparse`/`4:2` 无 cube 稀疏调用；算子名中的 "Indexer" 与硬件 4:2 稀疏无关）

### 7.3 性能

- **TransDataTo5HD：否**
- **高阶 ReduceSum（AR 模式）：否**（vector 侧为逐元素/`Sort-TopK` 类，无 `Reduce::AR`）
- **BilinearInterpolation：否**
- **其它命中信号**：存在大量 `DataCopyPad` 非对齐存储（`{1, uint16_t(bytes), 0, 0}` 形式）→ 属
  `stage_1_assess.md` Step 1.4 的 L2 判定信号「DataCopyPad 非对齐存储 → 950 建议 DataCopyUnAlign」。
- **高风险 shape 清单（性能回归）**：
  | # | 场景 | 触发原因 |
  |---|---|---|
  | 1 | `tSize=1`（纯 decode，单 token） | Mmad `m=16` 最小分块，cube 利用率最低；`M=1` 奇异分块风险 |
  | 2 | `s2` 非 128 对齐（如 129/255） | `S2_BASIC_BLOCK_L0=128` 尾块 + `DataCopyPad` 非对齐 |
  | 3 | `bSize=1, tSize=2048`（大 batch 单请求） | topk `Sort` 满规模 |
  | 4 | `routeCount=14`（`MAX_ROUTES` 上限） | union 归并最坏路径 |
  | 5 | `cacheSlotsSize` 接近 `MAX_CACHE_TOKENS=32640` | workspace 与 slot 分配边界 |
  | 6 | `usedCoreNum < blockDim`（部分核空闲） | 跨核同步参与集合边界 |

## 8. 同步协议盘点结果

**① 同步点清单**（`flagId` 全部来自 `ConstInfo`：`syncC1V1 = 0U`、`syncV1C1 = 0U`，tiling 未覆写）

| # | 位置 | 原语 | 模式/pipe | flagId | 角色 |
|---|---|---|---|---|---|
| 1 | `lightning_indexer_kernel.h:664` ProcessMain | CrossCoreSetFlag | `<2, PIPE_MTE2>` | `syncV1C1=0` | AIV（预灌第 1 拍） |
| 2 | `lightning_indexer_kernel.h:665` ProcessMain | CrossCoreSetFlag | `<2, PIPE_MTE2>` | `syncV1C1=0` | AIV（预灌第 2 拍） |
| 3 | `lightning_indexer_kernel.h:696` ProcessMain | CrossCoreWaitFlag | 裸（默认 mode 0 / PIPE_S） | `syncV1C1=0` | AIC（排空） |
| 4 | `lightning_indexer_kernel.h:697` ProcessMain | CrossCoreWaitFlag | 裸 | `syncV1C1=0` | AIC（排空） |
| 5 | `lightning_indexer_kernel.h:706` ProcessBaseBlock | CrossCoreWaitFlag | 裸 | `syncV1C1=0` | AIC（等 AIV 释放 L1） |
| 6 | `lightning_indexer_kernel.h:708` ProcessBaseBlock | CrossCoreSetFlag | `<2, PIPE_FIX>` | `syncC1V1=0` | AIC（mm1 回写完成） |
| 7 | `lightning_indexer_kernel.h:710` ProcessBaseBlock | CrossCoreWaitFlag | 裸 | `syncC1V1=0` | AIV（等 mm1 结果） |
| 8 | `lightning_indexer_kernel.h:712` ProcessBaseBlock | CrossCoreSetFlag | `<2, PIPE_MTE2>` | `syncV1C1=0` | AIV（释放 L1） |
| 9 | `lightning_indexer_kernel.h:722` ProcessDecode | SyncAll | `isAIVOnly=true`（默认） | 14（框架占用） | AIV 全核 |
| 10 | `fused_lightning_indexer_manage.cpp:257` | SyncAll | `isAIVOnly=true` | 14 | AIV 全核 |
| 11 | `fused_lightning_indexer_manage.cpp:279` | SyncAll | `isAIVOnly=true` | 14 | AIV 全核 |

**② 参与集合假设**

- #1/#2/#8 ↔ #3/#4/#5（AIV→AIC）：假设为「AIC ↔ 组内全部 AIV」。纸面计数（每 AI Core）：
  AIC 侧计数器由 AIV 的 Set 驱动，mode 2 下「每个 AIV 各 Set 一次」→ +1；两个 AIV 同步执行 → 每拍 +1。
  初始预灌 +2，循环内每拍 +1；AIC 侧消耗为循环内每拍 1（#5）+ 结尾 2（#3/#4）→ `2 + N == N + 2` ✅
- #6 ↔ #7（AIC→AIV）：假设为「AIC 1 次 Set ↔ 2 个 AIV 各 1 次 Wait」。循环内每拍 AIC Set 1 次、每个 AIV Wait 1 次 ✅
- #9/#10/#11：纯 AIV 全核（`isAIVOnly=true`），执行点均在 `if ASCEND_IS_AIV` 的**无条件路径**上
  （#9 所在 `ProcessDecode` 无 `usedCoreNum` 早退；#10/#11 在 `INVOKE_LI_MTP_TOPK` 顶层）。
- **单向复用同一编号成立**：`syncC1V1 == syncV1C1 == 0`，但 AIC 与每个 AIV 各自维护独立计数器
  （`CrossCoreSetFlag_ISASI.md`「flagId 取值范围说明」），两个方向互不干扰。

**③ 目标平台语义对照（ASCEND950 / DAV_3510）**

- 模式 2 在 950 支持（`CrossCoreSetFlag_ISASI.md` 表 3：模式 2 需 `__mix__(1,1)` / `__mix__(1,2)`；
  本 kernel 为 `KERNEL_TYPE_MIX_AIC_1_2` ✅）。参与集合「AIC ↔ 组内全部 AIV」与假设一致（含等号）✅
- 裸 `CrossCoreWaitFlag(flagId)`：950 上 `modeId` 默认 0，但 **模式 0/1/2 使用同一硬件指令，
  「省略 modeId 不影响接口功能」**（`CrossCoreWaitFlag_ISASI.md`「模板参数默认值及生效情况」950 分支）；
  `pipe` 默认 `PIPE_S` 且在 950 **生效** → 阻塞标量流水（指令发射口），等价于 2201 上「阻塞全部流水」的
  最强约束，**不会放松** ✅。为可读性将在 Stage 2 显式写成 `<2, PIPE_S>`，不改变阻塞强度。
- flagId 占用：本 kernel 未使用高阶 Matmul（无 `REGIST_MATMUL_OBJ`）→ `[0,2N-1]`、KFC 15/31 均不占用；
  `SyncAll`（`__mix__(1,2)` + `isAIVOnly=true`）在 950 占用 mode0/1/2 的 flagId 14（AIV）
  （`key_features.md` 表 1）→ 自定义 flagId 0 不冲突 ✅
- 参与集合恒定性：`ProcessMain` 的整核早退以 `aiCoreIdx` 判定，AIV 的 `aiCoreIdx = blockIdx/2` 与 AIC 的
  `blockIdx` 对应同一物理核集合 → 不存在「AIC 在跑而 AIV 缺席」的半核状态 ✅

**④ 处置决策**

| 同步点 | 结论 | 理由 |
|---|---|---|
| #1–#8 | 沿用（8 个） | 集合粒度 = 文档粒度（含等号）、模式 2 在 950 支持、参数语义已核对（裸等待为最强约束）、参与集合恒定；纸面计数闭合。Stage 2 仅把裸 Wait 显式化为 `<FIA_SYNC_MODE2, PIPE_S>`，不改语义 |
| #9–#11 | 沿用（3 个） | `isAIVOnly=true` 与 950 表 3 一致；执行点在 AIV 无条件路径；占用 flagId 14 与自定义 0 不冲突 |

`[SYNC-AUDIT] 同步点 11 个：沿用 11 个 / 禁用 0 个 / 改用 0 个`

## 9. A5 迁移必改项（非 API 差异，但阻断 A5 编译/选核）

| # | 文件 | 问题 |
|---|---|---|
| 1 | `op_kernel/fused_lightning_indexer_manage.cpp:319` | `#if (__CCE_AICORE__ == 310) \|\| defined(__DAV_310R6__) \|\| (__CCE_AICORE__ == 200)` 空分支 → A5 编出空 kernel，必须移除 `310` 早退 |
| 2 | `op_host/fused_lightning_indexer_manage_def.cpp:73-74` | 仅注册 `ascend910b` / `ascend910_93`，需追加 `ascend950` |
| 3 | `op_host/fused_lightning_indexer_manage_tiling.cpp:26-33` | SoC 白名单只允许 910B / 910_93，需放宽到 ASCEND950 |
| 4 | `op_kernel/lightning_indexer_service_cube.h:450,465` | `SetMMLayoutTransform(true/false)` **沿用不改**（先前"CANN 9.1.0 起改为 SetMMRowMajor/SetMMColumnMajor"的结论已证伪：全仓无该 API，且 A5 已验证文件 `quant_lightning_indexer_v2_service_cube_arch35.h:724,742` 仍在使用） |
| 5 | `op_kernel/lightning_indexer_service_cube.h:408-444` | `Fixp`：A5 分支改 `FixpipeParamsC310<CO2Layout::ROW_MAJOR>` + `Fixpipe<float,float,LI_CFG_ROW_MAJOR_GM>`（A3 分支保留 `DataCopyCO12DstParams`+`SetFixpipeNz2ndFlag`） |
| 6 | `op_kernel/lightning_indexer_service_cube.h:310-322,362-377` | `LoadQueryToL0a`/`LoadKeyToL0b` 的 3D 装载与 repeatTimes 版 `LoadData2DParams` 改为 `LoadData2DParamsV2`（A3 分支保留） |
| 7 | `op_kernel/lightning_indexer_service_cube.h:58-69` | 容量复核**通过**（A5 L1=512KB、L0C=256KB 未缩小）：Q L1 128KB + K L1 192KB = 320KB ≤ 512KB；L0A/L0B 各 64KB=上限；L0C 128KB ≤ 256KB → `M_BASIC_BLOCK/S2_BASIC_BLOCK/KEY_BUF_NUM` 无需调整 |
| 8 | `csrc/build_aclnn.sh` A5 分支 | 已把 `fused_lightning_indexer_manage` 加入 `CUSTOM_OPS_ARRAY` |
| 9 | `op_host/config/ascend950/` | 两算子均无该目录，且 `csrc/attention` 下无 `config/` binary.json 约定 → **无需新增** |
