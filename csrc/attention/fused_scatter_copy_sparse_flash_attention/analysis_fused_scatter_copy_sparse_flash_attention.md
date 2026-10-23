# 算子源码分析摘要：FusedScatterCopySparseFlashAttention

路线：DAV_2201（ASCEND910B / ASCEND910_93）→ DAV_3510（ASCEND950 / SOC_VERSION=ascend950dt_9574）
分析依据：源码逆向（`op_kernel/`、`op_host/`），API 差异依据 `api-diff-guide.md`，cube 依据 `cube-migration-guide.md`。

## 1. 算子功能描述

- **计算逻辑**：Sparse Flash Attention（MLA）+ source-aware gather + scatter-copy 的融合算子，用于
  `use_fused_copy_sfa` / `use_fused_overlap` 路径（默认关闭）。
  1. `IsBatchFirstFill()` 判定是否首次填充：成立则走 `RunFirstFillScatterCopy<T>`（AIV only），
     把 HBM 侧 key/rope/scatter 到 DRAM KV cache（`missSourceIds`/`missDstSlots`/`missCounts` 驱动的 gather-scatter）。
  2. `SyncAll<false>()`：MIX 全核屏障，建立 first-fill 的 MTE3 写 → Attention MTE2 读依赖。
  3. `RunFusedMtp<T, SOURCE_AWARE>`：`Init()` → `InitSourceAwareGather(dramKeyRope, dramKvCache, dramBlockTable,
     topkSourceIds, topkMissCounts, copyCap, dramMaxBlockNum)` → `Process()`。
  4. `Process()` → `ProcessBalance()`：AIC 侧两次 Mmad（mm1 = QK^T、mm2 = PV），AIV 侧 softmax/MergeKv/
     nUpdate 校正，形成 3 槽软件流水（`SFA_PRELOAD_TASK_CACHE_SIZE=3`，`PRELOAD_NUM=2`）。
- **算子类别**：attention（MLA sparse + cube/vector 混合流水）。

## 2. 接口签名（`op_host/fused_scatter_copy_sparse_flash_attention_def.cpp`）

- 输入（全部 REQUIRED，FORMAT_ND，AutoContiguous）：
  `query`(BF16/FP16)、`key`(BF16/FP16)、`value`(BF16/FP16)、`sparse_indices`(INT32)、`cache_tokens`(INT32)、
  `hbm_block_table`(INT32)、`actual_seq_lengths_query`(INT32)、`actual_seq_lengths_kv`(INT32)、
  `query_rope`(BF16/FP16)、`hbm_key_rope`(BF16/FP16)、`dram_key_rope`(BF16/FP16)、`dram_kv_cache`(BF16/FP16)、
  `dram_block_table`(INT32)、`topk_source_ids`(INT32)、`topk_miss_counts`(INT32)、`miss_src_ids`(INT32)、
  `miss_dst_slots`(INT32)、`miss_counts`(INT32)
- 输出：`attention_out`(BF16/FP16)
- 属性：`scale_value`(Float, 1.0)、`sparse_block_size`(Int, 1)、`layout_query`("TND")、
  `layout_kv`("PA_BSND")、`sparse_mode`(Int, 3)
- 实例化：`SFAType<T, T, T, false, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND, V_TEMPLATE(=1), SFA_STAGE_NORMAL,
  SOURCE_AWARE, true>`；`TEMPLATE_MODE == V_TEMPLATE` 恒成立（`.cpp:4` 宏固定为 1）
- 关键常量（`kernel_mla.h`）：`PRELOAD_NUM=2`、`N_BUFFER_M_BASIC_SIZE=256`、`SFA_PRELOAD_TASK_CACHE_SIZE=3`、
  `SYNC_V0_C1_FLAG=6`、`SYNC_C1_V1_FLAG=7`、`SYNC_V1_C2_FLAG=8`、`SYNC_C2_V2_FLAG=9`、`SYNC_C2_V1_FLAG=4`、
  `SYNC_V1_NUPDATE_C2_FLAG=5`、`SYNC_MM2RES_BUF{1,2}_FLAG=10/11`、`SYNC_FDOUTPUT_BUF_FLAG=12`（后三者仅声明、未使用）

## 3. 支持的数据类型

- 主计算：**BF16 / FP16**（`def.cpp` DataType 组合 `{DT_BF16, DT_FLOAT16}`）
- 中间计算：FP32（softmax 系数、Mmad L0C、`T` 模板参数在本算子的 vector 计算中为 float）
- 索引/计数/block table：INT32
- 无 FP8 / HiFloat8 / INT8；`int4b_t` 仅出现在 `common.h:101` 的类型分支，**未参与 Mmad**

## 4. 计算逻辑伪代码

```
kernel fused_scatter_copy_sparse_flash_attention(...):
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
  if TILING_KEY_IS(1):
    firstFill = IsBatchFirstFill(cacheTokens, missCounts, batchSize, missCap)   // AIC/AIV 同输入、同结果
    if firstFill:
        RunFirstFillScatterCopy<T>(...)        // AIV only：gather + scatter 到 DRAM KV
        SyncAll<false>()                       // MIX 全核屏障（AIC 与 AIV 均到达）
        if AIV: pipe.Reset()
    RunFusedMtp<T, SOURCE_AWARE>(...):         // firstFill ? SOURCE_AWARE=false : true
        Init(...) → InitSourceAwareGather(...) → Process()
```

`FusedScatterCopySparseFlashAttentionMla::Process()`：

```
if aiCoreIdx < usedCoreNum:
  AIV: vectorService.AllocEventID(); InitSoftmaxDefaultBuffer()
  AIC: matmulService.AllocEventID()
  ProcessBalance()
  AIV/AIC: FreeEventID()

ProcessBalance():
  AIC: Set<2,PIPE_FIX>(syncC2V1); Set<2,PIPE_MTE2>(3) ×4
  for bN2 / gS1 / s2（含 extraLoop=isEnd?2:0 排空）:
      PreloadPipeline(loop, ...):
          CalcParams / CalcSinnerTopKBegin
          if info0.isValid:
              AIC: Wait(syncV0C1); ComputeMm1(info0)      // mm1 + Set<2,PIPE_FIX>(syncC1V1)
              AIV: Wait(3); MergeKv(info0); Set<2,PIPE_MTE3>(syncV0C1)
          if info2.isValid:
              AIV: ProcessVec1L(info2)                    // Wait(syncC1V1)→Vec1→Set(syncV1C2)→Wait(syncC2V1)
                                                          // →ProcessAmlaNupdate→Set(syncV1NupdateC2)
              AIC: ComputeMm2(info2)                      // Wait(syncV1C2)→mm2→Set(syncC2V2)+Set(syncC2V1)
                   if V_TEMPLATE: Set<2,PIPE_MTE2>(3)
          if info1.isValid: AIV: ProcessVec2L(info1)      // Wait(syncC2V2)→Vec2
  AIV: Wait(syncC2V1); Wait(3) ×4
```

`InitOutputSingleCore()`（仅 AIV，且 `needInit && LAYOUT_T != TND`）：`InitOutput` 分段清零 + `SyncAll()`（`isAIVOnly=true`）。

## 5. 关键 API 使用

| 类别 | API |
|---|---|
| cube 装载 | A3：`LoadData3DParamsV2` / `IsResetLoad3dConfig`（`LOAD3DV2_CONFIG`）、`LoadData2DParams`、`LoadData`；A5：`LoadData2DParamsV2`（Stage 2 已按模式 A 改写，见 §9.1） |
| cube 计算 | `Mmad` + `MmadParams`（mm1/mm2 各 1 处，`cmatrixInitVal` 复用 L0C，`(m/16)*(n/16)<10` 时 `PipeBarrier<PIPE_M>`） |
| L0C 回写 | `FixpipeParamsV220` + `Fixpipe`（mm1/mm2 各 1 处）+ `SetAtomicAdd<MM_OUT_T>` |
| 数据搬运 | `DataCopy`、`DataCopyPad`、`DataCopyParams`、`DataCacheCleanAndInvalid` |
| vector 数学 | `Muls`、`Adds`、`Sub`、`Add`、`Mul`、`Div`、`Exp`、`Cast`、`Brcb`、`RowMuls`、`Maxs`、`Duplicate` |
| 规约 | `Reduce*`（softmax max/sum），无 `Pattern::Reduce::AR` 高阶用法 |
| 核内事件 | `SetFlag/WaitFlag<HardEvent::{V_MTE3,MTE3_V,V_MTE2,MTE2_V,MTE1_MTE2,MTE2_MTE1,MTE1_M,M_MTE1,M_FIX}>` |
| 跨核同步 | `CrossCoreSetFlag<2, pipe>` / `CrossCoreWaitFlag`（裸调用）/ `SyncAll<false>` / `SyncAll` |

## 6. 边界条件与特殊值

- `IsBatchFirstFill()`：首次填充走额外的 AIV-only gather/scatter + `SyncAll<false>`。
- `curActSeqLenIsZero` → `DealActSeqLenIsZero`（空 KV 行，写哨兵）。
- `actualSingleProcessSInnerSize == 0` → 跳过 nUpdate 校正（**只跳过 `Set(syncV1NupdateC2)`**，
  AIC 侧对应 Wait 随之不执行，见 §8）。
- `mSplitInfo.vecDealM == 0`（奇数 AIV 分不到行）→ `ProcessVec2SingleBuf` 提前返回。
- 溢出/NaN：所有 `Exp` 的输入在数学上有界（见 §7.1），不存在结构性下溢/溢出。仅
  `Stage2MergeAndCopyOut` 的 `Exp(M - mergedM)` 在 `M - mergedM ≪ 0` 时可下溢，但
  `mergedM = max(prevM, curM)` 保证必有一支 `exp(0)=1`，无除零、无 NaN。
- 输出清零：`matmul::InitOutput<OUT_T>`（adv_api 工具函数，非 Matmul 对象，不占跨核 flagId）。

## 7. ★ API 差异风险扫描结果

### 7.1 精度（Subnormal）

- **是否使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal：是（12 处）**

| # | 位置 | API | 说明 |
|---|---|---|---|
| 1 | `service_vector_mla.h:568` | `Exp(scaleUb, scaleUb, dealRowCount)` | `scale = exp(max * RECIP_OF_LN2 * LN2)`（Amla scale） |
| 2 | `service_vector_mla.h:766` | `Exp(nTmp, nTmp, calCount)` | 重标定系数 `cof` 的指数项 |
| 3 | `service_vector_mla.h:785` | `Div(tmpCofUb, nTmp, nUpdateTmp2)` | `cof(i) = exp(i)/exp(i-1)` |
| 4 | `service_vector_mla.h:789` | `Div(epsUb, cofValue, tmpCof)` | `1/cof(i)`（首次内循环） |
| 5 | `service_vector_mla.h:792` | `Div(epsUb, cofValuePrev, tmpCof)` | `cof(i-1)/cof(i)` |
| 6–7 | `service_vector_mla.h:2357,2441` | `Div(rawLUb/curLUb, ..., scaleUb)` | LSE 归一化 |
| 8–9 | `service_vector_mla.h:2453,2454` | `Exp(coefPrevUb)` / `Exp(coefCurUb)` | LSE 组合 |
| 10–12 | `service_vector_mla.h:2573,2586,2592` | `Div(dst, src0, src1, mask, ...)` | 按列/分片的除法变体 |

- **★ 是否存在结构性 subnormal 排除（策略 0）**：**是（有闭合证明，无需代码改动）**。

  逐站点输入域推导（`T` 在本算子 vector 计算中恒为 `float`，`service_vector_mla.h:32`）：

  | # | 站点 | `Exp`/`Div` 输入的构造式 | 输入域 | 输出域 | 结论 |
  |---|---|---|---|---|---|
  | 1 | `L568` | `M*(1/LN2) + round(-M*(1/LN2))` 再 `*LN2` = `frac(M/LN2)*LN2` | `[-0.347, 0.347]` | `[0.707, 1.414]` | FP32 normal ✅ |
  | 2 | `L766` | 同 #1（`L736` 取整 → `L760` 还原 → `L762` 相加 → `L764` 乘 `LN2`） | `[-0.347, 0.347]` | `[0.707, 1.414]` | FP32 normal ✅ |
  | 3 | `L768/L770` | `nTmp ∈ [0.707, 1.414]` 的 FP32→KV_T→FP32 往返 Cast | — | FP16 `[0.707,1.414]`（min normal 6.1e-5）；BF16 同理 | **不产生 subnormal** ✅ |
  | 4 | `L785` | 分母 = #3 的 `nUpdateTmp2` | `[0.707, 1.414]` | `cof ∈ [0.5, 2]` | 分母有界非零 ✅ |
  | 5 | `L789/L792` | 分母 = `cof ∈ [0.5, 2]` | — | `epsUb ∈ [0.25, 4]` | 分母有界非零 ✅ |
  | 6 | `L2357/L2441` | 分母 = `CalcAmlaScaleFromMax` 的 `Exp` 输出 | `[0.707, 1.414]` | — | 分母有界非零 ✅ |
  | 7 | `L2573/2586/2592` | `RowDivs` 实现，分母 = `scaleRowUb` 或 `mergedLRowUb` | 见 #1/#8 | — | 分母有界非零 ✅ |

  **关键机理**：`AmlaVecCompute`（`L724-819`）与 `CalcAmlaScaleFromMax`（`L551-574`）采用
  「**先取整、再还原小数部分**」的 Amla 重标定：`n = round(-M/LN2)`，`Exp` 的输入被显式构造成
  `(M/LN2 - n) * LN2 = frac(M/LN2) * LN2`，绝对值恒 `≤ LN2/2 ≈ 0.347`。
  **与 `M` 的绝对量级无关** —— 无论 `softmaxMaxUb` 多大多负，`Exp` 输出恒在
  `[e^-0.347, e^0.347] = [0.707, 1.414]`。因此 `L768/L770` 的往返 Cast **不会**在 A5 上因
  flush-to-zero 丢值，`L785/L789/L792` 的 `Div` 分母也不可能为 0，**不存在 `inf`/`NaN` 路径**。

- **★ 唯一潜在下溢点：`Stage2MergeAndCopyOut`（`L2386-2482`）的跨 `s2` 分块合并系数**
  `L2448 Max(mergedMUb, prevMUb, curMUb)` → `L2450/L2451 Sub(coefPrev/coefCur, M, mergedM)` →
  `L2453/L2454 Exp`。此处 `coef = M - mergedM ≤ 0` 且不受 `frac` 约束，可任意负（前一分块的 `max`
  远小于当前分块时）。A5 上 FP32 subnormal 若被 flush，`exp(coef)` 会变 0。
  **但无 NaN / 无除零**：`mergedM = max(prevM, curM)` 保证 `coefPrev`、`coefCur` 中**必有一个恰为 0**
  → 对应 `exp(0)=1`，该支的 `L` 被原样保留；`L2459 mergedLUb = prevL*e^coefPrev + curL*e^coefCur`
  恒 `> 0`，`L2476 RowDivs` 分母非零。被 flush 的那一支权重相对量级 `≤ 1.2e-38`，远低于最终
  FP16/BF16 输出的相对分辨率（`~1e-3`），对结果无可观测影响。
- **★ 输入是否可能包含 subnormal：是**

  1. **dtype 分析**：FP16 min normal = 6.10e-5、min subnormal = 5.96e-8；BF16 min normal = 1.18e-38、
     min subnormal ≈ 9.2e-41；FP32 min normal = 1.18e-38、min subnormal = 1.40e-45。
  2. **输入范围**：`softmaxMaxUb` 为累积最大 logit（可大范围取负）；`scale_value` 默认 1.0，
     调用方可传入任意正浮点。
  3. **中间计算**：Amla 重标定链的 `exp` 结果恒 `[0.707, 1.414]`，不产生 subnormal（见上表 #1–#3）；
     `L768/L770` 的往返 Cast 亦不会 flush。仅 `Stage2MergeAndCopyOut` 的 `exp(M-mergedM)` 可能下溢，
     但由 `max` 合并结构兜底，无除零。
  4. **常量分析**：见下表。
  5. **计算链**：`Div` 的除数（`cof`/`scale`/`mergedL`）由构造保证有界非零 → **有结构性下界**。

- **eps/常量 dtype 可表示性检查**

| 常量 | 值 | FP32（实际使用域） | FP16 | BF16 |
|---|---|---|---|---|
| `LN2` | 0.6931471805599453 | normal ✅ | normal ✅（3.3e-1 > 6.1e-5） | normal ✅ |
| `RECIP_OF_LN2` | 1.4426950408889634 | normal ✅ | normal ✅ | normal ✅ |
| `(T)(0.000001)`（`L803`） | 1e-6 | normal ✅（>> 1.18e-38） | **subnormal**（1e-6 < 6.1e-5）→ A5 flush 为 0 ⚠️ | normal ✅ |
| `(T)(-30.0)`（`L801`）、`(T)1.5`（`L799`） | -30 / 1.5 | normal ✅ | normal ✅ | normal ✅ |
| `FLOAT_ZERO` | 0 | exact ✅ | exact ✅ | exact ✅ |
| `SOFTMAX_MIN_NUM` | -2e38 | normal ✅（> -3.4028e38） | **不可表示 → -inf** ⚠️ | normal ✅ |
| `FLOAT_MAX` | 3.402823466e38 | normal ✅（FP32 最大值） | **不可表示 → +inf** ⚠️ | normal ✅ |
| `FLOAT_E_SCALAR` | 8388608（2^23） | 精确 ✅ | **不可表示 → +inf** ⚠️ | 精确 ✅ |

  说明：`T` 在本算子 vector 计算中恒为 `float`，故 `1e-6`/`SOFTMAX_MIN_NUM`/`FLOAT_MAX`/`FLOAT_E_SCALAR`
  的 FP16 不可表示性**当前不触发**；但若后续把该链下沉到 FP16 计算（或 `T=half` 实例化），
  `1e-6` 会 flush 为 0（`Div` 失去兜底）、`FLOAT_MAX/E_SCALAR` 会溢出为 inf。Stage 2 需保证 `T` 恒为 float。

- **选定适配策略：策略 0 —— 无需代码改动（已给出 §7.1 闭合证明）**

  - **不**给 `L785/L789/L792` 与 `L2357/L2441` 的 `Div` 分母加 `Maxs(denom, 1.17549435e-38F)` 钳位：
    分母已由构造落在 `[0.5, 2]`（`cof`）或 `[0.707, 1.414]`（`scale`）或恒 `> 0`（`mergedL`），
    钳位是死代码，反而会掩盖真实缺陷。
  - **不**给 `L766` 的 `Exp` 输入加 `Maxs(x, -87.0F)` 钳位：其输入域本身就是 `[-0.347, 0.347]`，
    加钳位等于引入永不生效的分支。
  - **不**消除 `L768/L770` 的 FP32→KV_T→FP32 往返 Cast：该 Cast 是 Amla 重标定的**精度设计**
    （把系数压到 KV_T 精度以便与 mm2 的 `KV_T` 输入复用同一缓冲区与 `RowMuls`），
    输出域 `[0.707, 1.414]` 远离 subnormal 区，无损失。
  - **Stage 4 必须覆盖的用例（唯一残留观测风险）**：
    ① `Stage2MergeAndCopyOut` 路径 —— 构造「同一 `s1` 行在相邻 `s2` 分块上 `max` 差值极大」
    （如首块 logit ≈ 0、次块 logit ≈ -1e4）以触发 `Exp(coef)` 下溢，断言输出**无 NaN / 无 Inf**，
    且与 CPU 参考的相对误差在容差内；
    ② `scale_value` 极小（`1e-8`）/ 极大（`1e4`）各一组，验证 Amla 的 `frac` 性质在极端 scale 下仍成立。

### 7.2 兼容性

- **是否使用 int4b_t 作为 Mmad 输入：否**（`common.h:101` 仅为类型分支；Mmad 输入为 `KV_T` = FP16/BF16）
- **是否使用 4:2 稀疏：否**

### 7.3 性能

- **TransDataTo5HD / vnchwconv：否**
- **高阶 ReduceSum（AR 模式）：否**（softmax max/sum 为 Block/Whole 级规约，非 AR）
- **BilinearInterpolation：否**
- **其它命中信号**：`DataCopyPad` 非对齐存储（L2 判定信号）；`FixpipeParamsV220` + 3 参 `Fixpipe` 在 A5
  仍受支持（`compressor/op_kernel/arch35/compressor_block_cube.h:318-331` 实证），A5 推荐写法为
  `FixpipeParamsC310<CO2Layout::ROW_MAJOR>`；L0C/L1 容量复核见 §9 条目 3（已通过，无需改）。
- **高风险 shape 清单（性能回归）**：

| # | 场景 | 触发原因 |
|---|---|---|
| 1 | `s1`（query 长度）= 1，decode 单 token | mm1/mm2 最小分块，`mBaseSize` 利用率最低 |
| 2 | `s2` 非 128 对齐（如 129、255、1023） | `nBufferMBaseSize=256` 尾块 + `DataCopyPad` 非对齐 |
| 3 | `topk` 取满（sparse_block_size=1、sparse_mode=3 的全稀疏） | `LoadData3DParamsV2` 逐块装载 + gather 索引密度最高 |
| 4 | `firstFill = true` 且 `missCount` 接近 `missCap` | `RunFirstFillScatterCopy` 全量 gather-scatter + `SyncAll<false>` |
| 5 | `preLoadNum=2` 边界：`s2LoopTimes` 恰为 2 / 3（`extraLoop=2` 排空） | 3 槽流水的首尾排空路径，同步计数最易失衡 |
| 6 | `gSize` 大（GQA 分组）导致 `bN2` 循环长 | mm2 与 Vec2 流水交错最频繁 |
| 7 | `usedCoreNum < blockDim` | 空闲核分支与同步参与集合边界 |
| 8 | `scale_value` 极小（如 1e-8）/ 极大（如 1e4） | `Exp` 输入域边界，subnormal/溢出 |

## 8. 同步协议盘点结果

**① 同步点清单**（`ConstInfo` 六个通道实际 flagId：`syncC2V1=4`、`syncV1NupdateC2=5`、`syncV0C1=6`、
`syncC1V1=7`、`syncV1C2=8`、`syncC2V2=9`；另有字面量 `3` 通道）

| # | 位置 | 原语 | 模式/pipe | flagId | 角色 |
|---|---|---|---|---|---|
| 1 | `kernel_mla.h:859` ComputeMm1 | Set | `<2, PIPE_FIX>` | 7 (C1V1) | AIC（mm1 回写完成） |
| 2 | `kernel_mla.h:873` ComputeMm2 | Wait | 裸 | 8 (V1C2) | AIC（等 AIV Vec1 释放 mm2 输入） |
| 3 | `kernel_mla.h:875` ComputeMm2 | Set | `<2, PIPE_FIX>` | 9 (C2V2) | AIC（mm2 结果就绪） |
| 4 | `kernel_mla.h:877` ComputeMm2 | Set | `<2, PIPE_FIX>` | 4 (C2V1) | AIC（mm2 结果可累加） |
| 5 | `kernel_mla.h:916` ProcessBalance 开头 | Set | `<2, PIPE_FIX>` | 4 (C2V1) | AIC（起始握手） |
| 6–9 | `kernel_mla.h:919-922` | Set ×4 | `<2, PIPE_MTE2>` | 3 | AIC（起始握手，V_TEMPLATE） |
| 10 | `kernel_mla.h:963` ProcessBalance 结尾 | Wait | 裸 | 4 (C2V1) | AIV（结束握手） |
| 11–14 | `kernel_mla.h:965-968` | Wait ×4 | 裸 | 3 | AIV（结束握手） |
| 15 | `kernel_mla.h:988` PreloadPipeline | Wait | 裸 | 6 (V0C1) | AIC（等 MergeKv 完成） |
| 16 | `kernel_mla.h:993` PreloadPipeline | Wait | 裸 | 3 | AIV（等 AIC 释放 mm2 输入） |
| 17 | `kernel_mla.h:995` PreloadPipeline | Set | `<2, PIPE_MTE3>` | 6 (V0C1) | AIV（MergeKv 完成） |
| 18 | `kernel_mla.h:1007` PreloadPipeline | Set | `<2, PIPE_MTE2>` | 3 | AIC（调度 AIV 下一拍） |
| 19 | `service_cube_mla.h:1055` ComputeMm2 尾部 | Wait | 裸 | 5 (V1NupdateC2) | AIC（等 nUpdate 写入） |
| 20 | `service_vector_mla.h:2151` ProcessVec1L | Wait | 裸 | 7 (C1V1) | AIV（等 mm1 结果） |
| 21 | `service_vector_mla.h:2154` ProcessVec1L | Set | `<2, PIPE_MTE3>` | 8 (V1C2) | AIV（Vec1 完成） |
| 22 | `service_vector_mla.h:2156` ProcessVec1L | Wait | 裸 | 4 (C2V1) | AIV（等 mm2 可累加） |
| 23 | `service_vector_mla.h:2160` ProcessVec1L | Set | `<2, PIPE_MTE3>` | 5 (V1NupdateC2) | AIV（nUpdate 写入，条件：`SInnerSize != 0`） |
| 24 | `service_vector_mla.h:2215` ProcessVec2L | Wait | 裸 | 9 (C2V2) | AIV（等 mm2 结果） |
| 25 | `fused_scatter_copy_sparse_flash_attention.cpp:174` | SyncAll | `isAIVOnly=false`（MIX） | 11（框架占用） | AIC+AIV 全核 |
| 26 | `kernel_mla.h:313` InitOutputSingleCore | SyncAll | `isAIVOnly=true`（默认） | 14（框架占用） | AIV 全核（`needInit && LAYOUT_T!=TND`） |

**② 参与集合假设**

- 全部 `flagId 4–9 / 3` 通道假设为「**AIC ↔ 组内全部 AIV**」（模式 2）：
  每个 AIC 的 Set 对应 2 个 AIV 各 1 次 Wait；2 个 AIV 各 1 次 Set 对应 AIC 1 次 Wait。
  代码中 AIC 侧 Set/Wait 与 AIV 侧 Wait/Set 均**未按 AIV 身份分支**（无 `blockIdx % 2` 区分），
  AIV 的 `GetBlockIdx() % 2` 只用于 `vecStartM/vecDealM` 数据切分（`L2146`, `L2211`），不参与同步编号 ✅
- 逐通道纸面计数（每个 `bN2/gS1/s2` 迭代）：

  | 通道 | AIC Set 次数 | AIV Wait 次数 | AIV Set 次数 | AIC Wait 次数 | 平衡 |
  |---|---|---|---|---|---|
  | 7 (C1V1) | 1 / nBuffer | — | — | — | 1:1（`#1`:`#20`） |
  | 8 (V1C2) | — | 1 / nBuffer（`#2`） | 1 / nBuffer（`#21`） | — | 1:1 |
  | 4 (C2V1) | 2 / nBuffer（`#4`,`#5`） | 2（`#10`,`#22`） | — | — | 2:2 |
  | 9 (C2V2) | 1 / nBuffer（`#3`） | 1（`#24`） | — | — | 1:1 |
  | 6 (V0C1) | — | 1 / 有效 info0（`#15`） | 1 / 有效 info0（`#17`） | — | 1:1 |
  | 5 (V1NupdateC2) | — | 1 / mm2 任务（`#19`） | 1 / `SInnerSize!=0`（`#23`） | — | 1:1（两侧同任务产出） |
  | 3 | 4（起始）+1/有效 info2（`#6–9`,`#18`） | — | — | 4（结尾）+1/有效 info0（`#11–14`,`#16`） | 有效任务在 3 槽轮转中各出现一次 → 总数相等 |
  | 11 / 14 | SyncAll | — | — | — | 框架内部平衡 |

- 空闲核：`Process()` 的 `aiCoreIdx < usedCoreNum` 门内才进入 `ProcessBalance()`。AIV 的
  `aiCoreIdx = blockIdx/2`、AIC 的 `aiCoreIdx = blockIdx` → 同一物理核整体进入/整体跳过 ✅
- `SyncAll<false>`（`#25`）位于 `if TILING_KEY_IS(1)` + `if firstFill` 下，`firstFill` 由 `missCounts`/
  `cacheTokens` 等**输入**决定，AIC/AIV 读到同一值 → 两侧都到达 ✅

**③ 目标平台语义对照（ASCEND950 / DAV_3510）**

- 模式 2 在 950 支持；kernel 为 `KERNEL_TYPE_MIX_AIC_1_2`，符合表 3 对模式 2 的要求 ✅
- 裸 `CrossCoreWaitFlag(flagId)`：`modeId` 默认 0，但模式 0/1/2 在 950 共用同一硬件指令
  （「省略 modeId 不影响接口功能」，`CrossCoreWaitFlag_ISASI.md` 950 分支）；`pipe` 默认 `PIPE_S` 在 950
  **生效**，阻塞标量流水（指令发射口）→ 严格程度 ≥ 2201 的「阻塞全部流水」，不会放松 ✅
  Stage 2 统一显式写为 `<2, PIPE_S>`，语义不变、可读性提升。
- flagId 占用核对（`key_features.md`）：未使用高阶 Matmul（低阶 `Mmad`）→ `[0,2N-1]` 不占用；
  未使用 KFC → 15/31 不占用；`SyncAll<false>`（`__mix__(1,2)`）在 950 占用 mode0/1/2 的 flagId 11
  （AIC）与模式 4 的 12/13/28/29 → 自定义 3–9 全部避开 ✅
  **风险提示**：`kernel_mla.h:110-112` 声明的 `SYNC_MM2RES_BUF1_FLAG=10/11`、`SYNC_FDOUTPUT_BUF_FLAG=12`
  为**未使用常量**；若后续启用，11/12 会与 950 `SyncAll<false>` 的内部占用冲突，禁止使用。
- `SyncAll<false>`（MIX）在 950 需 `__mix__(1,1)`/`__mix__(1,2)` ✅；`SyncAll()`（`isAIVOnly=true`）在 Mix
  kernel 中只完成 Vector 全核同步，与调用点（`if ASCEND_IS_AIV`）意图一致 ✅

**④ 处置决策**

| 同步点 | 结论 | 理由 |
|---|---|---|
| #1–#24（自定义 3–9 通道） | 沿用（24 个） | 集合粒度 = 文档粒度（含等号）；模式 2 在 950 支持且 kernel 类型匹配；参数语义已核对（裸 Wait 为最强约束，Stage 2 显式化为 `<2, PIPE_S>`）；参与集合恒定（无 AIV 身份分支、整核进入/跳过、计数表闭合）；flagId 3–9 避开所有文档占用区 |
| #25 | 沿用（1 个） | MIX 全核 SyncAll，AIC/AIV 同条件到达；占用 11 与自定义通道不冲突 |
| #26 | 沿用（1 个） | `isAIVOnly=true` 与调用核（AIV）一致；`needInit` 为 tiling 全局量，AIV 参与集合恒定 |

`[SYNC-AUDIT] 同步点 26 个：沿用 26 个 / 禁用 0 个 / 改用 0 个`

## 9. A5 迁移必改项与 Stage 2 落地状态

| # | 文件 | 问题 | Stage 2 状态 |
|---|---|---|---|
| 1 | `op_host/..._def.cpp` | 原仅注册 `ascend910_93` / `ascend910b` | **已完成**：`_def.cpp:67` 追加 `this->AICore().AddConfig("ascend950", config);` |
| 2 | `op_host/..._tiling_impl.cpp:1459-1462` | SoC 白名单原只允许 910B / 910_93 | **已完成**：`L1462` 追加 `socVersion_ != platform_ascendc::SocVersion::ASCEND950` |
| 3 | `op_kernel/.../service_cube_mla.h` L0C/L1 容量 | A5 容量与乒乓缓冲复核 | **已复核：通过，无需改**。`InitBuffers`（`L286-291`）中 `L1_BLOCK_SIZE = 64*(512+64)*sizeof(Q_T)` = 73728B（72KB）；`bufQPL1 = 4 块` = 288KB、`bufKVL1 = 3 块` = 216KB，合计 **504KB ≤ A5 L1 512KB**（未缩小）。`N_BUFFER_M_BASIC_SIZE=256`、`M_BASE_SIZE`、`S2` 基本块均为 tiling 侧常量，不依赖 L0C 容量；A5 L0C = 256KB ≥ A3 的 128KB |
| 4 | `op_kernel/.../service_vector_mla.h` 全部 `Div`/`Exp` | Subnormal 适配 | **策略 0：无需改**。见 §7.1 闭合证明（`Exp` 输入恒 `[-0.347, 0.347]`、`Div` 分母恒有界非零）；唯一可下溢点由 `max` 合并结构兜底，无 NaN |
| 5 | `csrc/build_aclnn.sh` | A5 分支需加入本算子 | **已完成**：`csrc/build_aclnn.sh:225` 在 `ascend950` 的 `CUSTOM_OPS_ARRAY` 中加入 `fused_scatter_copy_sparse_flash_attention` |
| 6 | 无 `arch22/arch35` 目录拆分 | 按「模式 A 就地宏隔离」适配 | **已落地（模式 A）**：`service_cube_mla.h` 的 4 个 L1→L0 装载点改为 `#if (__CCE_AICORE__ == 310)` 用 `LoadData2DParamsV2` / `#else` 保留原 3D+旧 2D，A3 分支零改动。见 §9.1 |
| 7 | `op_host/CMakeLists.txt` | 是否需加 arch35 目标 | **无需改**：已含 `--op_relocatable_kernel_binary=true`，无 board 条件分支；`op_kernel` 无独立 CMakeLists（include 由框架注入）。无需新增 `op_host/config/ascend950/` |

### 9.1 已落地的 kernel 侧装载改造（模式 A）

`LoadData3DParamsV2` / `LoadData2DParams(repeatTimes)` 在 A5 上必须改为 `LoadData2DParamsV2`
（`cube-migration-guide.md` 改动 2）。参数单位：`mStartPosition`/`mStep` = 16 元素；
`kStartPosition`/`kStep` = 32 字节；`srcStride`/`dstStride` = 512 字节（一个分形）。

| # | 函数（`service_cube_mla.h`） | `#if` 行 | A5 分支参数 |
|---|---|---|---|
| 1 | `LoadDataMm1A` | `L431-470` | `mStep=srcStride=dstStride=Align(mSize,16)/16`、`kStep=Align(kSize*sizeof(KV_T),32)/32`、`ifTranspose=false`（等价 `fa_base_matmul::LoadDataToL0A` 的 MK 分支） |
| 2 | `LoadDataMm1B` | `L480-500` | 以 `nSize` 为内维（NK 布局、非转置）：`mStep=srcStride=dstStride=Align(nSize,16)/16`、`kStep` 同上、`ifTranspose=false` |
| 3 | `ComputeMm2` ForB | `L1007-1046` | **KN 转置**：`ifTranspose=true`、`srcStride=Align(kL0SizeAlign,16)/16`、`dstStride=Align(nL1SizeAlign,16)/16`、`mStep=Align(kL0SizeAlign,16)/16`、`kStep=Align(nL1SizeAlign*sizeof(KV_T),32)/32` |
| 4 | `ComputeMm2` ForA | `L1049-1088` | **MK 非转置**：`ifTranspose=false`、`mStep=srcStride=dstStride=Align(mL1SizeAlign,16)/16`、`kStep=Align(kL0SizeAlign*sizeof(KV_T),32)/32`、源偏移 `aL1Tensor[kL0*baseK*mL1SizeAlign]` |

- **沿用不改（A5 已支持）**：
  - `FixpipeParamsV220` + 3 参 `Fixpipe`（mm1 回写 `L825-836`、mm2 回写 `L1122-1132`）——A5 arch35 实证见
    `csrc/attention/compressor/op_kernel/arch35/compressor_block_cube.h:318-331`。
  - 本算子**未调用** `SetFixpipeNz2ndFlag`，无需按迁移指南删除该全局设置。
- **待确证项（不声称 PASS）**：#3（ForB，KN 转置）的 `srcStride` 口径按 3D `l1H` 语义 1:1 映射为
  `Align(kL0SizeAlign,16)/16`，与 `csrc/attention/common/op_kernel/matmul.h` 中 `singleK` 形式在
  「按块偏移」场景下存在解释差异，**须以 Stage 3 编译 + Stage 4 精度结果确证**；代码中已加注释标记。
