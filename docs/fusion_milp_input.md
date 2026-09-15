# 把集合通信 MILP 用于融合算子（MM+AllReduce）时的输入要求

**默认分工：求解器不做 tiling，也不生成计算流图。** tiling 策略和 MM 的计算流图由计算编译器 / 算子实现给定；MILP 只在「货何时就绪」的约束下做通信编排。这比联合搜索算-传更干净，也才是 TACCL / TE-CCL / SyCCL 这类求解器能扩的范围。

集体综合器原来的输入是：**拓扑 + 链路代价 + 静态需求**。输出是「谁在何时把哪块 chunk 发给谁」。融合后静态需求变成**带就绪时间的时变需求**；就绪时间从给定的计算流图推出来，不是求解器搜出来的。

---

## 0. 为什么求解器不能做 tiling

Tiling 是另一类搜索：Cube/MMA 形状、L1/L0、K 向流水、ping-pong、bank 冲突、累加精度、epilogue 融合。这是 CUTLASS / Triton / TVM / AscendC 的问题，不是链路容量 MILP 的问题。

硬原因：

1. **变量空间不对。** 集体 MILP 的变量是 `send(chunk, link, epoch)`。tiling 的变量是 `BM,BN,BK,wave,pipeline`。塞进同一个 ILP，规模是通信变量乘上 tile 网格，SyCCL 那套对称切分也救不了。
2. **代价模型不对。** `alpha/beta` 描述链路；tile 时间取决于 Cube 吞吐、HBM 行缓冲、L0 复用。没有一份可信的 `T_mm(BM,BN,BK)` 曲面，求解器选出的「最优 tiling」在 950 上无意义。
3. **正确性在计算侧。** 数值稳定性、确定性累加、epilogue 是否越过 reduce，都由 kernel 图钉死。求解器若改 tiling，等于改 GEMM 实现。
4. **已有求解器从未搜过 GEMM。** TACCL/TE-CCL/SyCCL/OptCCL 的 chunk 是输入粒度。融合扩展应保持这一点：chunk 仍是输入，只是每个 chunk 多了一个 `ready` 时间。

所以「融合算子自动生成」在本方案里的含义是：

```text
计算编译器给定：tiling + MM 计算流图 + tile→chunk 生产关系 + 计算占用
MILP 只决定：  各 chunk 何时走哪条链路 / 哪个 VT，且不得早于 ready
```

不是让 MILP 同时发明一种 GEMM 实现和一张 AllReduce 图。

---

## 1. 推荐架构（简洁版）

```text
┌─────────────────────────────┐
│ 计算侧（给定，不进搜索）      │
│  tiling 策略                 │
│  MM 计算流图 / tile 序        │
│  produce(tile → chunk)       │
│  每 tile 完成时间或 T_mm      │
│  计算已占用的 Cube / HBM / STARS │
└──────────────┬──────────────┘
               │ 推导 release[c,r]
               ▼
┌─────────────────────────────┐
│ MILP（只编通信）              │
│  原集体约束 + 不早发          │
│  剩余链路 / VT / SDMA 容量    │
│  输出：通信 schedule          │
└─────────────────────────────┘
```

求解器**可以**决定的：在给定就绪之后，chunk 走哪条边、哪个 epoch、哪个 VT；通信与已占用计算时间轴如何错开。

求解器**不得**决定的：`BM,BN,BK`、tile 网格、K 向流水、tile 计算顺序、epilogue 位置、Cube 指令级调度。

连通信算法也可以冻：输入带 Ring/HD sketch 时，求解器只排每拍字节和与 MM 的重叠，不换邻居。这是最简可行域。

---

## 2. 原来集体 MILP 已经要的（必须仍给）

| 输入 | 内容 | 用途 |
| --- | --- | --- |
| 拓扑 `G=(R,L)` | rank、链路、交换机超边、NIC/Port 共享 | 路由与容量 |
| 链路代价 | 每条边 `alpha`、`beta` | 传输时间 |
| 集体语义 | AllReduce、reduce-op、dtype、确定性 | 需求与正确性 |
| 分块 | chunk 数 `C`、字节 `B`（与 tiling 对齐后给出） | 离散变量 |
| 时间离散 | epoch 长度 | 目标轴 |
| sketch / 对称 | 固定对端序列、轨道切分 | 可解性 |

AllReduce 需求仍在：每 rank 每 chunk 一份 partial，最后每人都有规约结果。partial 的出现时间由计算流图给出。

---

## 3. 计算侧必须给定的输入（不搜索）

### 3.1 tiling 策略

由计算编译器一次选定，MILP 当常量。

```text
tiling:
  BM, BN, BK
  wave / pipeline 级数
  输出 tile 网格（沿 M、N）
  写出路径  寄存器 → L2 → HBM，store_delay
```

通信 chunk 网格必须是这套 tiling 的**导出量**：每个 chunk = 若干完整输出 tile 的并。禁止 tile 跨两个 chunk。对不齐则拒实例，不要在 MILP 里加 reorder 变量。

### 3.2 计算流图

一张有向图，节点是 MM tile（或 wave），边是依赖。

```text
compute_dag:
  nodes    tile t @ rank r
  edges    t1 → t2  的偏序（通常已是总序：kernel 的 issue 序）
  T_mm[t]  或 每个节点的绝对 finish_time（相对 kernel 起点）
  occupancy[t]  该节点占用的 Cube / HBM / STARS 配额
```

流图是输入，不是决策变量。求解器不得重排 tile 去「让某条链路更早有货」——那是在改 GEMM。

由流图机械推出通信就绪：

```text
produce[t, r] → c
ready[c, r] = max{ finish(t) + store_delay | produce(t,r)=c }
send(c, r, *) ≥ ready[c, r]
```

`produce` 必须满射到所有 chunk。这张表跟 tiling 一起由计算侧给出。

### 3.3 算子语义（校验用，不搜）

```text
gemm:     M,N,K / dtype / layout
parallel: row_tp → AllReduce(Y)；列并行是 AllGather，勿混
reduce:   张量、轴、op
epilogue: 钉死在 AR 前或后
```

用来检查 `pattern` 与集体是否匹配、chunk 字节是否等于生产 tile 之和，不进入搜索。

---

## 4. 求解器还要的通信侧增量

计算占用已经从流图来了，通信侧只需**剩余容量**：

```text
comm_slots     同时可跑的 VT / CCU Mission（≤ 硬件上限 − 计算已占）
hbm_rest       计算占用之后剩给 SDMA/UB 的 HBM
exclusive[]    与计算互斥的资源（950PR 禁止 HBM 中继）
workspace      未发出 chunk 的缓冲档数（双缓冲是 tiling 给定的，这里只读档数）
```

不要让求解器再估「算满速时链路也满速」。占用时间轴是输入。

---

## 5. 正确性（可行域开关）

1. 不早发：`start_send(c) ≥ ready(c)`，`ready` 只读计算流图。
2. 流守恒 + 规约守恒；确定性开则 reduce 顺序是输入偏序。
3. 任意时刻 in-flight chunk ≤ workspace（计算 live tile 已由流图固定，不必再搜）。
4. 默认冻通信 sketch（Ring/HD 对端序列）。解开 sketch 等于退回「纯集体综合」，规模按 SyCCL 切，仍不搜 tiling。
5. epilogue 位置只读。

目标：

```text
min T_end
T_end >= 计算流图的 sink 完成时间      # 常量下界
T_end >= 每个 rank 上 AR 结果可用时间  # 决策变量
```

计算 sink 是常量，求解器真正动的是通信完成时间。只要 ready 允许，通信会尽量与计算尾部重叠；不会去改 tile 序来制造更早的 ready。

---

## 6. 实例合同（默认：只编通信）

```text
instance = {
  /* 旧集体 */
  topology, alpha_beta, collective=AllReduce, epoch,
  sketch = Ring | HD | ...,          # 默认冻算法边

  /* 计算侧给定，不搜索 */
  tiling{BM,BN,BK,wave,store_delay},
  compute_dag{nodes, edges, finish_time | T_mm, occupancy},
  produce[tile,rank] -> chunk,
  chunk{C,B},                        # 由 tiling 导出
  gemm{M,N,K,dtype}, parallel{row_tp,P,rank_map},
  epilogue_after_ar,

  /* 通信剩余资源 */
  comm_slots, hbm_rest, exclusive_resources[], workspace,

  objective = fused_makespan,        # 实质是 min 通信完成，下界钉在 dag.sink
  deterministic = bool
}
```

求解前拒掉：

- tiling 与 chunk 不对齐（tile 跨 chunk，或字节对不上）。
- `produce` 未覆盖全部 chunk。
- `pattern` 与集体不匹配。
- `finish_time` / `T_mm` 与 `alpha/beta` 时间单位不一致。
- `comm_slots` 超过剩余硬件配额。
- 缺计算流图或缺 tiling：直接拒，不要回退去搜 BM/BN。

---

## 7. 对原求解器的改动（因此才简洁）

| 原约束 | 默认融合扩展 |
| --- | --- |
| chunk 在 t=0 可发 | 换成输入表 `ready[c,r]` |
| 链路容量 / epoch | 仍在；epoch 轴要覆盖 dag 时间，计算占用当已用容量 |
| 流守恒 / reduce | 仍在；reduce 不得早于 partial 到达且不得早于本地 ready |
| 对称 | 计算序已对称则可继续切轨道 |
| 输出 | 通信 schedule + 与 dag 节点的就绪边；**不输出新的 tiling** |

内核几乎只加一类约束：`send ≥ ready`。这就是「比较简洁」的来源。

---

## 8. 明确不做（除非另开课题）

| 不做 | 原因 |
| --- | --- |
| 搜 `BM,BN,BK` | 计算编译器的事 |
| 重排 tile 序 | 改的是 GEMM，不是通信 |
| 把 epilogue 挪过 reduce | 改变数值语义 |
| 枚举 Jetty 绑定 | VT 数当输入常量 |
| 联合 MILP 同时出 kernel 与通信图 | 变量乘积不可解 |

若以后要「换一种 tiling 再编一次通信」，做法是**外层枚举少量 tiling 候选**（计算编译器给出 3～5 套），每套跑一次本 MILP，比较 fused makespan。tiling 仍不进求解器内部。

---

## 9. 950 上计算侧还要写进 occupancy 的量

| 输入 | 写在哪 |
| --- | --- |
| 展开模式 AICPU_TS / CCU_SCHED | 通信拍边界；ready 信号在哪一层 |
| Cube 与 CCU 同拍与否 | occupancy 里计算与通信是否互斥 |
| HBM 档 PR 1.6 / DT 4 | `hbm_rest` |
| 禁止 HBM 中继（PR） | `exclusive` |
| 计算已占的 STARS / CCU / SDMA | `comm_slots` 的上限 |

这些由 kernel 实现和 SKU 决定，随计算流图一起给，不搜。
