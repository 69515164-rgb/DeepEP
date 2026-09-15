# 把集合通信 MILP 用于融合算子（MM+AllReduce）时的输入要求

集体综合器（TACCL / TE-CCL / SyCCL / OptCCL）的输入是：**拓扑 + 链路代价 + 静态需求**。输出是一张「谁在何时把哪块 chunk 发给谁」的 schedule。

MM+AllReduce 融合要出的不是一张独立 AllReduce 图，而是一张 **算-传联合 schedule**：哪个 MM tile 何时算完、哪块 AllReduce chunk 何时才能发、计算核 / HBM / CCU 如何并行。原求解器可以复用容量约束和流守恒，但输入必须从「静态需求」改成「带就绪时间的时变需求」，并补上计算与争用模型。缺这三类，求解器只会再出一张和 MM 脱节的通信图。

---

## 1. 原来集体 MILP 已经要的（必须仍给）

这些一项都不能省，语义与 TE-CCL / TACCL 相同。

| 输入 | 内容 | 不做融合时的用途 |
| --- | --- | --- |
| 拓扑 `G=(R,L)` | rank、链路、交换机超边、NIC/Port 共享 | 路由与容量 |
| 链路代价 | 每条边的 `alpha`（启动）、`beta`（每字节） | 传输时间 |
| 集体语义 | AllReduce、reduce-op（sum/max）、dtype、是否确定性规约 | 需求与正确性 |
| 分块 | chunk 数 `C`、chunk 字节 `B` | 离散变量规模 |
| 时间离散 | epoch 长度，或连续 start/send 时间 | 目标 makespan |
| 目标 | min 通信结束时间 | 最优性 |
| 可选剪枝 | 对称、sketch、最短路集合 | 可解性 |

AllReduce 的静态需求仍在：每个 rank 对每个 chunk 有一份 partial，每个 rank 最后都要有规约后的 chunk。融合后这份需求**还在**，只是 partial **不是 t=0 就存在**。

---

## 2. 融合后必须新增的三类输入

### 2.1 算子语义与并行切分（决定「减的是什么」）

求解器必须知道 MM 和 AllReduce 在张量上怎么咬合，不能只写「先 MM 再 AllReduce」。

必给：

```text
gemm:
  shape     M, N, K
  dtype / 累加精度
  layout    行主/列主、是否转置
  并行切分  哪一维被 rank 切开

parallel:
  模式      常见：行并行 Linear → 各 rank 算 Y_i = X_i W_i，AllReduce(Y)
            （列并行通常是 AllGather，不要误标成 AllReduce）
  world     P
  rank 映射 逻辑 rank → 物理 NPU / 超节点内外

reduce:
  被约简张量   Y[M, N]（或指定轴）
  reduce-op    与集体输入一致
  epilogue     bias / residual / 激活 在 AR 前还是后
```

没有「并行切分 + 被约简张量」，chunk 的源和汇会对错。行并行的 AllReduce 和列并行的 AllGather 需求矩阵不同，不能复用同一张集体实例。

### 2.2 tile ↔ chunk 对齐与就绪关系（决定「什么时候有货」）

这是相对原求解器**最关键的增量**。集体 MILP 假定所有 chunk 在 epoch 0 可发。融合后：

```text
ready(chunk c, rank r) >= finish(所有生产 c 的 MM tile)
send(c, r, *) 不得早于 ready(c, r)
```

必给：

| 输入 | 要求 |
| --- | --- |
| MM 输出 tile 网格 | `(BM, BN)`，必要时 `BK`（K 向分块影响何时写出一块 C） |
| 通信 chunk 网格 | 沿被约简张量的划分，与 2.1 一致 |
| **对齐约束** | 每个 chunk 必须是若干完整 MM tile 的并；禁止 tile 跨两个 chunk（否则要引入 reorder 变量，规模爆） |
| 生产映射 `produce` | `tile t @ rank r → chunk c`（一对一或多对一） |
| 写出语义 | tile 算完是否立刻对通信可见（寄存器/L2/HBM）；若要等存到 HBM，加 `store_delay` |
| 就绪粒度 | 按 tile 还是按 wave（同一拍多 tile 同时完成，FlashOverlap 那一层） |

对齐是输入合法性条件，不是求解后再修。对不齐就拒绝实例，或显式打开「通信前 reorder」并把它的代价写进输入。

两种用法，输入量不同：

| 模式 | 还要多给什么 | 求解器决定什么 |
| --- | --- | --- |
| A. 固定计算序 | 每个 `(tile, rank)` 的 `finish_time` 或 `release_time[c,r]` | 只排通信（原 MILP + 就绪约束） |
| B. 联合生成 | 每个 tile 的计算代价、核资源、允许的 tile 顺序偏序 | 同时排 MM 顺序和通信 |

「融合算子自动生成」走 B。A 只是把现成 GEMM 的完成时间喂给旧求解器，不是在生成融合算子。

### 2.3 计算资源与争用（决定「算和传能否真重叠」）

集体 MILP 只给链路容量。融合后计算和通信抢的是同一块芯片上的核、HBM、STARS。

必给：

```text
compute:
  单元        Cube / AIC 数，或可重叠的计算 mission 数
  tile 代价   T_mm(BM,BN,BK,dtype)  或  FLOP 与实测峰值
  并发        同一时刻最多几个 MM tile（wave 宽）

memory:
  HBM 带宽    读 A/B、写 C、SDMA/UB 拷贝 是否走同一组通道
  工作区上限  同时活着的 tile / 未发出 chunk 数
  双缓冲      有几档 C 缓冲可给通信用

overlap:
  计算与 CCU/SDMA 能否并行（950：STARS 上 Cube 与 CCU 可同拍）
  不能并行的资源集合（例如 950PR 上 HBM 1.6TB/s，L2 中继会打满）
  STARS 配额  留给通信的 CCU Mission / SDMA 条数（与 VT 数一致）
```

争用必须写成**容量约束**，不能只当注释。否则求解器会假设「算满速的同时链路也满速」，出的融合图在 950PR 上不可执行。

---

## 3. 正确性约束（作为输入开关，不是求解后再查）

这些要以可行域的形式喂进去：

1. **不早发**：`start_send(c)` ≥ `ready(c)`。
2. **流守恒 + 规约守恒**：每个 chunk 的各 rank partial 恰好被 reduce 一次（或按确定性顺序）。
3. **确定性**：若 `hcclDeterministic` 开，reduce 树/顺序必须是输入里固定的偏序，求解器不得换结合顺序。
4. **内存上界**：任意时刻 `live_tiles + in_flight_chunks` ≤ workspace。
5. **算法边可冻可解**：若只生成融合重叠、不换 AllReduce 算法（Ring/HD 已定），输入要带「通信 sketch / 固定对端序列」，求解器只决定每拍各 VT 的字节和与 MM 的重叠。若连通信图一起生成，sketch 可空，但规模按 SyCCL 方式切对称子问题。
6. **数值与 epilogue**：AR 前融合激活会改变 AllReduce 对象；必须在输入里钉死，禁止求解器自行把 epilogue 挪过 reduce。

---

## 4. 目标函数要改

原目标：`min T_comm`。

融合目标：

```text
min T_end
T_end >= 每个 rank 上最后一个 MM tile 的完成时间
T_end >= 每个 rank 上 AllReduce 结果可用时间
```

只最小化通信 makespan 会把 MM 排成「先算完再发」，融合退化成两阶段。输入里必须声明目标是 **fused makespan**，并给计算时间变量（模式 B）或固定 release（模式 A）。

可选次目标：最大 HBM 峰值、STARS 占用、确定性优先。用字典序或加权，权重也是输入。

---

## 5. 喂给求解器的实例合同（一份就能解）

```text
instance = {
  /* 旧集体输入 */
  topology, alpha_beta, collective=AllReduce, chunk{C,B}, epoch,
  symmetry | sketch,

  /* 新增：语义 */
  gemm{M,N,K,dtype,layout},
  parallel{pattern=row_tp, P, rank_map},
  reduce{tensor=Y, axis, op, epilogue_after_ar},

  /* 新增：对齐与就绪 */
  mm_tile{BM,BN,BK},
  produce[tile,rank] -> chunk,          # 必须满射到所有 chunk
  align = "tile 不跨 chunk",
  ready_model = "tile_store_to_hbm" | "wave",

  /* 新增：计算与争用（模式 B） */
  T_mm[tile],
  compute_slots,                         # 同时可跑的 tile 数
  hbm_cap, workspace_cap,
  comm_slots,                            # 同时可跑的 VT / CCU Mission
  exclusive_resources[],                 # 互斥集合

  /* 目标 */
  objective = fused_makespan,
  deterministic = bool
}
```

合法性检查（求解前拒掉）：

- `pattern` 与 `collective` 匹配（行并行 ↔ AllReduce，列并行 ↔ AllGather）。
- 每个 chunk 的字节 = 其生产 tile 的输出字节之和。
- `C` 能被 `P` 和 tile 网格整除（Ring/HD 的分 chunk 习惯）。
- `T_mm`、`alpha_beta` 用同一时间单位；整数化前放大，避免 TACCL 那种截断成 0。
- `comm_slots` ≤ 硬件上限（950：CCU 32、Jetty 64、SDMA 32，还要给计算留配额）。
- 模式 A 必须带齐 `release_time[c,r]`；模式 B 必须带齐 `T_mm` 和 `compute_slots`。两者都缺则无解。

---

## 6. 对原求解器编码的最小改动（输入侧能对上什么）

| 原约束 | 融合后 |
| --- | --- |
| chunk 在 t=0 可发 | 加 `release[c,r]`，来自 2.2 |
| 链路容量 / epoch | 仍在；epoch 要覆盖计算时间轴，不能只覆盖通信 |
| 流守恒 / 复制 / reduce | 仍在；reduce 节点不得早于各输入 partial 的到达 **且** 不得早于本地 tile ready |
| 对称 | 仍可用；计算序若也对称（各 rank 同一 tile 顺序），SyCCL 切分仍然成立 |
| 输出 MSCCL-EF | 不够；还要输出 tile 序、tile→chunk 就绪边、STARS 上计算 mission 与 CCU mission 的边 |

若坚持不改求解器内核：只能走模式 A，把 MM 当成外部预言机，输入多一张 `release_time` 表。那是「通信图叠在固定 GEMM 上」，不是融合算子生成。

---

## 7. 规模上对输入的要求

联合 MILP 的变量大约是

```text
O( tiles * ranks * compute_slots  +  chunks * links * epochs )
```

所以输入必须已经做过粗化：

- tile 不要落到 MMA 指令；`BM,BN` 至少与通信 chunk 同级（建议 chunk ≥ 一个输出 tile）。
- epoch 长度取 `max(T_mm_min, alpha)` 这一档，不要按时钟周期。
- 用对称把 `P` 很大的超节点收成一个轨道（SyCCL 那套），计算序在轨道内复制。
- 950 上通信 VT 数先固定成输入（8 条 UB / 4 条 UBoE），不要让求解器枚举 Jetty 绑定。

粗化也是输入：`granularity` 必须由调用方给定，求解器不负责从 kernel 指令往上抬。

---

## 8. 950 上多给的硬件输入

| 输入 | 为什么 |
| --- | --- |
| 展开模式 `AICPU_TS` / `CCU_SCHED` | 通信拍边界是否暴露、stall/就绪信号在哪一层 |
| Cube 与 CCU 可否同拍 | 决定 overlap 约束是「真并行」还是「只是排队」 |
| HBM 带宽档（PR 1.6 / DT 4 TB/s） | 争用容量 |
| 禁止 HBM 中继（PR） | 通信 L2 只能走 IO Die，写进 `exclusive_resources` |
| STARS 剩余配额 | 融合后计算 mission 已占一部分 |

这些不进输入，求解器会按 NVLink+NCCL 的假设出图，在 950 上不可执行。
