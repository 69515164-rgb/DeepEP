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
计算编译器给定：tiling + 计算流图
               （produce、ready、占用时间轴都从这两份推，不再单独要）
MILP 只决定：  各 chunk 何时走哪条链路 / 哪个 VT，且不得早于 ready
```

不是让 MILP 同时发明一种 GEMM 实现和一张 AllReduce 图。

---

## 1. 推荐架构（简洁版）

```text
┌──────────────────────────────┐
│ 计算编译器只给两份              │
│  1. tiling 策略               │
│  2. 计算流图（tile 依赖 + 耗时） │
└──────────────┬───────────────┘
               │ 机械推导
               │   produce、ready、占用时间轴
               ▼
┌──────────────────────────────┐
│ MILP 只编通信                  │
│  原集体约束 + send ≥ ready     │
└──────────────────────────────┘
```

`produce` 和占用时间轴**不是**编译器再交的第三、第四份产物。上一版把它们写在输入清单里，容易理解成还要单独导出一张映射表和一条 occupancy trace。只要 tiling 和流图给全了，这两样是函数，不是决策。

求解器**可以**决定的：ready 之后 chunk 走哪条边、哪个 epoch、哪个 VT。

求解器**不得**决定的：`BM,BN,BK`、tile 网格、K 向流水、tile 序、epilogue、Cube 指令级调度。

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

## 3. tiling 和计算流图各给什么（够不够推出 produce / 占用轴）

**够。** 编译器只需这两份。`produce` 是 tiling 上的分块函数；占用时间轴是流图按时间展开后的资源占用。不要让编译器再导一张映射表、一条 occupancy trace——那是重复信息，还容易和 tiling 打架。

只多两个**标量约定**（不是图）：通信把几块 C tile 收成一个 chunk（`tiles_per_chunk`），以及 C 写到通信可见层要多久（`store_delay`）。前者是集体分块习惯，后者是存储层次常数。

### 3.1 tiling 策略给什么

描述**空间怎么切**，不描述谁先算。对象是张量上的网格，不是时间轴。

```text
tiling:
  问题形状     M, N, K, dtype, layout
  输出块       BM × BN          → C 被切成 ceil(M/BM) × ceil(N/BN) 个 C-tile
  K 向块       BK               → 每个 C-tile 要 ceil(K/BK) 次 MMA 才写完
  迭代空间     循环嵌套顺序      例如：N-tile 外、M-tile 中、K 内
  流水         wave / pipeline 级数（同一拍几个 C-tile 在 Cube 上）
  可见性       C-tile 完成 K 归约后，经过哪一级才对通信可见
               store_delay = 寄存器→L2→HBM 的固定延迟
```

它回答的问题：

| 问题 | tiling 给的答案 |
| --- | --- |
| 一块可发给 AllReduce 的数据最小是什么 | 一个写完的 C-tile（BM×BN） |
| 有多少块、每块多大 | 网格 `Tm × Tn`，每块 `BM*BN*sizeof` |
| 一块要算多久才「算完」 | 由 BK、K、流水级数决定（流图用这个当节点粒度） |
| chunk 允许怎么切 | 只能沿 C-tile 网格做**整块合并**，不能横切一块 C-tile |

**tiling 不够单独当通信输入**：它没有「第 3 个 C-tile 何时算完」，也没有「那时 Cube 占着没有」。时间在流图里。

### 3.2 计算流图给什么

描述**这些 tile 按什么依赖、花多长时间算**。节点必须就是 3.1 里的 C-tile（或 wave = 同一拍的几个 C-tile），不能另起一套名字。

```text
compute_dag:
  nodes   每个 C-tile（或 wave）@ rank
  edges   依赖：同一 C-tile 的 K 片必须按序；
          issue 序（kernel 已排好的总序，或 wave 级偏序）
  dur[t]  该节点占用 Cube 的时长（由 BM,BN,BK 和硬件模型算出，
          编译器给数即可，MILP 不当变量）
```

它必须已经是**排好序的执行计划**，不是「还有自由度的 DAG」。若只有偏序、没有 issue 序和 `dur`，就推不出占用时间轴，那份流图不合格。

排好之后，对每个节点：

```text
start[t]  = max{ start[pred]+dur[pred] }   # 沿 issue 序
finish[t] = start[t] + dur[t]
```

这就是计算时间轴。通信只读 `finish`。

**流图不够单独当通信输入**：节点是 C-tile，AllReduce 的 chunk 可能是 4 个 C-tile 拼的。谁拼给谁，要靠 tiling 网格 + `tiles_per_chunk`，不是靠依赖边。

### 3.3 从这两份推 produce 和占用轴

**produce**（C-tile → 通信 chunk）：

```text
约定：沿被约简张量的切分轴，每连续 tiles_per_chunk 个 C-tile 合成一个 chunk
      （行并行 AllReduce 通常沿 N 或沿 rank 内的输出行）

produce[t] = floor( tile_index(t) / tiles_per_chunk )
ready[c]   = max{ finish[t] + store_delay | produce[t]=c }
```

`tiles_per_chunk=1` 时 chunk ≡ C-tile，produce 是恒等。这是最简对齐。只要 tiling 的 C 网格和集体 chunk 数能整除，produce 就是这个函数，不必编译器再吐一张表。

**占用时间轴**（何时占了哪些计算资源）：

```text
每个节点带资源向量 u[t] = (Cube, HBM_rw, STARS_compute, ...)
         —— 同类 tile 相同，是硬件表，不是每实例一张图

occupancy(τ) = sum{ u[t] | start[t] ≤ τ < finish[t] }
comm_residual(τ) = 硬件上限 − occupancy(τ)
```

流图有 `start/finish` 和每类节点的 `u`，占用轴就是扫描线。不必编译器再导 occupancy.trace。

```text
tiling  ─────────┐
  空间网格        ├── produce、chunk 字节、tiles_per_chunk 合法性
计算流图 ────────┤
  finish[t]       ├── ready[c]
  start/finish+u  └── occupancy(τ) / 通信剩余容量
```

还缺的只是两个标量：`tiles_per_chunk`、`store_delay`。再加一张很小的**资源常量表** `u`（Cube 一拍占几条、HBM 一 tile 读多少），对所有 MM+AR 实例共用。

### 3.4 算子语义（校验，不搜）

```text
parallel: row_tp → AllReduce(Y)；列并行是 AllGather，勿混
epilogue: 钉死在 AR 前或后
```

形状已在 tiling 里。这里只查模式和集体是否匹配。

---

## 4. 求解器还要的通信侧增量

占用轴推出之后，通信侧用的是**剩余容量**（推导结果，仍不必手填一条 trace）：

```text
comm_slots(τ)  硬件 VT/CCU 上限 − occupancy(τ).stars
hbm_rest(τ)    HBM 上限 − occupancy(τ).hbm
exclusive[]    SKU 常量（950PR 禁止 HBM 中继）
workspace      来自 tiling 的 C 双缓冲档数
```

不要让求解器假设「算满速时链路也满速」。剩余容量随 `τ` 变，来自流图，不是再搜。

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
  sketch = Ring | HD | ...,

  /* 编译器只给这两份 */
  tiling{BM,BN,BK,wave,loop_order,store_delay},
  compute_dag{nodes=C-tiles, issue_order, dur[t]},

  /* 标量约定 + 硬件表，不是图 */
  tiles_per_chunk,                   # 默认 1：chunk ≡ C-tile
  u = (Cube, HBM, STARS_compute),    # 每类 tile 的资源向量，实例间共用
  exclusive[],                       # SKU 常量

  /* 下面全部推导，禁止手填另一份 */
  # produce, chunk{C,B}, ready, occupancy, comm_slots(τ), hbm_rest(τ)

  parallel{row_tp,P,rank_map}, epilogue_after_ar,
  objective = fused_makespan,
  deterministic = bool
}
```

求解前拒掉：

- 流图节点不是 tiling 的 C-tile / wave（两套网格对不上）。
- 流图没有 issue 序或没有 `dur`（推不出时间轴）。
- `tiles_per_chunk` 不能整除 C-tile 数。
- `pattern` 与集体不匹配。
- `dur` 与 `alpha/beta` 时间单位不一致。
- 缺 tiling 或缺流图：直接拒，不要回退去搜 BM/BN。

---

## 7. 对原求解器的改动（因此才简洁）

| 原约束 | 默认融合扩展 |
| --- | --- |
| chunk 在 t=0 可发 | 换成推导出的 `ready[c]` |
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

## 9. 950 上写进资源表 `u` / `exclusive` 的量

| 输入 | 写在哪 |
| --- | --- |
| 展开模式 AICPU_TS / CCU_SCHED | 通信拍边界；ready 信号在哪一层 |
| Cube 与 CCU 同拍与否 | occupancy 里计算与通信是否互斥 |
| HBM 档 PR 1.6 / DT 4 | `hbm_rest` |
| 禁止 HBM 中继（PR） | `exclusive` |
| 计算已占的 STARS / CCU / SDMA | `comm_slots` 的上限 |

这些由 kernel 实现和 SKU 决定，随计算流图一起给，不搜。
