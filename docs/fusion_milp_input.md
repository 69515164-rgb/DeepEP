# 把集合通信 MILP 用于融合算子（MM+AllReduce）时的输入要求

**默认分工：求解器不做 tiling，也不生成计算流图。** tiling 策略和 MM 的计算流图由计算编译器 / 算子实现给定；MILP 只在「货何时就绪」的约束下做通信编排。

领导汇报稿：`docs/融合算子自动生成_MM_AllReduce.pptx`（背景名词已在页内标注）。
`dur` 怎么来的手算实例见本文第 10 节（2×2 C-tile，每块 10 μs）。
详细合同：本文。

集体综合器原来的输入是：**拓扑 + 链路代价 + 静态需求**。输出应理解成「谁把哪块 chunk 发给谁、由什么事件触发」，不是墙上时钟的闹钟。融合后多出来的是**计算→通信的先后依赖**；离线求解若要比较「几种打包/几条 VT 谁更快」，才用流图估一个 `ready`，运行时不按这个数启动。

运行时正确性与估时的分工见第 11 节（抖动、访存争抢下不需要、也不该给出通信启动时刻）。
950 上「Y 不落 HBM、片上直接发」见第 12 节。

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
               （produce 边必须有；dur / ready 只给离线搜索用）
MILP 只决定：  各 chunk 走哪条链路 / 哪个 VT、由哪个计算完成事件触发
运行时：       等事件，不按墙上时钟发
```

不是让 MILP 同时发明一种 GEMM 实现和一张 AllReduce 图。

---

## 1. 推荐架构（简洁版）

```text
┌──────────────────────────────┐
│ 计算编译器                     │
│  1. tiling                    │
│  2. 计算流图 = 节点 + 发射序     │
│     （dur 可选，只为离线选型）    │
└──────────────┬───────────────┘
               │ produce 边（必须）
               │ ready 估时（仅当还要搜打包/VT）
               ▼
┌──────────────┐     下降      ┌──────────────────────────┐
│ MILP 搜结构   │  ─────────►  │ 运行时：等 C-tile 完成事件  │
│ 不搜切块      │   事件边      │ 再 post WQE；无启动时刻表   │
└──────────────┘              └──────────────────────────┘
```

`produce` 和占用时间轴**不是**编译器再交的第三、第四份产物。上一版把它们写在输入清单里，容易理解成还要单独导出一张映射表和一条 occupancy trace。只要 tiling 和流图给全了，这两样是函数，不是决策。

求解器**可以**决定的：chunk 走哪条边、哪个 VT、由哪个计算完成事件触发（以及要不要等几块再打一包）。epoch / `ready` 是离线搜索坐标，下降后变成事件边，不是内核里的 `sleep`。

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

只多两个**标量约定**（不是图）：通信把几块 C tile 收成一个 chunk（`tiles_per_chunk`），以及规划用的写出延迟（`store_delay`）。前者是集体分块习惯；后者只进离线屋顶线，运行时改成「写完成 / fence」事件，不是等 2 μs。

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

描述**这些 tile 的先后依赖**。节点必须就是 3.1 里的 C-tile（或 wave = 同一拍的几个 C-tile），不能另起一套名字。

```text
compute_dag:
  nodes   每个 C-tile（或 wave）@ rank          # 运行时必须有
  edges   produce：写完这些节点 → 允许发对应 chunk  # 运行时必须有
          issue 序（kernel 已排好，不搜）
  dur[t]  估时，可选。只给离线 MILP 比较结构用
          运行时内核不读这个数
```

**运行时合格**：有节点、有 produce 边、有 issue 序。没有 `dur` 也能生成「写完就发」的融合算子。

**离线 MILP 合格**：还要 `dur`，才能估 `ready`、比较「几块打一包 / 几条 VT」谁的 fused makespan 更短。若只有偏序、没有 issue 序和 `dur`，推不出规划用时间轴，求解器拒收——但那是选型问题，不是运行时正确性问题。

规划用展开（仅离线）：

```text
start[t]  = max{ start[pred]+dur[pred] }   # 沿 issue 序
finish[t] = start[t] + dur[t]
```

通信求解器读的是估出来的 `finish`；运行时读的是 Cube 完成事件。

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

运行时（硬约束，不依赖估时）：

1. 不早发：chunk `c` 的 WQE 不得在 `produce^{-1}(c)` 写完成之前 post。这是事件边，不是 `start_send ≥ 12μs`。
2. 流守恒 + 规约守恒；确定性开则 reduce 顺序是输入偏序。
3. in-flight chunk ≤ workspace。
4. 默认冻通信 sketch（Ring/HD 对端序列）。
5. epilogue 位置只读。

离线 MILP（软约束，用估时搜结构）：

```text
规划：  send_epoch(c) ≥ ready_est(c)
下降：  wait(finish_event of produce^{-1}(c)); post_wqe(c, link, VT)
目标：  min 估出来的 fused makespan，用来比结构，不写进内核
```

计算 sink 的估时只是规划下界。求解器真正选的是：打包粒度、VT、链路。选出之后丢掉微秒数，只留下事件边。

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
- 运行时路径：流图没有 produce 边或没有 issue 序。
- 离线 MILP 路径：还缺 `dur`（推不出规划用 ready）。「写完就发 + 冻 Ring」不走这条，可以没有 `dur`。
- `tiles_per_chunk` 不能整除 C-tile 数。
- `pattern` 与集体不匹配。
- 缺 tiling 或缺流图：直接拒，不要回退去搜 BM/BN。
- 输出里出现墙上时钟启动点（`sleep` / 定时器 post）：视为错误下降，拒。

---

## 7. 对原求解器的改动（因此才简洁）

| 原约束 | 默认融合扩展 |
| --- | --- |
| chunk 在 t=0 可发 | 换成 produce 边；离线可再加 `send_epoch ≥ ready_est` |
| 链路容量 / epoch | 仅离线；epoch 是搜索坐标，不进运行时 |
| 流守恒 / reduce | 仍在；reduce 等远端到达 **且** 等本地 produce 事件 |
| 对称 | 计算序已对称则可继续切轨道 |
| 输出 | **事件边 + 链路/VT/打包**；不输出启动时刻，不输出新 tiling |

内核几乎只加一类约束：通信 WQE 挂在对应 C-tile 的完成事件上。离线多出来的 `send ≥ ready_est` 只用来选结构。

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
| 展开模式 AICPU_TS / CCU_SCHED | 通信拍边界；ready 对应流图哪一层 finish |
| Cube 与 CCU 同拍与否 | `u` 里 STARS 是否与通信互斥 |
| HBM 档 PR 1.6 / DT 4 | 硬件上限，占用轴展开后得 `hbm_rest(τ)` |
| 禁止 HBM 中继（PR） | `exclusive` |
| 每类 C-tile 占的 Cube / HBM / STARS | 资源表 `u`，不是 occupancy trace |

这些是 SKU / kernel 常数，随 tiling 走，不搜、不另给时间轴。

---

## 10. 实例：`dur` 从哪来，流图怎么变成完成时刻

**先说结论：计算流图不会“算出”耗时。**  
`dur[t]` 是计算编译器事先填在节点上的数。流图只做第二步：按已经排好的发射顺序，把这些数串成 `start / finish`。没有代价模型，空流图给不出任何微秒数。

```text
第 1 步  代价模型（编译器，不进 MILP）
         tiling 的 BM,BN,BK,K  +  芯片峰值/带宽
         → 每个 C-tile 一个 dur

第 2 步  流图展开（机械，不搜）
         issue 序 + dur[t]
         → start[t], finish[t]

第 3 步  给通信用
         ready[c] = max finish[t] + store_delay
```

下面用一套**为手算取整**的数字。峰值不是某款芯片手册值，只保证算术能对上；真实 950 换一张 `Peak / BW` 表即可，步骤不变。

### 10.1 问题与切块（tiling）

本 rank 上的局部矩阵乘（行并行切过 K 之后）：

```text
Y_partial = X @ W      FP16
M = 256,  N = 256,  K = 1024

BM = 128, BN = 128, BK = 256
C-tile 网格: Tm = 2, Tn = 2  →  四个块

        N 方向
      ┌──────────┬──────────┐
  M   │ C00      │ C01      │     每块输出 128×128
      ├──────────┼──────────┤     每块都要把全部 K=1024 乘完
      │ C10      │ C11      │     才能写出、才能进 AllReduce
      └──────────┴──────────┘

K 向折数 n_k = K / BK = 4     （每块内部 4 次 MMA 累加）
wave = 1                      （一次只占 1 条 Cube，四块串行）
```

### 10.2 代价模型怎么得到 `dur = 10 μs`

一块 C-tile 的算术量、搬运量是死的：

```text
FLOPs  = 2 × BM × BN × K
       = 2 × 128 × 128 × 1024
       = 33,554,432

读 X   = BM × K × 2 字节 = 128 × 1024 × 2 = 262,144
读 W   = K × BN × 2 字节 = 1024 × 128 × 2 = 262,144
写 Y   = BM × BN × 2 字节 = 128 × 128 × 2 = 32,768
Bytes  = 557,056
```

示意硬件表（本例手算用）：

```text
Peak_cube_eff = 4.2 TFLOPS     # 这条 Cube 核的有效吞吐，已含利用率
BW_mem        = 1.6 TB/s       # 本 tile 读写下的有效带宽
T_fill        = 1.0 μs         # 第一条 K 片灌流水
T_drain       = 1.0 μs         # 最后一次累加写出前的排空
```

屋顶线（roofline，背景：算得动还是搬得动，取较慢的那边）：

```text
T_flops = FLOPs / Peak_cube_eff
        = 33,554,432 / 4.2e12
        = 8.0 μs

T_bytes = Bytes / BW_mem
        = 557,056 / 1.6e12
        = 0.35 μs

T_roofline = max(T_flops, T_bytes) = 8.0 μs     ← 算力墙，内存不是瓶颈
dur[t]     = T_fill + T_roofline + T_drain
           = 1.0 + 8.0 + 1.0
           = 10.0 μs
```

四块形状相同，所以

```text
dur[C00] = dur[C01] = dur[C10] = dur[C11] = 10 μs
```

编译器把这四个 10 写进流图节点。MILP 把它们当常数，不再改。

更细的编译器会按 K 片估：

```text
T_k = max( 2×BM×BN×BK / Peak ,  (BM×BK + BK×BN)×2 / BW )
dur = T_fill + n_k × T_k + T_drain
```

本例 `n_k=4`，若每片 2.0 μs，同样得到 `1 + 4×2 + 1 = 10 μs`。  
工程上第三种来源是**实测**：把 kernel 打点，读每个 C-tile 写完的时间戳，回填 `dur`。三种来源都发生在计算编译器，不发生在通信求解器。

### 10.3 流图把 `dur` 串成完成时刻

kernel 已排好的发射顺序（行主序，N 在内）：

```text
issue:  C00 → C01 → C10 → C11
边:     同一 C-tile 的 4 个 K 片必须按序（已折进 dur 里，节点不再拆 K）
```

展开公式：

```text
start[第一个] = 0
start[t]      = start[前驱] + dur[前驱]
finish[t]     = start[t] + dur[t]
```

手算时间轴（单位 μs）：

```text
tile   start   dur   finish
C00      0      10     10
C01     10      10     20
C10     20      10     30
C11     30      10     40
```

```text
时间 →
0        10       20       30       40
|--------|--------|--------|--------|
  C00       C01       C10       C11
```

这就是「计算流图得到具体 tile 耗时」的全部含义：  
**耗时数字来自 10.2；落在哪一微秒来自 10.3。**

### 10.4 再推通信 `ready`

```text
store_delay = 2 μs          # 从 Cube 写完到 HBM/L2 对通信可见
tiles_per_chunk = 1         # 一块 C-tile 就是一包
```

```text
ready[C00] = 10 + 2 = 12 μs     ← 离线估算：第一包大约这时才有货
ready[C01] = 20 + 2 = 22
ready[C10] = 30 + 2 = 32
ready[C11] = 40 + 2 = 42
```

若约定沿 N 每 2 块打一包（`tiles_per_chunk=2`）：

```text
chunk0 = {C00, C01}   ready_est = max(12, 22) = 22 μs
chunk1 = {C10, C11}   ready_est = max(32, 42) = 42 μs
```

离线求解器只看见这张估时表，用来比较「一块一包」和「两块一包」谁更快。它不会改 10 μs，也不会改 C00 先于 C01。  
运行时下降成 `C00.done → send0`，没有 12 这个数。见第 11 节。

### 10.5 对比：同样切块，wave=2 时时间轴变了

tiling 网格不变，只是同一拍两条 Cube 并行（C00 与 C01 一拍，C10 与 C11 一拍）：

```text
tile   start   dur   finish   ready(+2)
C00      0      10     10       12
C01      0      10     10       12      ← 与 C00 同时出炉
C10     10      10     20       22
C11     10      10     20       22
```

第一包从 12 μs 就能发，不必等到 22。  
**这就是为什么必须给计算流图，而不是只给 tiling：** 切成哪几块是空间问题；何时出炉取决于发射顺序和 wave。两者缺一，`ready` 都是空的。

### 10.6 真实规模怎么套同一套算法

把本例的 M、N 放大到线性层常见尺寸即可，公式不变：

```text
M=4096, N=4096, K=1024, BM=BN=128
C-tile 数 = 32 × 32 = 1024
若仍 dur=10 μs 且 wave=1：总计算 ≈ 10.24 ms
若 wave=8：总计算 ≈ 1.28 ms
ready[第 k 个发出的 tile] = finish[k] + store_delay
```

不要为 1024 个节点手填 `dur`。同类 C-tile 共用一个代价模型输出；流图只保存节点名、发射序、以及「这一类 dur=10 μs」。

第 10 节的 12/22/32/42 μs **只是离线估算**。第 11 节说明运行时为什么不能、也不需要按这些数启动通信。

---

## 11. 运行时只要先后依赖，不要通信启动时刻

**结论：正确性只需要计算→通信的先后边。不需要、也不该把通信启动时刻写进内核。**  
第 10 节算出 `ready[C00]=12 μs`，那是给 MILP 比较「几种打包谁更快」用的。运行时若 `sleep(12μs); send()`，访存争抢和抖动会让这个数立刻失效，而且可能在数据还没写完时就发——那是正确性事故，不是性能误差。

### 11.1 两层各要什么

| | 运行时（必须对） | 离线 MILP（用来选结构） |
| --- | --- | --- |
| 要的东西 | produce 边：这些 C-tile 写完 → 允许发 chunk `c` | 估时 `dur`、`alpha/beta`，推出 `ready_est` |
| 启动方式 | Cube/写完成事件、fence、CQE | 没有启动；只有搜索坐标 `epoch` |
| 重叠怎么发生 | 下一块在算，上一块的 WQE 已在链路上 | 用估时判断「现在发是否划算」 |
| 估时不准会怎样 | **不影响正确性**（事件晚到就晚发） | 选出的打包/VT 可能不是最优，但仍合法 |
| 墙上时钟 | 禁止 | 允许作为目标函数的数字，下降时丢掉 |

下降只有一句话：

```text
规划：  send(c, link, VT, epoch)  且 epoch ≥ ready_est[c]
运行：  wait( finish_event(produce^{-1}(c)) );  post_wqe(c, link, VT)
```

`epoch` 还编码通信自己的先后（Ring 第 i 跳的 CQE → 第 i+1 跳的 WQE）。这些也是事件边，同样不是定时器。

### 11.2 什么时候连估时都可以不要

策略若已经钉死：

```text
冻 Ring / HD
tiles_per_chunk = 1      # 一块 C-tile 就是一包
写完就发                 # 不故意等下一块来合并
VT 数、绑定已给定
```

则**没有选择可搜**。不需要 MILP，不需要 `dur`，不需要 `ready`。  
编译器只要给出 tiling + produce 边，运行时就是融合算子：C00 写完 post 第一包，同时 C01 在算。重叠是事件调度的自然结果。

只有还要在下面这些里做选择时，才值得估时、跑 MILP：

- 几块打一包（等下一块能摊薄 alpha，但推迟 ready）
- 几条 VT / 哪条链路
- 要不要故意推迟某包，给 HBM 让路（规划期的粗预算，不是运行时时刻表）

选出的是**结构**（打包粒度、映射），不是一张启动时刻表。

### 11.3 重叠、访存争抢、抖动为什么估不准，以及为什么无所谓

估时会偏，原因就是融合本身：

```text
规划时假设：Cube 独占 HBM，链路独占 HBM
运行时事实：边算边传，两边抢同一套 HBM / 互连
另外还有：缓存、对端拥塞、CQE 抖动、其他流
```

这是循环依赖：通信一启动，计算的 `dur` 就变；`dur` 一变，规划的启动点就错。  
**拆开循环的办法不是把启动点估得更准，而是不要用启动点。**

```text
C00 写完 ──事件──► post 第 1 包
                    同时 C01 继续算     ← 重叠，HBM 会抢
C01 实际 14 μs 才写完（规划以为 10）
         ──事件──► post 第 2 包         ← 晚 4 μs，结果仍对
```

争抢的后果是两边都变慢一点，总时间变长，但先后关系不变。  
占用轴 `occupancy(τ)` 因此只做**规划期粗预算**（例如假定算、传各分到 70% HBM，避免选出明显打满的打包），不要下降成微秒级 HBM 时刻表。残差交给硬件仲裁；若还要运行时再切 VT，走 PReCCL 那类自适应，而不是改闹钟。

### 11.4 和第 10 节数字怎么对齐

```text
离线看见：ready[C00]=12, C01=22, C10=32, C11=42
下降成：  C00.done → send0 ; C01.done → send1 ; ...
运行时：  没有 12 这个数；C00.done 何时来何时发
```

若实测 C00 因争抢变成 14 μs 才 done，第 1 包 14 μs 才上链路——这正是该有的行为，不是 bug。

### 11.5 领导一句话

**依赖保证对；估时只用来选方案；时钟不进内核。**  
问「要不要给出通信启动时间」：运行时不要。离线可以估一个，用完就扔。

---

## 12. 昇腾 950：计算输出不落 HBM，直接发起通信

**结论：中间结果不要走 HBM，但也不能从 Cube 累加器直接发。**  
握手层是 **UB 域的片上缓冲（UB Memory）**：Cube 把一块写完的 C-tile 放进 ping-pong 槽，STARS 事件一到，CCU 用该槽当地址发 UB WQE。IO Die 端口转发不进计算 Die，也就不占 HBM。

这和第 11 节一致：运行时仍是「写完才发」，只是「写完」的可见层从 HBM 改成 UB Memory。

### 12.1 950 上各层谁看得见通信

先把两个都叫 UB 的东西分开（背景）：

```text
核上 UB     AscendC Unified Buffer，Cube/Vector 核内暂存
互连 UB     Unified Bus，超节点内芯片互连
UB Memory   互连 UB 引擎可当 WQE 源/目的的片上缓冲（通信可见）
```

| 层 | 算得完 C-tile？ | URMA / UB 引擎能当 src？ | 落不落 HBM | 融合里扮演什么 |
| --- | --- | --- | --- | --- |
| L0C | 累加器，K 没折完不能发 | 否 | 否 | 只算，不发 |
| L1 / 核上 UB | 可暂存一块 | 通常否（不是 rank 间地址） | 否 | 核内搬运 |
| **UB Memory** | 放得下一块 C-tile | **是，CCU 的 WQE src/dst** | **否** | **计算-通信握手层** |
| HBM | 放得下整表 Y | 是，但是 AICPU_TS 默认路径 | 是 | 融合失败才回退；最终 Y 可在此落地 |
| UB Port / IO Die | — | 转发 | 不进计算 Die | 超节点内直传 |
| UBoE | — | 超节点间 | 不默认等同片上直传 | 不要假设还能不落 HBM |

```text
X、W 仍从 HBM 读进来    ← 矩阵乘输入太大，片上放不下
Y_partial  不写 HBM     ← 本节省掉的就是这一笔
AllReduce 在 UB Memory 上收、加、再发
规约完的 Y  需要时再写回 HBM 给下一层    ← 这是融合算子结束之后，不是中间
```

950PR 片上/HBM 档 1.6TB/s。中间再写一笔 Y，计算和通信会在这堵墙上对打；PReCCL 方案里 PR 的 L2 也**禁止 HBM 中继**。所以 950PR 上融合的主路径必须是 UB 域，不是「Cube 写 HBM + HCCL 再读 HBM」。

### 12.2 推荐路径：`CCU_SCHED` + STARS 同拍

不要用默认 `AICPU_TS` 做这条融合。官方语义里 AI CPU 模式是 **HBM↔HBM**；CCU 调度才是「不用 CcuBuffer，rank 间片上内存直传」。

```text
STARS 同时挂两条 Mission（通信占用建议 ≤16，给 Cube 留配额）

  Cube Mission
    MMA 折完一块 C-tile
    DataMove / Vector：L0C → UB Memory 槽 s     # 片上搬，不经 HBM
    发 Notify / 完成事件

  CCU Mission   （等上面这条事件，不是等 12 μs）
    UB WQE { src = 槽 s, dst = 对端 UB Memory 槽 }
    Jetty k → Transport Channel k → UB Port
    IO Die 可互转，不进计算 Die

  对端
    收到后在 UB Memory 上 Vector 做 reduce
    下一跳仍以 UB Memory 为 src
```

事件链（和第 11 节同一句话，只换可见层）：

```text
wait( UB Memory 槽 s 写完成 );  post_ub_wqe(src=s, vt=k)
槽 s 在对应 CQE 回来之前禁止 Cube 覆写     # workspace = ping-pong 槽数
```

`store_delay` 在这条路径上不是「寄存器→HBM」的微秒数，而是片上 fence / Notify，规划里可当 0。

### 12.3 ping-pong：片上放不下整张 Y

UB Memory 远小于 `M×N`。所以「不落 HBM」**逼出**第 10 节那种按 C-tile 流式发，而不是等全部算完。

```text
槽数建议 2～4（send 用 2 槽 ping-pong，Ring 再加 1 个 recv 槽）

例：C-tile 128×128 FP16 = 32KB
    4 槽 ≈ 128KB，远小于 950PR 的 HBM，也小于 CCU 文档里 512KB～2MB 的通信 Tile

因此融合主路径默认 tiles_per_chunk = 1
想打成 512KB 一包，必须先在 UB Memory 里攒够，或承认会溢到 HBM
```

和 PReCCL 推荐 Tile 的冲突要显式处理：

| | 通信切分（PReCCL） | 片上融合 |
| --- | --- | --- |
| 粒度 | 512KB～2MB，摊 alpha | 一块 C-tile（几十 KB） |
| 源地址 | 往往假定大 buffer | 必须是当前活着的槽 |
| 首期 | 融合路径**不做动态切分**（见 950 设计 2.1） | 槽位已很紧，再迁尾部会打乱覆写规则 |

首期：冻 Ring、一块一包、写完就发、不做 VT 动态迁徙。等融合跑稳再谈切分。

### 12.4 AllReduce 中间也不要回 HBM

Ring / HD 每一跳都是「收一块 + 本地加 + 再发」。若 reduce 写回 HBM 再读出发，中间融合就废了。

```text
recv 槽  ⊂ UB Memory
acc      ⊂ UB Memory（Vector add，或小块回 Cube 再写回槽）
send 槽  ⊂ UB Memory
禁止：recv → HBM → send
```

对端关系仍冻在 sketch 里。求解器不改邻居，只决定这块挂哪条 Jetty——但 src 始终是槽地址，不是 HBM VA。

### 12.5 哪些情况做不到「全程不落 HBM」

| 情况 | 怎么办 |
| --- | --- |
| 输入 X、W | 本来就在 HBM，允许。省的是 Y_partial |
| 融合结束、下一层要 GM | 规约完的 Y **一次**写回 HBM。这是算子出口，不是中间 |
| 超节点间 UBoE | CCU 调度对象是 UB WQE。跨超节点不要假设仍能片上直传；机内 UB 融合 + 机间允许落 HBM/走 AI CPU |
| 单机大 AR 触发官方回退 | 回退 `AICPU_TS` = HBM↔HBM，本 CCT 放弃片上路径 |
| CCU 配额被算满（32 Mission） | 同样回退。融合优先占 Mission，通信 VT 建议 ≤8 |
| AscendC+CCU 自定义核常见上限 | ≤8 卡 FullMesh、单次 ≤256MB。128 卡 950PR 用 **Ring/NHR 而不是 FullMesh** |
| 950PR L2 拥塞 | 只允许 IO Die 换端口，禁止 SDMA 经 HBM 中继（与本路径一致） |

验收：中间路径的 SDMA/HBM 计数应为 0（X/W 读和最终 Y 写除外）。这和 `docs/ascend950_preccl_design.md` 里「950PR 上 L2 不得出现额外 HBM 中继」同一根尺子。

### 12.6 对融合合同改什么（相对落 HBM 版）

```text
visibility     = UB_MEMORY          # 不再是 HBM
produce 边     = C-tile 写入槽 s 完成
workspace      = ping-pong 槽数（2～4），不是 HBM 双缓冲
store_delay    ≈ 0（片上 Notify）
u.HBM_Y        = 0                  # Y_partial 不占 HBM
u.HBM_XW       = 读 X/W 的带宽      # 仍在
u.STARS        = Cube Mission + CCU Mission 同拍
exclusive[]    += 950PR：Y_partial 禁止 HBM；禁止融合 CCT 内 HBM 中继
expansion      = CCU_SCHED          # 禁止把 AICPU_TS 当融合主路径
tiles_per_chunk= 1                  # 被槽容量逼出来的默认
动态 VT 切分   = 首期关
```

运行时仍然**没有通信启动时刻**。Cube 写完槽 s 的事件就是 post 的唯一扳机。

### 12.7 和 2×2 手算怎么对齐

第 10 节四个 C-tile，落在 2 个发送槽上：

```text
C00 → 槽0 写完 → send0（VT k）     同时 C01 往槽1 写
C01 → 槽1 写完 → send1             槽0 要等 send0 的 CQE 才能给 C10 覆写
C10 → 槽0
C11 → 槽1
```

规划里的 12 μs 仍然只是估时。运行时是槽写完就发；HBM 上没有这四块 Y_partial。

### 12.8 领导一句话

**950 上的融合 = CCU 调度 + UB Memory 槽 + 写完事件。**  
不是 Cube 直连网口，也不是写 HBM 再 HCCL。950PR 尤其不能走中间 HBM，否则打在 1.6TB/s 墙上。

---

## 13. 求解器输出件：对齐 MSCCL/HCCL 的 XML→可执行件，但不要只出集体 XML

集体综合器（MSCCL / MSCCLang / 昇腾 HCCLang）的终点是一张**只含通信步骤**的算法描述，再编成运行时能加载的东西（XML、再转 bin，或直接生成 HCCL 的 `.h/.cc` 编排）。融合求解器不能停在这一张上：HCCL 假定数据已经在输入 buffer 里；融合多出来的是「哪一块 C-tile 写完才允许发哪一包」，以及这包在 **UB Memory 槽** 而不在 HBM/CCL buffer。

### 13.1 建议三层，不要一个文件包打天下

```text
求解器唯一合同     fused_mm_ar.json     （人能 diff、能版本化）
        │
        ├─ 下降 A  集体子图  →  现有 XML → bin / HCCLang → HCCL 能调
        ├─ 下降 B  事件表    →  STARS Notify：槽写完才 post
        └─ 下降 C  槽位表    →  UB Memory ping-pong 布局
```

`fused_mm_ar.json` 是求解器的**输出件**。XML/bin 是给 HCCL 通信引擎的**派生物**，不要让求解器直接只吐 XML。

### 13.2 合同里必须有的字段（运行时要读）

```text
meta
  op            = mm_allreduce
  sketch        = Ring | HD          # 冻死的邻居关系
  expansion     = CCU_SCHED
  visibility    = UB_MEMORY
  dtype, nranks

tiling_echo                         # 回显编译器输入，求解器不改
  BM, BN, BK, grid, issue_order
  tiles_per_chunk

workspace
  kind          = ub_memory
  nslots, slot_bytes
  slot_of[tile]                     # C00→槽0, C01→槽1, ...

produce                             # 融合相对纯集体多出来的核心
  { tile, chunk, slot, event }      # tile 写完槽 → 允许发 chunk

collective                          # 和 MSCCL 同构的那一段
  chunk 字节、每 rank 每跳：
    send/recv/reduce
    对端 rank、VT/Jetty
    依赖：等哪条 produce 事件、等哪条远端 CQE
```

### 13.3 合同里禁止出现的字段

墙上时钟（`start_us`、`ready_us`、`sleep`）、新的 `BM/BN`、新的 tile 序。估时只允许写在可选的 `debug_plan.json`，运行时加载器不得读。

### 13.4 下降到 HCCL 时 XML 里多什么、不多什么

现有集体 XML 继续描述 send/recv/reduce。融合只多两类节点，不要把 MMA 写进 XML：

```text
<wait event="tile_C00_slot0"/>     # 本地 Cube 写完槽，不是定时器
<send chunk="0" vt="0" src="slot0"/>
```

HCCL 侧把 `<wait event>` 落到 LocalNotify / STARS 依赖；`<send src=slot>` 落到 CCU WQE 的 UB Memory 地址。资源计算要多报：槽字节、槽数、每条 produce 边一个 Notify。这和 HCCL 自定义算法的「资源计算 + 编排」是对齐的，只是 Input 从 CCL buffer 换成槽。

### 13.5 和 2×2 例子对应的最小输出

```text
produce:  C00→chunk0@槽0, C01→chunk1@槽1, C10→chunk2@槽0, C11→chunk3@槽1
comm:     Ring 上 chunk0..3 的 send/recv/reduce（邻居来自 sketch）
constraint: 覆写槽0 必须晚于 chunk0 的发送 CQE
```

没有 12 μs。HCCL 拿到的 bin 里是事件边和槽地址。

---

## 14. 替换路径：框架 API 不变，换的是算子内部的 plan

CANN 已经有融合算子接口（如 `aclnnGroupedMatMulAllReduce` / `aclnnMatmulAllReduce`）。框架（PyTorch / MindSpore）只调这一层：先 GetWorkspaceSize，再 Execute。**自动生成不新增、不改这些参数。** 生成件不是给框架调用的新 API，而是这个算子在 GetWorkspaceSize 里查到的一份 **plan**，用来换掉出厂 kernel 的内部编排。

和 `HCCL_ALGO`、MSCCL 的 `XML_FILES` 同一类机制：调用点不变，实现被选中。

```text
框架
  npu_grouped_matmul_all_reduce(x, w, ..., group)     # 一行不改
        │
        ▼
aclnnGroupedMatMulAllReduceGetWorkspaceSize(...)      # 签名不变
        │  按 (shape, dtype, world, SKU, group) 查表
        ├─ 命中生成件 → executor 挂上这份 plan
        └─ 未命中     → 出厂 MC2 kernel（现网兜底）
        ▼
aclnnGroupedMatMulAllReduce(workspace, executor, stream)
```

### 14.1 输出件交给谁、放哪

离线求解按「问题签名」各出一份，安装进算子能搜到的目录，不进训练脚本。

```text
$MC2_FUSED_PLAN_DIR/          # 或打进自定义 opp
  manifest.json               # 签名 → 文件名
  gmm_ar_<signature>.bin      # 运行时真正加载的
  gmm_ar_<signature>.json     # 可选，人读/对账，运行时可不读
```

`manifest.json` 用输入就能算出来的键索引，例如：

```text
op, sku, nranks, dtype, M, N, K, group_pattern, reduce=sum
```

GMM 的 `groupList` 是动态的：签名里写分组模式（组数、各组 M 是否同构），不要把某一次运行的绝对指针写进 plan。

### 14.2 GetWorkspaceSize 里做什么（API 内部，调用者看不见）

1. 用本次 `x/weight/groupList/group/dtype` 算签名。  
2. 在 `MC2_FUSED_PLAN_DIR` 查 manifest。  
3. 命中：`workspaceSize` 按 plan 里的槽数/字节返回（可以和出厂值不同，这仍是原接口的出参）；executor 保存 plan 句柄。  
4. 未命中或校验失败（邻居数对不上、可见层不是 UB Memory）：走出现厂实现，行为与今天完全一致。  
5. Execute 只按 executor 里已绑定的 plan 下 Cube Mission 和 CCU WQE。

`commTurn` 等现有参数**保留在 API 上**，保证 ABI。命中生成件时以内编排为准，该参数不再自己切通信份数。

### 14.3 谁在何时生成

作业前或编译自定义 opp 时，对要保的形状各跑一次求解器，把 bin 装进目录。训练进程启动后只读盘、不求解。形状变了找不到签名，自动回退出厂，而不是改框架代码。

### 14.4 和出厂实现不是同一种 bin

出厂 GMM+AllReduce **不是**「一个形状一份算法 bin」。它是普通 AscendC 算子：

- 芯片上的 **kernel `.o`**：按 SoC / tilingKey 编好的一段通用程序，里面是 MatMul + `HcclServer`，从 tiling 结构读切块和通信份数。  
- Host 上的 **tiling 库**：`GetWorkspaceSize` 现场根据本次 shape 算出那块结构。  
- **aclnn 动态库**：框架调的入口。

通信步骤写在 C++/AscendC 里，不写成 MSCCL 那种 XML。自动生成要加的 `gmm_ar_<签名>.bin` 才是按问题签名实例化的 plan，用来**替换 Host 算出来的那份编排**（以及 kernel 怎么发），不是 CANN 今天已经在用的交付形态。
