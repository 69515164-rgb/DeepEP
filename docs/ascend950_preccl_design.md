# 昇腾 950 上实现 PReCCL 式运行时拥塞检测与数据切分

目标：在 **不换 HCCL 算法图、不拆单 Jetty 保序、不中途切集体** 的前提下，把一次集体的字节从慢 VT 迁到快 VT，让各 VT 完成时间对齐。PReCCL 的三件事在 950 上分别落到 CCU / URMA / HCCL tiling，而不是 NCCL channel。

适用范围：昇腾 950PR / 950DT + CANN HCCL，超节点内 UB、超节点间 UBoE。默认展开模式 `HCCL_OP_EXPANSION_MODE=CCU_SCHED`（950PR 不支持 `CCU_MS`）。

不在本方案范围：换整张 schedule（Theseus）、MoE DeepEP/UBEP 式 P2P 编排（FAST）、传输层改 DCQCN/PFC。

---

## 1. 950 上为什么不能照抄 NCCL 版 PReCCL

| NCCL / GPU 假设 | 950 实际 | 设计含义 |
| --- | --- | --- |
| VT = NCCL channel = 1 QP | 集体由 **CCU** 展开成并行小任务，经 **URMA Jetty** 出 **UB Port / UBoE** | VT 必须绑到「一条可独立等待完成的 CCU Mission + Jetty」 |
| GPU SM 空转计数 = stall | CCU / STARS 在等 CQE、Notify、credit，AI Core 时钟跨片不同步 | stall 记在 **CCU wait 循环 / Jetty SQ 不前进**，不用 AIC 时钟差 |
| 机内 NVLink + 机间 IB | 超节点内是一张 UB（最多理论 8192 卡，生产常见 128 卡）；出超节点才是 2×400G UBoE | 检测和切分要分 **UB 域** 与 **UBoE 域** |
| ECMP 极化是主因 | UB 域是 credit / 端口 / 片上转发；UBoE 域才是以太 ECMP | UB 内 L1 是换 Port/Transport Channel，不是换五元组 |
| 中途不能切 QP | URMA RTP 多 Transport Channel 仍保序；CTP 无端到端重传 | **禁止对同一 Jetty 喷包或中途改目的** |

白皮书硬规格（实现时当容量上限）：

- STARS 最多并发 **32 个 CCU 任务**、**64 个 UB Jetty**、**32 条 SDMA**。
- URMA **RTP**：可靠、约 4 Port 可靠带宽，多 Transport Channel 做多路径。
- URMA **CTP**：无端到端重传，约 9 Port，也可多路径。
- 片上 **9 个 x4 Port** 可在 IO Die 互转，不进计算 Die、不占 HBM。
- CCU 把 Broadcast / RS / AG / AR / All2All(v) 展开成循环小粒度任务。

推荐 VT 数量：**8～16**。超过 16 会挤占 CCU/Jetty，和计算融合抢 STARS；少于 8 则 UBoE 两口、UB 多 Port 的路径熵不够。

---

## 2. 对象模型：VT 在 950 上绑什么

```text
一次 HCCL 集体
  └── 逻辑算法（Ring / NHR / Pipeline / 分组 FullMesh）   ← 本方案不改
        └── Tile[0..T)          ← 字节切分的最小单位（HCCL 已有 repeat/TileLen）
              └── VT[k]         ← 本方案的调度对象
                    ├── CCU Mission k     （STARS 可见的一条通信任务）
                    ├── URMA Jetty k      （SQE/CQE，FIFO）
                    ├── Transport Channel k
                    └── 出端口：UB Port 或 UBoE 口
```

硬约束：

```text
VT = CCU Mission = URMA Jetty = 1 条 Transport Channel
```

一条 VT 只走一个 Jetty。Jetty 内 FIFO，才能把「等 CQE」当成 stall。禁止「一个 Mission 条带多个 Jetty」——否则 stall 无法归因，字节也无法按 VT 迁。

### 2.1 超节点内（UB 域）映射

假设生产超节点 128 卡、芯片 18 个 UB x4 Port，集合常用若干 Port 做 scale-up。

推荐 8 VT：

| VT | 角色 | 绑定 |
| --- | --- | --- |
| 0–1 | Port 组 A 的两条 Transport Channel | 同 Port 不同 Channel，或相邻 Port |
| 2–3 | Port 组 B | 同上 |
| 4–5 | Port 组 C | 同上 |
| 6–7 | Port 组 D | 同上 |

同组两条 VT 必须能被观测为两条独立 FIFO。若硬件把它们哈希到同一条物理出路，只当一条容量用。

### 2.2 超节点间（UBoE 域）映射

整芯片 2×400G UBoE，和 UB SerDes 静态复用。

推荐 4 VT：每条 UBoE 口 2 个 Jetty（不同 UDP 源端口），制造 ECMP 熵。这是和 NCCL `channel=QP` 最像的一层。

跨超节点集体（梯度 Reduce、DP）走这 4 VT；超节点内 EP/TP 走上面 8 VT。两套 VT 表不要混在一次集体里。

### 2.3 和 HCCL 展开模式的关系

| 模式 | 950PR | 本方案 |
| --- | --- | --- |
| `CCU_SCHED` | 推荐 | **主路径**：CCU 向 UB 下 WQE，HBM↔HBM |
| `CCU_MS` | 不支持 | 不用 |
| `AI_CPU` | 大数据量会自动回退 | 回退时 VT 绑 AI CPU 侧并发 stream，stall 改记 stream wait；精度下降但协议不变 |
| `AIV` | AlltoAll 小消息 | 仅 <1MB 点对点；本方案不在 AIV 上做动态切分 |

图模式（Ascend IR / aclgraph）：首次捕获用当时的 tile→VT 表。**权重变化后必须使该通信子图失效并重建**，或只在单算子模式做动态切分。首期建议：动态切分只开在单算子 / 非捕获路径；图模式用捕获前的静态权重。

---

## 3. 运行时拥塞检测

原则与 PReCCL 相同：**跨芯片时钟不可比，只比「这条 FIFO 有没有往前走」**。950 上 stall 的定义是 CCU/URMA 的等待，不是 AIC 自旋。

### 3.1 主信号（带内，每 VT，必须有）

每个 VT 在本集体（一个 CCT）内累计：

| 计数 | 在哪记 | 含义 |
| --- | --- | --- |
| `sq_stall` | Jetty SQ 满或 doorbell 发不出去的循环次数 | 本端发送被信用/下游堵住 |
| `cq_stall` | 已提交 WQE、等 CQE/Notify 的循环次数 | 路径或对端慢 |
| `bytes_done` | CQE 完成长度之和 | 实际前进量 |
| `wqe_posted` | 已下发 WQE 数 | 和完成数一起算 in-flight |
| `retry` | RTP 重传 / 链路层重传次数 | 区分拥塞和丢包 |

实现位置：

- `CCU_SCHED`：在 CCU Mission 等待 UB 完成的 poll 里加 1，**不要**用 AIC 时间戳相减。
- 计数器放在 Host-visible 或 UB Memory 上的 per-VT 结构，64-bit，epoch 号打头。
- 随 HCCL 已有的集体同步（ring 的 ack、tree 的 notify、AllGather 的尾包）**捎带** 8–16 字节：`{epoch, vt_id, stall, bytes_done}`。不新建控制连接。

完成时间估计（各 rank 本地、公式相同）：

```text
T_hat[k] = alpha[k] + bytes_remain[k] / B_hat[k]

B_hat[k] = bytes_done[k] / max(active_time[k], eps)
active_time[k] 用「本 VT 自己的 stall 循环次数」标定，
              不用跨 rank 墙钟。
alpha[k] 用最近 W 个 epoch 的空载/小消息拟合，在线更新。
```

`stall` 高且 `bytes_done` 低 → 这条 VT 慢。`stall` 低且 `bytes_done` 已到份额 → 这条 VT 快，下一 epoch 可多吃字节。

### 3.2 辅信号（判因，不直接改权重）

| 域 | 信号 | 用途 |
| --- | --- | --- |
| UB Port | 端口 credit 耗尽、片上转发队列档位 | 同组 VT 一起慢 → Port/路径级，不是单 Jetty |
| RTP Channel | credit stall、重传 | 换 Transport Channel（L1） |
| UBoE | ECN 标记、PFC 暂停、QP RTT | 超节点间 ECMP / 下行 incast |
| STARS | 该 CCU Mission 的 TOP-DOWN 耗时 | 和 stall 交叉验证 |
| `hccn_tool` / 交换机 telemetry | 仅 UBoE 域 | 辅助 L2，不进热路径 |

辅信号可以晚一个 epoch，允许不准。**权重只由主信号的 `T_hat` 决定**，避免交换机和控制面抖动直接改切分。

### 3.3 判因（决定切哪一层，不决定切多少）

对每个 Port 组（UB）或每条 UBoE 口：

```text
组内 VT 的 T_hat 变异系数 > θ_split 且组均值正常
    → L1：组内换 Channel / 迁字节，不换 Port

组内所有 VT 的 T_hat 都 > θ_hot，且辅信号显示该 Port / 该 UBoE 口热
    → L2：降低整组份额，迁到冷 Port 或另一条 UBoE
          UB 域可走 IO Die 片上转发换出端口（不进 HBM）
          UBoE 域经 UB Memory / SDMA 绕到另一 400G 口

单 VT 连续 E 个 epoch timeout 或 retry 暴涨
    → 隔离该 VT，保留 min_share 探测；不要拆 communicator
      （HCCL_OP_RETRY / lane borrowing 继续处理真故障）
```

UB 域的 L2 优先用 **片上 Port 转发**（白皮书：IO Die 九口互转、不进计算 Die）。这比 NVL72 的 PXN 便宜，不要先走 HBM 中继。

UBoE 的 L2 才类似 NVL72 跨 rail：从另一 400G 进对端，再 UB Memory 到目标 NPU。两口都热就不要再绕，只做组内切分。

### 3.4 稳定化

- 进入阈值高于退出阈值（滞回）。
- `T_hat` 用指数滑动平均，窗口 4～8 个 epoch。
- 噪声属性（瞬时 credit）必须平滑后再参与判因。
- 探测份额：每个健康 VT 至少 `min_share`（建议总字节的 1/32），避免饿死无法观测。

---

## 4. 数据切分

切的是 **Tile 到 VT 的映射**，不是 WQE 内切片，也不是算法边。

### 4.1 切分单位

HCCL 已把 buffer 切成 `TileLen` 的 repeat 块（Prepare 的 `repeat`）。本方案规定：

```text
用户 buffer
  → 按算法切成 rank 块（ReduceScatter/AllGather 语义不变）
    → 每个 rank 块再切成 T 个 Tile
      → 每个 Tile 绑到恰好一个 VT
```

- Tile 是迁徙粒度。建议 Tile = 512KB～2MB（UB 大消息）或 64KB～256KB（UBoE）。
- 一次集体开始前，tile→VT 表冻结。
- **禁止** 把已下到 Jetty 的 WQE 改绑。
- 连续 Tile 尽量给同一 VT，减少地址跳跃；迁徙时迁「尾部连续区间」，对应 PReCCL 的 contiguous byte range。

### 4.2 各集体怎么切

**AllReduce / ReduceScatter / AllGather / Ring / NHR / Pipeline**

- 每个 VT 拿一段连续字节（或连续 Tile）。
- 算法图不变：还是原来的 ring/tree 邻居。
- 慢 VT 的尾部区间划给快 VT。总字节守恒。
- Pipeline 算法已经机内/机间并发：只切 **同一层**（超节点内或超节点间）的 VT，不要把 level0 字节改派到 level1。

**Tree**

- 同样按连续区间切。
- 注意树的父子依赖：迁徙后每个 VT 仍走同一棵逻辑树的同一角色，只是 payload 变短/变长。

**AlltoAll / AlltoAllv / AlltoAllvc**

- 先按目的 rank 切，再在每个目的上按 Tile 分给多个 VT。
- 超节点内：分组 FullMesh 已有「控并发、防一打多」。本方案 **不增加扇出**，只在已有并发组内调 Tile 份额。
- 超节点间：默认并发更低。不要为了切分把 FullMesh 打成更密的一打多。
- 仅当 L2 判定某入端口 incast，且片上转发或对端 UB Memory 绕路的收益 > 多一跳开销时，才启用 **单跳中继**（PReCCL AlltoAll 的 NVLink relay 在 950 上对应 IO Die 转发或 UB Memory）。中继开关是 epoch 级布尔，不是逐包。

**AlltoAllv 倾斜**

- 本方案不替代 FAST。矩阵未知或 DeepEP/UBEP P2P 路径 **不做** Birkhoff。
- 若业务走 HCCL `AlltoAllv` 且已有 recv counts：只在已有 VT 上按「该目的的字节」加权，仍然不换配对关系。

### 4.3 Epoch 与一致性

```text
epoch e:
  1. 所有 rank 用同一张 tile→VT 表跑完本次集体
  2. 同步捎带每 VT 的 {stall, bytes_done}
  3. 各 rank 用同一确定性函数算 epoch e+1 的权重
  4. 下一次 HCCL 集体（下一次 Prepare）加载新表
```

触发同步的频率：每 `K` 次集体一次（建议训练 K=1，即每步；推理 decode K 可更大或关闭）。Agreement 用 HCCL 已有的小 AllReduce 做向量规约（取 stall 的 max 或 sum，公式固定）。不要上 Raft。

图模式：新权重 → 使通信子图 cache 失效。做不到就保持静态切分。

和 `HCCL_OP_RETRY` 的关系：retry 是 CQE 错了整算子重做；本方案是成功集体之间改份额。两者同时开：retry 成功后本 epoch 权重不变，下一 epoch 再根据 stall 调。

### 4.4 分配器（切多少）

目标：健康 VT 的 `T_hat` 尽量接近。禁止每 VT 独立 AIMD。

```text
输入：T_hat[k], B_hat[k], 组标签(Port 或 UBoE 口), 健康位, 上轮权重 w_e
输出：w_{e+1}，sum(w)=总字节

1. 按组汇总 T_hat
2. L1：组内按 B_hat 正比分配（完成时间均衡）
3. L2：整组过热则按组降低总额，补给最冷组
4. clip：
     w[k] >= min_share
     |w[k]-w_e[k]| <= delta_max * 总字节     # 建议 12.5%～25%
     单组不超过该 Port/口的标定容量
5. 把 w 量化到 Tile 个数（向下取整），余数给当前最快 VT
```

伪代码：

```text
for each group G:
    if all_hot(G):
        budget[G] *= (1 - beta)          # L2 降额
    else:
        budget[G] 保持
cold = 最冷组
budget[cold] += 从热组扣下的字节

for each group G:
    for vt in G:
        w[vt] = budget[G] * B_hat[vt] / sum(B_hat in G)

quantize_to_tiles(w)
apply_min_share_and_delta(w, w_prev)
```

`beta` 建议 0.25。一次只迁一个组的额度，避免振荡。

---

## 5. 控制面与数据面位置

```text
Host / AI CPU
  Allocator（确定性，每 epoch 一次）
  tile→VT 表写入 CCL 配置

STARS
  按表下发最多 32 路 CCU Mission

CCU (CCU_SCHED)
  每 Mission 只向绑定 Jetty 下 WQE
  poll 里累加 stall

URMA
  Jetty k → Transport Channel k → UB Port 或 UBoE

对端
  同 VT 号收齐 Notify/CQE
  算法层 reduce/copy 仍按 HCCL 原语义
```

首期改动面：

1. **HCCL 执行器**（cann-hccl）：tile 绑定、stall 计数、捎带、权重表。
2. **CCU Mission 模板**：一对一 Jetty，禁止一条 Mission 多 Jetty 条带。
3. **Allocator**：可先放 Host，公式纯函数，便于各 rank 复现。
4. 不改 UB 交换机、不改 DCQCN。

---

## 6. 和已有 HCCL 能力的边界

| 已有能力 | 做什么 | 本方案不替代 |
| --- | --- | --- |
| `HCCL_ALGO` 自适应 | 选 Ring/NHR/Pipeline | 不在运行时换算法 |
| `HCCL_OP_RETRY` + lane borrowing | CQE 错了换路重做整个算子 | 那是故障，不是拥塞切分 |
| Pipeline 算法 | 机内/机间两层流水 | 只在单层内切 VT |
| 分组 FullMesh AlltoAll | 控超节点间扇出 | 不加密网 |
| STARS 融合计算通信 | 算力与 CCU 并发 | VT 数给 CCU 留余量 |

---

## 7. 分 SKU / 分域怎么开

| 场景 | 建议 |
| --- | --- |
| 950DT 超节点内 TP/EP（UB） | 8 VT，L1 为主，L2 用片上 Port 转发 |
| 950DT 超节点间 DP Reduce（UBoE） | 4 VT，L1+L2 都开，最像原版 PReCCL |
| 950PR Prefill，通信域在 128 卡 UB 内 | 8 VT，L1；UBoE 关闭 |
| 跨超节点 EP | **不要**用本方案硬切；EP 应落在单 UB 域（此前 FSDP/EP 结论） |
| Decode + UBEP/DeepEP 类 P2P | **关闭**。没有 alltoallv 矩阵，也不能为编排先 AllGather |
| aclgraph 捕获的通信 | 静态切分或关闭 |

950PR 片上内存带宽 1.6TB/s，比 950DT 的 4TB/s 紧。L2 若误走 HBM 中继会打满 PR 的内存墙。PR 上 L2 **只允许 IO Die 转发**，禁止「先拷到本端 HBM 再从另一口发出」的软件中继。

---

## 8. 落地步骤

**阶段 A — 可观测（不切数据）**

- CCU Mission 绑定 1 Jetty，打 stall / bytes_done。
- 捎带上集体同步，Host 打日志。
- 用已知限速（单 Port / 单 UBoE 口 pacing）验证：慢 VT 的 stall 明显上升。

**阶段 B — 只 L1 切分**

- 冻结算法图，只在同 Port 组或同 UBoE 口内迁连续 Tile。
- 开 min_share、delta 限幅、滞回。
- 单算子模式 AllReduce / AllGather 先跑通。

**阶段 C — L2**

- UB：片上换出端口。
- UBoE：跨 400G 口 + 对端 UB Memory。
- AlltoAll 单跳中继默认关，incast 再开。

**阶段 D — 生产约束**

- 图模式 cache 失效策略。
- 与 `HCCL_OP_RETRY` 共存测试。
- VT=8/16 对计算融合的 STARS 抢占测量。

---

## 9. 风险

- **CCU 内部若已对多 Jetty 做条带**：必须关掉，否则检测和切分都假。
- **CTP 无端到端重传**：VT 隔离丢包，但不能在 Jetty 内 SACK；慢 VT 只能降权或隔离。
- **aclgraph 重放旧权重**：会和下一 epoch 表不一致，必须失效或不用动态切分。
- **AlltoAllv + P2P 库**：条件不满足就保持静态，不要为 FAST 式匹配去 AllGather。
- **VT 过多**：32 个 CCU 任务是整片上限，还要给别的通信域留配额。

---

## 10. 验收

- 人为把 1 个 UB Port 或 1 条 UBoE 限到 10%：CCT 相对均分应接近「健康 VT 仍满速」的下界，而不是整次集体跟死链路走。
- 权重只在集体边界变；用 STARS 轨迹确认没有中途改 Jetty。
- 各 rank 下一 epoch 权重向量一致（确定性分配）。
- 控制开销：64MB 级消息上 stall 捎带 <1%（对标 PReCCL）。
- 950PR 上 L2 不得出现额外 HBM 中继拷贝（用 STARS SDMA 计数核对）。
