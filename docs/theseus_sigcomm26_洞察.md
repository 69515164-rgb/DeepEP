# Theseus（SIGCOMM 2026）洞察

论文：*Theseus: Runtime-Adaptive GPU Collective Communication with Hot-Swappable Schedules*  
作者：北大黄群组 + 阿里云（曹家敏、高嘉琦）。DOI `10.1145/3789240.3829134`。后端走 MSCCL++。不要和 Voltron 的 Theseus 混。

官方摘要的四句话已经把贡献钉死：CCL 的 schedule 和选择逻辑在 communicator 初始化时冻死；长作业几小时后负载和硬件健康都变了；Theseus 允许用户灌自定义 schedule 和选择策略，用**集群级运行时属性**（不只是库内计数）选图，并在 GPU 之间**一致、低开销热插拔**；drop-in 替换 NCCL。相对 NCCL：稳态通信最多约 1.61×，动态环境最多约 2.46×，端到端 JCT 最多约 1.84×。

---

## 1. 它真正打的点

NCCL 出厂大约 13 套预置图（Ring / Tree / NVLS …），用 α-β 模型在 **init 选一次**。选完之后：

- MoE expert 倾斜会漂；
- 单 NIC fail-slow / fail-stop；
- 综合器过几小时才倒出更好的图。

库没有「换图」入口。AutoCCL、TCCL 仍停在静态空间或启动时搜。Theseus 把 schedule 从 init 常量变成 **runtime 可替换模块**。

它换的是**整张 schedule**（全局 / GPU / TB / OP 资源），不是 PReCCL 那种同一张图上改 VT 字节。也不在线长出一张从没见过的图——候选必须事先灌进去。

---

## 2. 机制：选图 + 换图，两段都要一致

```text
属性（集群级） → 选择策略 → 所有 GPU agreement → delta 迁移资源树 → 下一 CCT 用新图
```

**选。** 用户定义 selection context：可以看 workload 特征（AlltoAll 倾斜）、硬件健康（NIC 速率、单口挂死），不限于 NCCL 内部的 busbw。自定义 agreement：每隔 N 次集体对齐一次「下一张图是谁」，均摊后开销接近零。不能各 GPU 自己换——Ring 邻居和 AlltoAll 配对会对不上。

**换。** 资源做成树，候选图之间共享通道、TB、缓冲。热插拔做 **delta migration**：只建增量，不把 communicator 拆了重建。这是相对「销毁 comm 再 `ncclCommInit`」的全部理由——长作业不能停。

执行层是 MSCCL++，所以预置几乎只有 NCCL Ring；真正能换的图靠 `NewSched` 灌候选。接口形态（论文实现）：灌 schedule、注册属性、设 exo/选择策略。没有这些，Theseus 退化成「会热切换的 Ring」。

---

## 3. 评测里真正灌进去的图，覆盖什么、盖不住什么

| 候选 | 场景 | 盖不住 |
| --- | --- | --- |
| 分层 mesh / ring / 双二叉树 | 稳态 AllReduce 换结构吃带宽 | 同一张 ring 上某 VT 变慢（该 PReCCL） |
| AlltoAll 两档：低倾斜 PXN 形 vs 高倾斜 FAST-lite | expert 倾斜跨过阈值换档 | 在线 alltoallv matching（该 FAST）；DeepEP 式 P2P 没有全局矩阵 |
| 16 卡单 NIC、同节点中继 | 一口 10% / 0% 时绕路 | 多口同时坏、组合故障不在候选里 |
| TECCL 倒出的中间解（评测两套） | 综合器异步算完，运行时接住 | 综合器还没算完的那几小时 |
| 9 种 TB 网格 | 换并行度 / SM 占用 | 不改算法边 |

fail-slow 是最能说明「换图 ≠ 调字节」的实验：一口限到 10%，NCCL 迭代从约 1.75s 掉到约 3.76s；一口 0%，NCCL 直接停。Theseus 切到中继图回到约 2.03s（约 1.16× 健康值）。PReCCL 在死链上只能把字节从死 VT 迁走，迁不走「必须经过那口」的边。

---

## 4. 数字怎么读

| 数字 | 含义 | 别过度解读 |
| --- | --- | --- |
| 稳态 1.61× 通信 | 相对 NCCL | **混了 MSCCL++ 执行空间**。作者也承认。不是纯热插拔的收益 |
| 动态 2.46× | 图过时之后再换 | 这才是论文主价值 |
| JCT 1.84× | fail-slow 端到端 | 通信不是 step 的 100%，e2e 被算力摊薄 |
| MoE EP32 约 1.07× | 训练效率 | 倾斜换档，不是 EP 库重做；100k step 大约省 8 小时墙钟 |

稳态 1.61× 拿来跟 SyCCL/OptCCL 比「算法质量」不公平；动态 2.46× 拿来跟 PReCCL 比「同一张图调字节」也不公平。

---

## 5. 在 2024–2026 栈上的位置

```text
综合器（作业前 / 近线）     TE-CCL / SyCCL / OptCCL / ForestColl
        ↓ 出候选图
运行时换图                 Theseus          ← 本篇
运行时换字节               PReCCL
执行器                     MSCCL++ / ResCCL
EP 库（不走标准集体）       UBEP / DeepEP / SwiftEP
```

阿里云同时出现在 Theseus 和 PReCCL 作者里，分工清楚：SyCCL 一类综合器出图，Theseus 热接住，PReCCL 在图还对时切 VT。缺的那一环仍是 **运行时增量重综合**——漂移发生后当场长出合法新图。Theseus 的遥测–决策–热插拔环是现成的，OptCCL 的最优综合还是离线的。

和 FAST：Theseus 的高倾斜 AlltoAll 档是预先做好的 FAST-lite 形图，不是 64 GPU 上 221µs 现场 Birkhoff。DeepEP 类 P2P 没有标准 alltoallv 矩阵，Theseus 换不了。

---

## 6. 落到 950 / HCCL

HCCL 的 `HCCL_ALGO` 自适应仍然是启动或首次展开时选 Ring/NHR/Pipeline/Mesh，没有热插拔。要对齐 Theseus，需要：

- 多套 **CCU Mission 模板 / AI CPU 编排图** 作为候选（不是多套 `tile→VT`）；
- 集群属性（UB Port 健康、UBoE 口限速、AlltoAll 倾斜）进选择策略；
- 各 rank agreement 之后 **delta 换模板**，下一 CCT 再生效。

950 上 Theseus 值得预置的候选，和 PReCCL 字节切分正交：

| 候选 | 何时切过来 |
| --- | --- |
| 单 UB Port 挂死 → IO Die 转发 / 少 Port 的 Ring | 结构变了，迁字节不够 |
| 一条 UBoE 0% → Pipeline 只走另一口，或降到 AI CPU 4 VT 图 | 同左 |
| AlltoAll 分组 FullMesh 高低并发两档 | 倾斜跨阈值 |
| 8 卡 FullMesh（Ascend C CCU 上限）vs 128 卡 NHR | 域大小变了 |

不要用 Theseus 做「同一 Jetty 集合上慢 VT 少吃字节」——那是 PReCCL，更轻，也不需要预计算图。首期：PReCCL 切字节；Theseus 只预留换模板接口，候选先做「单口故障中继」一张图。
