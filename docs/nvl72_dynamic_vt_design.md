# NVL72 多 QP 动态 VT 调度方案

## 1. 汇报目标

- 在 ConnectX-8 配合非 Spectrum 以太交换机、无法使用逐包自适应路由与乱序接收时，缓解 ECMP 冲突、链路拥塞和接收端 incast。
- 借鉴 PReCCL，将每个 `channel` 与一个 QP 一一映射为独立 Virtual Topology。
- 保持 QP 内有序，在 collective 边界动态调整各 VT 的字节份额。
- 优先 rail 内换路径，只有 rail 级拥塞或目的端 incast 时才跨 rail，并通过本托盘 NVLink/PXN 到达目标 GPU。

## 2. 核心结论

- 方案可行，但它模拟的是流级多路径和动态加权，不替代乱序接收或选择性重传。
- 网卡池化边界是一个 compute tray：4 GPU、4 CX8、4 rail、单 OS。
- `VT = channel = QP`，推荐每个 rail 2 个 VT，总计 8 个 VT。
- 调度采用两级动作：
  1. rail 内：在不同 ECMP 路径的 QP 间调整份额。
  2. 跨 rail：目的 rail 整体拥塞或 incast 时，改用其它 NIC 入站，再经 NVLink 到目标 GPU。
- 不在单次 collective 中途切换；下一 epoch 全 rank 一致生效。

## 3. 背景与约束

- Spectrum-X 的逐包自适应路由依赖交换机与 SuperNIC 端到端协同。
- 非 Spectrum 环境下，每个 QP 必须保持五元组稳定，由 ECMP 固定映射到单一路径。
- 增加 QP 可以增加路径熵，但无法保证不与其他作业的大象流冲突。
- 单 QP 拥塞会拖慢对应 channel；collective 完成时间由最慢 VT 决定。
- 多 QP 只能隔离拥塞和丢包影响，不能实现选择性重传。

## 4. NVL72 拓扑与池化边界

- 每个 NVL72 机柜包含 18 个 compute tray，每个 tray 有 4 GPU 和 4 张计算网卡。
- 72 GPU 处于同一 NVLink 域，但 18 个 tray 分属不同 OS。
- PXN 只能在单 OS 内使用，因此默认池化范围只能是本 tray 的 4 张 CX8。
- 相同 GPU/NIC 索引连接相同 rail；跨 rail 才能绕开目的 rail 的 leaf 下行 incast。
- 机柜级借用其它 tray 的 NIC 需要软件中继，不作为性能路径，只保留为故障备份研究项。

## 5. VT 与 QP 映射

推荐配置：

| VT | Rail | QP | ECMP 路径 |
|---|---:|---:|---|
| VT0 | 0 | QP0 | path 0A |
| VT1 | 0 | QP1 | path 0B |
| VT2 | 1 | QP2 | path 1A |
| VT3 | 1 | QP3 | path 1B |
| VT4 | 2 | QP4 | path 2A |
| VT5 | 2 | QP5 | path 2B |
| VT6 | 3 | QP6 | path 3A |
| VT7 | 3 | QP7 | path 3B |

- 每个 QP 固定源端口，保证 QP 内有序。
- 初始化时通过交换机流表或主动探测确认实际 ECMP 上行。
- 同 rail 的两个 QP 应落到不同上行；无法正交时不把它们视为两个独立容量资源。

## 6. 两级动态调度

### L1：rail 内调整

- 触发条件：同 rail 的部分 VT 慢、部分 VT 正常。
- 判断：更可能是某条 leaf-to-spine 上行与其它作业发生 ECMP 冲突。
- 动作：只在同 rail 的 QP/VT 之间迁移下一 epoch 的字节份额。
- 优点：不增加 NVLink 绕路，不占用其它 rail 的 NIC。

### L2：跨 rail 绕路

- 触发条件：同 rail 所有 VT 同时恶化，交换机遥测显示目的 leaf 下行或目的 NIC 入口拥塞。
- 动作：将部分份额切至其它 rail 对应 VT。
- 路径：发送 GPU 经本 tray NVLink/PXN 到 NIC-k，网络走 rail-k，接收端 NIC-k 收到后经 NVLink/PXN 送往目标 GPU。
- 约束：限制单 GPU 同时借用的异地 NIC 数量，避免侵占兄弟 GPU 带宽。

## 7. 遥测与判因

### 通信库主信号

- 每 VT 统计发送 FIFO stall、接收 FIFO stall、完成字节数和完成时间。
- 使用在线标定模型估算每 VT 的有效带宽和预计完成时间。
- 遥测随 collective 同步消息携带，不增加独立控制连接。

### 可编程交换机辅助信号

- QP 对应的实际 ECMP 上行编号。
- leaf-to-spine 出口队列或 ECN 档位。
- 目的 leaf 下行和 NIC 端口队列档位。
- 交换机只负责观测和初始化路径校验，不对同一 QP 逐包改路。

### 判因规则

- 同 rail 内快慢分化：优先判为上行路径冲突，执行 L1。
- 同 rail 全部变慢、目的下行热：判为 rail 级 incast，执行 L2。
- 单 VT 连续超时：屏蔽该 VT，并在后续 epoch 探测恢复。

## 8. Epoch 一致性协议

1. 本 epoch 使用固定的 VT 字节权重完成 collective。
2. 各 rank 汇总同一 epoch 的 VT 遥测。
3. 所有 rank 使用确定性分配函数计算下一组权重。
4. 权重携带 epoch 编号，下一次 collective 同步生效。
5. 禁止中途迁移已提交到 QP 的字节。

控制稳定性：

- 每 VT 保留最小探测份额。
- 限制单 epoch 权重变化幅度。
- 设置拥塞进入和退出的不同阈值，避免抖动。
- 部分故障时屏蔽 VT；所有候选 VT 均失效才进入 communicator 恢复。

## 9. 分配算法

目标是使所有健康 VT 的预计完成时间尽量一致：

```text
输入：每个 VT 的有效带宽、固定开销、rail/path 标签、交换机拥塞档位
输出：下一 epoch 的字节权重

先在每个 rail 内按有效带宽分配；
若 rail 整体健康，维持 rail 总份额；
若 rail 级 incast，降低该 rail 总份额并迁移到冷 rail；
最后执行最小份额、变化限幅和容量上限约束。
```

- 不使用每个 VT 独立 AIMD，避免多个 VT 同时迁移形成振荡。
- 按 collective 类型分别配置：AllReduce/AllGather 采用连续字节区间；AlltoAll 还可启用本 tray NVLink 中继。

## 10. 实施边界

- 只优化跨 NVL72 机柜的 Ethernet collective；机柜内 NVLink/NVLS 不进入该控制环。
- PXN 代理必须位于同一 OS，因此 rank 和并行维度需要与 4 GPU tray 对齐。
- 不实现软件选择性重传，不修改 CX8 的可靠传输语义。
- 不进行逐包喷洒；已有 QP 的 ECMP 路径在生命周期内保持稳定。
- 小消息默认关闭动态重分配，建议从 64 MB 阈值开始测试。

## 11. 验证计划

### 第一阶段：路径与基础性能

- 验证固定源端口能稳定映射到指定 ECMP 上行。
- 测量 1/2/4/8 QP 时的吞吐、时延、PCIe/C2C、GPU SM 和 QP cache 开销。
- 验证本地 NIC 和跨 rail PXN 的额外成本。

### 第二阶段：可控故障

- 注入单上行拥塞，验证 L1 只在 rail 内收敛。
- 注入目的 leaf 下行 incast，验证 L2 跨 rail 绕路。
- 注入 QP 丢包、限速和链路失效，验证屏蔽与恢复。

### 第三阶段：集合通信与训练

- AllReduce、AllGather、ReduceScatter、AlltoAll，覆盖 64 MB 到多 GB。
- 32/72/144/576 GPU，覆盖单机柜和多机柜。
- 多租户训练，重点观察中位数、P95/P99 CCT、JCT 和 NIC 利用率。

## 12. 成功判据与风险

建议成功判据：

- 无拥塞稳态性能不低于 NCCL 基线 98%。
- 单上行冲突后 3–5 个 collective 内完成 rail 内收敛。
- rail 级 incast 时跨 rail 绕路显著降低 P95 CCT。
- 单 VT 故障时 collective 不挂死，维持主要带宽。

主要风险：

- 多 QP 实际哈希到相同上行，形成虚假路径多样性。
- communicator/rank 映射未与 tray 和 rail 对齐，导致 PXN 不可用。
- 交换机遥测延迟或噪声导致误判和振荡。
- PXN 借用兄弟 GPU/NIC 影响其它通信域或租户。
- channel 数过多导致小 chunk、GPU 资源和 NIC 上下文开销上升。

## 13. 决策建议

- 第一版采用 8 个 VT：每 rail 2 个 `channel/QP`。
- 采用两级控制：rail 内路径调整优先，跨 rail PXN 仅处理 rail 级 incast。
- 将交换机用于遥测与路径校验，不用于逐包改路。
- 先以大消息 AllReduce 和多租户冲突验证，再扩展 AlltoAll。
- 体系结构上预留 Theseus 式整调度热切换接口，但首期只实现 PReCCL 式字节重分配。
