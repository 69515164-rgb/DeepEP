# 昇腾 950 PReCCL 方案时序图文件

源文件（可再渲染）：

| 文件 | 内容 |
| --- | --- |
| `a_component_layers.mmd` / `.svg` | 图 A：AI CPU 七层组件 |
| `d1_hop_layers.mmd` / `.svg` | 图 D1：算法拍 i / 拍间 / 拍 i+1 × L0～L6 |
| `d2_hop_sequence.mmd` / `.svg` | 图 D2：同一次 AllReduce 内拍间时序 |
| `c_epoch_sequence.mmd` / `.svg` | 图 C：跨集体 epoch 时序（对照） |

| `fused_plan_load.mmd` | 融合 plan.bin：离线生成 → GetWorkspaceSize 按签名加载 |

演示文稿：`docs/昇腾950_PReCCL运行时切分方案.pptx`（图已按层次重绘进幻灯片）。
重生成：`python3 tools/generate_ascend950_preccl_ppt.py`
