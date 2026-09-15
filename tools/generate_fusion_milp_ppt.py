#!/usr/bin/env python3
"""Leadership deck: fused MM+AllReduce auto-generation, with background callouts."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parents[1] / "docs" / "融合算子自动生成_MM_AllReduce.pptx"

NAVY = RGBColor(10, 24, 48)
BLUE = RGBColor(38, 108, 255)
CYAN = RGBColor(37, 200, 221)
GREEN = RGBColor(33, 181, 115)
ORANGE = RGBColor(255, 153, 51)
RED = RGBColor(225, 72, 72)
WHITE = RGBColor(248, 250, 253)
LIGHT = RGBColor(229, 236, 245)
MID = RGBColor(130, 148, 174)
DARK = RGBColor(28, 39, 57)
PALE = RGBColor(19, 49, 83)
NOTE = RGBColor(58, 48, 22)
FONT = "Droid Sans Fallback"


def add_text(slide, text, x, y, w, h, size=20, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill, radius=False, line=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line or fill
    return s


def pill(slide, text, x, y, w, color=BLUE, size=12):
    rect(slide, x, y, w, 0.34, color, True)
    add_text(slide, text, x, y, w, 0.34, size, WHITE, True, PP_ALIGN.CENTER)


def card(slide, title, body, x, y, w, h, accent=BLUE, title_size=15, body_size=13):
    rect(slide, x, y, w, h, DARK, True, RGBColor(51, 69, 96))
    rect(slide, x, y, 0.07, h, accent)
    add_text(slide, title, x + 0.2, y + 0.1, w - 0.32, 0.34, title_size, WHITE, True)
    add_text(slide, body, x + 0.2, y + 0.48, w - 0.36, h - 0.6,
             body_size, LIGHT, False, valign=MSO_ANCHOR.TOP)


def note(slide, text, x, y, w, h):
    """Yellow callout for background knowledge."""
    rect(slide, x, y, w, h, NOTE, True, ORANGE)
    add_text(slide, "背景  " + text, x + 0.16, y + 0.06, w - 0.28, h - 0.12,
             12, LIGHT, False, valign=MSO_ANCHOR.TOP)


def bullets(slide, items, x, y, w, h, size=15, color=LIGHT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + item
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(8)
    return box


def base_slide(prs, title, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    if kicker:
        add_text(slide, kicker, 0.6, 0.24, 10, 0.24, 10, CYAN, True)
    add_text(slide, title, 0.6, 0.48, 12.1, 0.5, 24, WHITE, True)
    rect(slide, 0.6, 1.04, 1.0, 0.045, BLUE)
    add_text(slide, str(len(prs.slides)), 12.3, 7.1, 0.45, 0.2, 8, MID,
             align=PP_ALIGN.RIGHT)
    return slide


def title_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY
    rect(s, 0, 0, 13.333, 0.12, BLUE)
    pill(s, "技术方案汇报", 0.7, 0.55, 1.7, CYAN, 12)
    add_text(s, "融合算子自动生成\nMM + AllReduce", 0.7, 1.15, 7.8, 1.7, 32, WHITE, True)
    add_text(s, "把集合通信求解器扩展到「边算边传」：\n运行时只认计算→通信依赖；估时只为离线选型，不进内核当闹钟。",
             0.72, 3.1, 7.5, 0.95, 16, LIGHT)
    rect(s, 8.65, 1.2, 3.95, 4.55, DARK, True, RGBColor(51, 69, 96))
    for i, (t, c) in enumerate([
        ("1  计算怎么切、谁先算完", BLUE),
        ("2  写完才允许发给谁", ORANGE),
        ("3  重叠靠事件，不靠闹钟", GREEN),
    ]):
        pill(s, t, 8.95, 1.7 + i * 1.2, 3.35, c, 13)
    add_text(s, "面向非本领域听众  |  2026-09\n对应 docs/fusion_milp_input.md",
             0.72, 6.55, 6.5, 0.55, 12, MID)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_slide(prs)

    s = base_slide(prs, "今天要拍的板：运行时认依赖，不认启动时刻", "结论先行")
    card(s, "建议采用", "计算编译器给：tiling + 谁生产哪一包\n运行时：这块写完才允许发这一包\n重叠靠事件自然发生\n\n估时只在离线比较「几种打包/VT」时用\n用完扔掉微秒数，不写进内核",
         0.6, 1.35, 6.05, 3.45, GREEN, 16, 15)
    card(s, "不建议", "不要让通信求解器去选矩阵乘切块\n不要把「第 12 微秒启动」写进算子\n访存争抢会让闹钟又偏又可能早发\n\n写完就发且 Ring 已定时，连 MILP 都可以不做",
         6.9, 1.35, 5.85, 3.45, RED, 16, 14)
    note(s, "MILP = 混合整数线性规划，工业界用来离线搜「谁发给谁、走哪条路」。代表：TACCL、TE-CCL、SyCCL。它们输出的日程下降到库里也是等 CQE，不是墙上时钟。融合只多挂一条本地写完事件。",
         0.6, 5.05, 12.15, 1.45)

    s = base_slide(prs, "业务场景：一张卡算不完的线性层，要切到多卡", "问题从哪来")
    add_text(s, "大模型里最常见的一层：Y = X W     （矩阵乘，也叫 GEMM / Linear）",
             0.7, 1.25, 12, 0.4, 16, WHITE, True)
    for i, (t, b, c) in enumerate([
        ("单卡", "X、W 都在一张 NPU 上\n算出完整的 Y\n没有卡间通信", MID),
        ("行并行（张量并行）", "每张卡只持有 W 的一竖条\n各卡算出 Y 的一块「半成品」\n必须 AllReduce 才能得到完整 Y", ORANGE),
        ("我们要生成的融合算子", "不要等全部半成品算完再通信\n一块半成品写出来就往外传\n计算和通信在时间上重叠", GREEN),
    ]):
        card(s, t, b, 0.6 + i * 4.15, 1.8, 3.95, 2.55, c, 16, 14)
    note(s, "张量并行（TP）：把一个很大的矩阵乘拆到多张卡上算。行并行时，每张卡的局部结果要加在一起——这就是 AllReduce。列并行对应的是 AllGather，不要和 AllReduce 混用。",
         0.6, 4.6, 12.15, 1.85)

    s = base_slide(prs, "背景：集合通信是什么，AllReduce 做什么", "名词")
    card(s, "集合通信", "一群卡按固定语义交换数据。\n常见：AllReduce / AllGather /\nReduceScatter / Broadcast。\n通信库：NCCL（NVIDIA）、HCCL（昇腾）。",
         0.6, 1.3, 4.0, 3.15, BLUE, 16, 14)
    card(s, "AllReduce", "每张卡有一份同形状的数。\n结束后每张卡都拿到\n「所有卡对应位置加总」的结果。\n训练里用来把切开来的线性层拼回去。",
         4.8, 1.3, 4.0, 3.15, CYAN, 16, 14)
    card(s, "算法长什么样", "常见 Ring（围成环，一拍传一块）\n和 HD（对半加减，对端每拍换）。\n本方案默认把这种邻居关系冻住，\n求解器不换算法，只排何时发。",
         9.0, 1.3, 3.75, 3.15, GREEN, 16, 13)
    note(s, "可以把 AllReduce 想成「开会时每人报一个数，最后每人手里都是总和」。Ring 是按座位传纸条；HD 是按 1、2、4… 号距离对喊。求解器若连座位表一起搜，规模会爆，所以座位表（算法）建议给定。",
         0.6, 4.7, 12.15, 1.8)

    s = base_slide(prs, "背景：矩阵乘不能整块算，必须切成 tile", "名词  /  tiling")
    add_text(s, "Y 很大，片上缓存放不下。编译器把 Y 切成许多小矩形，一块一块算完再写出。",
             0.7, 1.22, 12, 0.38, 15, LIGHT)
    for i, (t, b, c) in enumerate([
        ("BM × BN", "输出 Y 上一个小矩形的高和宽。\n这是能交给通信的最小「成品块」，\n叫 C-tile。", BLUE),
        ("BK", "沿 K 方向一次吃多长。\n一个 C-tile 要折好几次 K\n才算完、才能写出去。", CYAN),
        ("循环顺序 / wave", "先算哪一行、哪一列；\n同一拍片上能并几块。\n这决定「第几块何时出炉」。", GREEN),
    ]):
        card(s, t, b, 0.6 + i * 4.15, 1.7, 3.95, 2.55, c, 16, 14)
    note(s, "tiling = 切块策略，是计算编译器（Triton / CUTLASS / AscendC）的主业。它只回答「空间上切成哪些块」，不回答「通信走哪条网线」。选错切块，卡的算力都喂不饱，这不是通信求解器能替代的。",
         0.6, 4.5, 12.15, 1.95)

    s = base_slide(prs, "背景：计算流图 = 这些 tile 按什么顺序、花多长时间", "名词  /  计算流图")
    card(s, "节点", "每一个 C-tile（或同一拍的一组，叫 wave）。\n必须和 tiling 切出来的块是同一套名字，\n不能另编一套编号。",
         0.6, 1.3, 4.0, 2.7, BLUE, 16, 14)
    card(s, "边 = 依赖", "同一块输出沿 K 必须按序累加；\nkernel 里已经排好的发射顺序。\n本方案要求这是「已经排好的计划」，\n不是还可以随便重排的草稿。",
         4.8, 1.3, 4.0, 2.7, CYAN, 16, 13)
    card(s, "dur = 耗时", "这块占计算单元多久。\n由切块大小和芯片峰值估出来，\n当已知数，不当求解变量。\n有了顺序和 dur，就能推出完成时刻。",
         9.0, 1.3, 3.75, 2.7, GREEN, 16, 13)
    note(s, "流图可以想成工厂流水线的工单：第 1 块 10 微秒、第 2 块接着做……。没有「谁先谁后」和「做多久」，就不知道半成品何时能装箱发货，通信求解器无法开工。",
         0.6, 4.25, 12.15, 2.15)

    s = base_slide(prs, "关键澄清：流图不会发明耗时，只负责串成时间轴", "实例  /  dur 从哪来")
    card(s, "第 1 步  代价模型填 dur", "计算编译器拿切块尺寸和芯片峰值\n算出「这一块占 Cube 多久」\n写进节点，当成已知数\n\n三种来源均可：屋顶线公式 / 按 K 片估 / kernel 打点实测",
         0.6, 1.28, 6.05, 3.35, ORANGE, 16, 14)
    card(s, "第 2 步  流图只做加法", "已经排好谁先谁后\nstart[下一块] = 上一块结束时刻\nfinish = start + dur\n\n没有第 1 步的微秒数，空流图给不出任何时间",
         6.9, 1.28, 5.85, 3.35, GREEN, 16, 14)
    note(s, "后面 3 页用同一套手算数字：2×2 个 C-tile，每块 10 微秒。峰值取整是为了能口算，不是芯片手册；换 950 的峰值表，步骤不变。",
         0.6, 4.85, 12.15, 1.55)

    s = base_slide(prs, "手算实例：本 rank 上的一块局部矩阵乘", "实例  /  形状")
    add_text(s, "Y_partial = X @ W，FP16，K 已被张量并行切开    M=256  N=256  K=1024",
             0.7, 1.22, 12, 0.35, 15, WHITE, True)
    card(s, "切块 tiling", "BM=BN=128，BK=256\nC-tile 网格 2×2，共 4 块\n每块内部沿 K 折 4 次才写完\nwave=1：一次只算一块，四块串行",
         0.6, 1.7, 6.05, 2.85, BLUE, 16, 14)
    card(s, "四个输出块怎么摆", "        N0        N1\n   M0   C00       C01\n   M1   C10       C11\n\n每块写出 128×128 的半成品\n必须整块写完才能进 AllReduce",
         6.9, 1.7, 5.85, 2.85, CYAN, 16, 14)
    note(s, "背景：大矩阵放不进片上缓存，所以切成 C-tile。本例故意切成 4 块，方便一页写完；真实线性层可能是几百块，公式相同。",
         0.6, 4.75, 12.15, 1.65)

    s = base_slide(prs, "一步算出每块 dur = 10 微秒", "实例  /  代价模型")
    card(s, "一块的算术量和搬运量", "乘法次数 FLOPs = 2×128×128×1024 = 3355 万次\n读 X 262KB + 读 W 262KB + 写 Y 32KB\n合计搬运 557KB",
         0.6, 1.25, 6.05, 2.55, BLUE, 16, 14)
    card(s, "屋顶线：算得动还是搬得动", "示意峰值 4.2 TFLOPS → 算要 8.0 μs\n示意带宽 1.6 TB/s → 搬要 0.35 μs\n取较慢的一边 = 8.0 μs（算力墙）\n再加灌流水 1 μs + 排空 1 μs → dur=10 μs",
         6.9, 1.25, 5.85, 2.55, ORANGE, 16, 13)
    add_text(s, "四块形状相同  →  dur[C00]=dur[C01]=dur[C10]=dur[C11]=10 μs   （写进流图，MILP 当常数）",
             0.7, 4.0, 12, 0.4, 15, WHITE, True)
    note(s, "屋顶线：把「算」和「搬」各估一次时间，谁慢听谁的。编译器也可以按 4 次 K 片估（1+4×2+1=10），或对 kernel 打点回填。三种都不进通信求解器。",
         0.6, 4.55, 12.15, 1.85)

    s = base_slide(prs, "按发射顺序串起来，才得到「第几块几点钟出炉」", "实例  /  时间轴")
    add_text(s, "issue 序：C00 → C01 → C10 → C11     公式：下一块开始 = 上一块结束",
             0.7, 1.2, 12, 0.32, 15, LIGHT)
    # mini timeline table as cards
    for i, (name, st, fn, rd) in enumerate([
        ("C00", "0", "10", "12"),
        ("C01", "10", "20", "22"),
        ("C10", "20", "30", "32"),
        ("C11", "30", "40", "42"),
    ]):
        x = 0.6 + i * 3.15
        rect(s, x, 1.6, 2.95, 2.35, DARK, True, BLUE if i == 0 else CYAN)
        pill(s, name, x + 0.85, 1.75, 1.2, BLUE if i % 2 == 0 else CYAN, 14)
        add_text(s, f"开始 {st} μs\n写完 {fn} μs\nready {rd} μs",
                 x + 0.15, 2.25, 2.65, 1.45, 15, LIGHT, False, PP_ALIGN.CENTER)
    note(s, "ready = 写完 + 2 μs（写到通信可见的固定延迟）。这 12/22/32/42 只是离线估算，用来比较打包方案。运行时是 C00 写完就发，不是到点闹钟。见下一页。",
         0.6, 4.2, 12.15, 2.2)

    s = base_slide(prs, "运行时只要先后依赖，不要通信启动时刻", "拍板  /  抖动与争抢")
    card(s, "正确性：一条边就够", "C-tile 写完（事件）→ 才允许 post 这一包\n下一块继续算，上一包已经在路上\n重叠是事件调度的自然结果\n\n不需要、也不该给出「第 12 微秒启动」",
         0.6, 1.28, 6.05, 3.45, GREEN, 16, 14)
    card(s, "估时不准是必然，也没关系", "边算边传会抢同一套访存\n规划时的 dur 一定会被带偏\n再估也估不圆，因为一发通信 dur 就变\n\n拆开循环：不用启动点，用完成事件\n晚到就晚发，结果仍对",
         6.9, 1.28, 5.85, 3.45, ORANGE, 16, 14)
    note(s, "背景：集合通信库（NCCL/HCCL）本来就是「上一跳 CQE 到了再发下一跳」，没有墙上时钟。融合只是多挂一条「本地这块矩阵乘写完」。定时器 post 在争抢下既慢又可能早发，属于错误下降。",
         0.6, 4.95, 12.15, 1.5)

    s = base_slide(prs, "估时只给离线选型用，用完扔掉微秒数", "拍板  /  两层分工")
    card(s, "策略已钉死：可以没有 dur", "冻 Ring、一块就是一包、写完就发、VT 已定\n没有选择可搜 → 不要 MILP，不要启动时刻\n编译器只给 tiling +「谁生产哪一包」这条边",
         0.6, 1.28, 6.05, 2.85, BLUE, 16, 14)
    card(s, "还要选结构：才估时跑求解器", "几块打一包、几条 VT、走哪条路\n用估时比较谁的总时间短\n交出的是打包和映射，不是闹钟\n下降：wait(写完事件); post_wqe",
         6.9, 1.28, 5.85, 2.85, CYAN, 16, 14)
    note(s, "一句话：依赖保证对；估时只用来选方案；时钟不进内核。访存争抢的残差交给硬件仲裁；若还要运行时再切分，走自适应 VT，而不是改启动时刻。",
         0.6, 4.35, 12.15, 2.05)

    s = base_slide(prs, "背景：现有求解器本来只干一件事——编通信", "名词  /  MILP 综合器")
    add_text(s, "输入：卡怎么连、每条链路多慢、要做哪种集合、数据切成几块。\n输出：一张时刻表——谁在何时把哪一块发给谁。",
             0.7, 1.22, 12, 0.7, 16, LIGHT)
    for i, (t, b, c) in enumerate([
        ("TACCL", "把拓扑和集体语义写成整数规划\n搜通信日程", BLUE),
        ("TE-CCL / OptCCL", "更细的时间或代价模型\n仍只搜通信", CYAN),
        ("SyCCL", "用对称把大规模切成小问题\n仍然不碰矩阵乘切块", GREEN),
    ]):
        card(s, t, b, 0.6 + i * 4.15, 2.1, 3.95, 2.15, c, 16, 14)
    note(s, "这些求解器假设「数据一开始就在」。融合场景里，数据是矩阵乘一块块生产出来的——所以我们只多加一条规矩：这块还没算完，不准发。求解器本体不用推倒重来。",
         0.6, 4.55, 12.15, 1.9)

    s = base_slide(prs, "要解决的低效：算完再传 vs 边算边传", "动机")
    card(s, "现在常见（两阶段）", "全部 C-tile 算完\n→ 再启动一次 AllReduce\n时间 ≈ 计算 + 通信\n卡在通信时计算单元空转",
         0.6, 1.3, 5.9, 3.05, RED, 16, 15)
    card(s, "融合算子（目标）", "第 1 块写出来就开始传\n后面的块边算边传\n时间 ≈ max(计算, 通信) 量级\n这就是 MM+AllReduce 融合",
         6.85, 1.3, 5.9, 3.05, GREEN, 16, 15)
    note(s, "「融合算子」不是把两个 API 写成一个函数名就完了。关键是时间上重叠，且重叠必须合法：没生产出来的数不能提前进 AllReduce，否则结果是错的。",
         0.6, 4.6, 12.15, 1.8)

    s = base_slide(prs, "为什么不让求解器连切块一起搜", "分工理由")
    for i, (t, b, c) in enumerate([
        ("搜的东西不一样", "通信变量：哪块数据走哪条链路。\n切块变量：小矩形长宽、流水级数。\n乘在一起，方程规模爆炸。", RED),
        ("快慢模型不一样", "链路用「启动延迟 + 每字节时间」。\n切块快慢取决于计算单元和内存层次。\n通信求解器没有一份可信的切块耗时表。", ORANGE),
        ("改切块等于改实现", "切块、累加顺序、尾处理（bias/激活）\n影响数值。这是计算编译器的职责，\n通信求解器改了就是改 GEMM。", CYAN),
    ]):
        card(s, t, b, 0.6 + i * 4.15, 1.3, 3.95, 3.15, c, 16, 13)
    note(s, "如果以后要试几种切块，放在外层：计算编译器给出 3～5 套候选，每套跑一次通信求解器，比较总完成时间。切块仍然不进求解器内部。",
         0.6, 4.7, 12.15, 1.7)

    s = base_slide(prs, "方案：编译器给两份，求解器只编通信", "架构")
    steps = [
        ("计算编译器", "① tiling\n② 计算流图", BLUE),
        ("机械推导", "produce 边必须有\ndur 仅离线可选", ORANGE),
        ("通信 MILP", "离线搜结构\n打包 / 链路 / VT", GREEN),
        ("运行时", "等写完事件再发\n无启动时刻表", CYAN),
    ]
    for i, (t, b, c) in enumerate(steps):
        x = 0.55 + i * 3.2
        rect(s, x, 1.35, 2.95, 2.55, DARK, True, c)
        pill(s, t.replace("\n", "  "), x + 0.15, 1.55, 2.65, c, 13)
        add_text(s, b, x + 0.18, 2.15, 2.6, 1.5, 15, LIGHT, False, PP_ALIGN.CENTER)
        if i < 3:
            add_text(s, "→", x + 2.85, 2.2, 0.4, 0.5, 22, MID, True, PP_ALIGN.CENTER)
    bullets(s, [
        "求解器可以决定：走哪条链路、哪个 VT、由哪个计算完成事件触发",
        "求解器不可以决定：切块长宽、计算顺序、以及内核里的通信启动时刻",
        "写完就发且 Ring/VT 已定时：连 MILP 都可以不做，只要 produce 边",
    ], 0.7, 4.2, 12, 1.55, 15)
    note(s, "和上一页合起来：扩展的是「数据必须先算完再发」这条边。估时只为离线比较结构，不写进内核当闹钟。",
         0.6, 5.85, 12.15, 1.05)

    s = base_slide(prs, "两份输入分别回答什么", "编译器接口")
    card(s, "tiling = 空间怎么切", "Y 切成多少个 C-tile、每块多大\n沿 K 要折几次才写完一块\n循环先走哪一维、一拍并几块\n写到内存后过多久通信才看得见\n\n没有「第 3 块几点钟出炉」",
         0.6, 1.28, 6.05, 3.55, BLUE, 16, 14)
    card(s, "计算流图 = 这些块何时算完", "节点 = 上面那些 C-tile（同一套编号）\n边 = 已排好的计算顺序\n每节点做多久\n\n没有「几块拼成一次 AllReduce 的一包」\n那是切块网格上的简单合并规则",
         6.9, 1.28, 5.85, 3.55, GREEN, 16, 14)
    note(s, "只有 tiling：知道切成哪些块，不知道何时有货。只有流图：知道何时算完哪块，不知道通信按几块打一包。所以两份都要，但各自只答一件事。",
         0.6, 5.05, 12.15, 1.4)

    s = base_slide(prs, "不必再要第三份：有货时间和资源占用都能推出来", "推导  /  避免重复输入")
    add_text(s, "上一轮讨论里出现过 produce、占用时间轴。它们不是编译器另交的图，而是下面两个函数。",
             0.7, 1.22, 12, 0.4, 14, LIGHT)
    card(s, "何时有货 ready", "约定：连续若干个 C-tile 合成通信的一包\n（默认同一块 C-tile 就是一包，最简单）\n这一包的 ready = 其中最晚写完的那块\n再加一个固定的写出延迟",
         0.6, 1.7, 6.05, 2.85, ORANGE, 16, 14)
    card(s, "计算占了哪些资源", "每个 C-tile 带一份固定配额\n（占几条计算单元、吃多少内存带宽）\n按流图的起止时间扫一遍\n就得到每个时刻还剩多少给通信",
         6.9, 1.7, 5.85, 2.85, CYAN, 16, 14)
    note(s, "若让编译器再手填一张「谁生产哪一包」的表，容易和切块网格不一致。规定「只能按 C-tile 整块合并」，表就是一个除法，检查对齐即可。",
         0.6, 4.8, 12.15, 1.65)

    s = base_slide(prs, "求解器看见什么、交出什么", "合同一页纸")
    card(s, "吃进去", "原来就有的：网络怎么连、链路多慢、\nAllReduce、Ring/HD 怎么传\n\n新加上的：tiling、计算流图\n（以及切几块打一包、写出延迟两个数）",
         0.6, 1.28, 6.05, 3.35, BLUE, 16, 14)
    card(s, "交出来", "结构：哪一包、走哪条路、挂在哪个写完事件上\n保证：没有事件就绝不发\n\n不交：墙上时钟启动点、新的切块、新的计算顺序",
         6.9, 1.28, 5.85, 3.35, GREEN, 16, 14)
    note(s, "离线目标函数可以用估时的总完成时间来比方案。比完之后微秒数丢掉，只把事件边和映射写进算子。",
         0.6, 4.9, 12.15, 1.55)

    s = base_slide(prs, "一张图看完数据怎么走", "直觉")
    phases = [
        ("切", "编译器把 Y\n切成 C-tile", BLUE),
        ("算", "按流图一块块算\n写完才算有货", CYAN),
        ("叠", "写完事件一到\n立刻 post 这一包", ORANGE),
        ("齐", "所有包加总完成\n每张卡都有完整 Y", GREEN),
    ]
    for i, (n, b, c) in enumerate(phases):
        x = 0.7 + i * 3.15
        rect(s, x, 1.4, 2.85, 2.35, DARK, True, c)
        pill(s, n, x + 0.85, 1.6, 1.15, c, 16)
        add_text(s, b, x + 0.15, 2.15, 2.55, 1.35, 15, LIGHT, False, PP_ALIGN.CENTER)
        if i < 3:
            add_text(s, "→", x + 2.75, 2.25, 0.45, 0.45, 22, MID, True)
    bullets(s, [
        "「切」和「算」的规则来自计算编译器，开会前就定死",
        "「叠」是挂事件：写完才发。不要下降成启动时刻表",
        "任何试图在「叠」里改「切」的方案，都退回上一页的不建议",
    ], 0.7, 4.05, 12, 1.45, 16)
    note(s, "验收很好懂：融合后的总时间，应明显短于「先算完再 AllReduce」；且数值与两阶段结果一致（加法顺序若要求确定，邻居关系必须冻住）。",
         0.6, 5.6, 12.15, 1.3)

    s = base_slide(prs, "明确不做，避免范围膨胀", "边界")
    rows = [
        ("不搜切块长宽", "交给 Triton / AscendC / 手工 kernel"),
        ("不重排计算顺序", "改顺序就是改矩阵乘实现"),
        ("不把激活挪过 AllReduce", "会改数值，必须事先钉死在前或在后"),
        ("不把启动时刻写进内核", "运行时只认写完事件；估时不准是必然"),
        ("不把「试几套切块」塞进同一个方程", "外层试 3～5 套，每套单独求一次通信"),
    ]
    for i, (t, b) in enumerate(rows):
        y = 1.28 + i * 0.85
        pill(s, t, 0.65, y, 3.35, RED if i < 2 else ORANGE, 13)
        add_text(s, b, 4.2, y, 8.4, 0.4, 16, LIGHT)

    s = base_slide(prs, "建议与下一步", "决策")
    add_text(s, "立项口径：运行时只加「写完才发」的依赖；估时仅离线选打包/VT。切块仍由计算编译器给定，启动时刻不进内核。",
             0.7, 1.25, 12, 0.7, 17, WHITE, True)
    for i, (n, t, b, c) in enumerate([
        ("01", "冻结接口", "tiling + produce 边\n必给；dur 仅离线可选", BLUE),
        ("02", "最小内核改动", "挂写完事件再 post\n禁止定时器启动", CYAN),
        ("03", "先冻通信算法", "Ring/HD、写完就发\n无选择时不做 MILP", GREEN),
        ("04", "用总时间验收", "对比两阶段基线\n数值必须一致", ORANGE),
    ]):
        x = 0.6 + i * 3.15
        rect(s, x, 2.2, 2.95, 2.55, DARK, True, c)
        pill(s, n, x + 0.18, 2.38, 0.55, c, 12)
        add_text(s, t, x + 0.2, 2.95, 2.55, 0.4, 16, WHITE, True)
        add_text(s, b, x + 0.2, 3.45, 2.55, 1.05, 13, LIGHT)
    note(s, "详细合同见 docs/fusion_milp_input.md 第 11 节。切块归计算编译器；运行时只认写完事件；时钟不进内核。",
         0.6, 5.05, 12.15, 1.4)

    s = base_slide(prs, "附录：一分钟对照表", "备忘  /  可留在材料最后")
    rows = [
        ("GEMM / MM / Linear", "矩阵乘 Y = XW，大模型最重的计算之一"),
        ("AllReduce", "多卡各持一份数，结束时每卡都有总和"),
        ("tiling", "把大矩阵切成小矩形（C-tile）以便放进片上缓存"),
        ("计算流图", "这些 C-tile 的发射顺序；耗时 dur 由代价模型事先填入"),
        ("屋顶线 / dur", "算的时间和搬的时间取较慢边；本例每块 dur=10 μs"),
        ("MILP 综合器", "离线搜通信结构；输出下降成事件边，不是闹钟"),
        ("融合算子", "同一段内核里边算矩阵乘边做 AllReduce"),
        ("produce 边", "这些 C-tile 写完才允许发对应一包（运行时硬约束）"),
        ("ready 估时", "离线用的数字；运行时不读。不准是必然，只影响是否选到最优结构"),
        ("Ring / HD", "AllReduce 的两种经典传法；本方案建议给定、不搜"),
    ]
    for i, (t, b) in enumerate(rows):
        y = 1.16 + i * 0.54
        add_text(s, t, 0.7, y, 3.3, 0.46, 12, CYAN, True)
        add_text(s, b, 4.1, y, 8.5, 0.46, 13, LIGHT)

    prs.core_properties.title = "融合算子自动生成 MM+AllReduce"
    prs.core_properties.subject = "领导汇报：计算编译器给定切块与流图，MILP 只编通信"
    prs.core_properties.author = "Cursor"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
