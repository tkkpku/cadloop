"""cadloop.coupons — 标定试件库（T1 公差 / T2 材料）。

.. deprecated:: 2026-09-13
    **本模块已停用为「按需工具」，不再属于主流程。** 依据 2026-09-13 的
    标定策略决策：

    1. **先问，再测**：切片器孔补偿值是可**问出来**的已知信息（问老师），
       不需要打一堆阶梯孔试件去穷举反推；
    2. **敏感度判据**：材料参数估错会不会翻转结论？块状结构裕度 5–30 倍
       （`robustness` 实测：齿轮齿根 S=31.75、机架 S=5.33），参数无所谓；
       只有薄壁/细杆/柔性件裕度 <2.5 倍时才必须实测；
    3. 故采用**保守兜底值**（注塑值×0.8，层间再×0.5，SF=2.0）替代实测，
       通道在裕度落入敏感区时会**自动要求实测**——把「要不要打试件」
       从主观判断变成机械判据。

    试件几何（ASTM D638 Type V 等）保留在 ``coupons/on_demand/`` 供按需调用。
    物理删除需少爷批准（文件操作纪律）。

---

原模块说明（历史）
------------------
定位
----
本模块产出的是**工艺标定工具**，不是设计作品：
* T1 公差试件 → 回答「本机这台打印机，孔画多大才能装得进」
* T2 材料试件 → 回答「本机这台打印机，PLA 的实际 E / σy 是多少，层间折减多少」

它们的设计意图是**采集数据**，因此几何是标准化/扫描式的，没有造型自由度。

⚠️ 为什么必须单独立一套试件，而不是"边做作品边调"
------------------------------------------------
FDM 件的尺寸精度与力学性能**强依赖于打印参数**（层高、墙数、填充、方向、喷嘴磨损）。
参数一换，之前的数据全部失效。所以必须：

1. 先用固定参数打一套标定件；
2. 实测 → 写回 ``printenv.yaml`` → ``calibrated: true``；
3. 之后所有零件的结论才允许升级为"设计依据"。

在此之前，``PrintEnv.allow_conclusive_claims`` 为假，报告一律降级措辞。

标准依据
--------
* 拉伸试件：ASTM D638 **Type V**（总长 63.5 / 窄段 9.53×3.18 / 标距 7.62 /
  端部宽 9.53 / 厚 3.2 mm）。选 Type V 而非 Type IV 的关键原因：需打印**竖立版**
  测层间强度，Type IV 竖立为 115 mm 高细条（高厚比 36）极易被喷嘴推倒；
  Type V 仅 63.5 mm（高厚比 20），打印可行性显著更高且省料。
  尺寸来源：Shimadzu ASTM D638 规格表（仪器厂商官方页）。
* 三点弯曲：参考 ASTM D790 常用尺寸 80 × 10 × 4 mm（跨距 64 mm）。
* 公差试件：自设计扫描式阶梯孔/轴，无对应标准（这是工艺标定，不是材料测试）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import PrintEnv

# ⚠️ 弃用标记放在 ``from __future__`` 之后（见 rules_print.py 同类注释）
__deprecated__ = True
__deprecated_reason__ = "标定策略改为「先问再测 + 保守兜底 + 敏感度触发」；试件改为按需调用"
__usage__ = "按需：from cadloop.coupons import build_all, export_all（不再进主流程）"

# ==============================================================================
# 试件登记
# ==============================================================================


@dataclass
class Coupon:
    """一个标定试件的描述。"""

    name: str
    purpose: str            # 测什么
    build: Callable[[], Any]
    quantity: int
    orientation: str        # flat / upright
    measures: str           # 产出哪些数据字段
    note: str = ""
    pre_rotate_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """导出前预旋转。用于把"竖立打印"的试件直接转成可切片的摆放姿态，
    这样老师拿到 STL 不用自己想怎么摆。"""

    def __str__(self) -> str:
        return f"{self.name} ({self.orientation}, x{self.quantity}) — {self.purpose}"


# ==============================================================================
# 几何基元
# ==============================================================================
def _require():
    from .geom import require_build123d

    require_build123d()


def _arc_pts(cx: float, cy: float, r: float, th0: float, th1: float, n: int = 20):
    """圆弧离散点。

    参数化 ``P(θ) = (cx + r·sinθ, cy − r·cosθ)``，θ=0 对应圆弧最低点 ``(cx, cy−r)``。
    """
    out = []
    for i in range(n + 1):
        t = th0 + (th1 - th0) * i / n
        out.append((cx + r * math.sin(t), cy - r * math.cos(t)))
    return out


def dogbone_outline(
    l_narrow: float,
    w_narrow: float,
    l_overall: float,
    w_grip: float,
    fillet_r: float,
) -> list[tuple[float, float]]:
    """生成狗骨（哑铃）形试件的闭合轮廓点（逆时针，中心在原点）。

    过渡段用**解析圆弧**离散为直线段（20 段/侧），误差 < 0.01 mm，
    对力学测量无影响；相比 3D 布尔 fillet，几何构造稳定得多。
    """
    x1 = l_narrow / 2.0
    x_end = l_overall / 2.0
    half_n, half_g = w_narrow / 2.0, w_grip / 2.0

    dy = half_g - half_n
    if dy <= 0:
        raise ValueError("端部宽度必须大于窄段宽度")
    cos_t = (fillet_r - dy) / fillet_r
    cos_t = max(-1.0, min(1.0, cos_t))
    th_max = math.acos(cos_t)

    cy = half_n + fillet_r          # 圆弧圆心 y
    cx = x1                         # 圆弧圆心 x（切点在 (x1, half_n)）
    x_arc_end = cx + fillet_r * math.sin(th_max)

    upper_arc = _arc_pts(cx, cy, fillet_r, 0.0, th_max)
    lower_arc = [(x, -y) for (x, y) in upper_arc]

    # 右半边：从窄段下边出发，经右下过渡、端部、右上过渡，回到窄段上边
    right: list[tuple[float, float]] = [(0.0, -half_n)]
    right.extend(lower_arc)                       # (x1,-hn) → (x_arc_end,-hg)
    right.append((x_end, -half_g))
    right.append((x_end, half_g))
    right.extend(reversed(upper_arc))             # (x_arc_end,hg) → (x1,hn)
    right.append((0.0, half_n))

    # 左半边：镜像右半（去掉两个位于 x=0 的端点）并逆序，使轮廓闭合
    left = [(-x, y) for (x, y) in reversed(right[1:-1])]
    return right + left


def _extrude_profile(pts: Sequence[tuple[float, float]], thickness: float):
    """把二维轮廓拉成三维实体。"""
    _require()
    from build123d import Polyline, extrude, make_face

    wire = Polyline(*pts, close=True)
    try:
        face = make_face(wire)
    except Exception:
        face = make_face(wire.edges())
    return extrude(face, amount=thickness)


def _dot_marker(n: int, radius: float = 0.6, height: float = 0.6):
    """用 n 个凸点做位号标记（不依赖字体，打印后一眼可辨）。"""
    _require()
    from build123d import Cylinder, Pos

    if n <= 0:
        return None
    spacing = radius * 2.6
    total = (n - 1) * spacing
    markers = None
    for i in range(n):
        c = Pos(i * spacing - total / 2.0, 0, height / 2.0) * Cylinder(radius, height)
        markers = c if markers is None else markers + c
    return markers


# 凸点参数（行列标用）
DOT_RADIUS = 0.7
DOT_HEIGHT = 0.6
DOT_SPACING = 1.9


def _dot_row(
    count: int,
    cx: float,
    cy: float,
    z_base: float,
    radius: float = DOT_RADIUS,
    spacing: float = DOT_SPACING,
    height: float = DOT_HEIGHT,
):
    """在 (cx, cy) 处水平排布 ``count`` 个凸点（整体居中）。

    用途：**行列标**。顶部一排点表示列号、左侧一排点表示行号，
    这样孔位可以用「行-列」二元组唯一确定，而不需要印数字（无需字体）。
    """
    _require()
    from build123d import Cylinder, Pos

    if count <= 0:
        return None
    total = (count - 1) * spacing
    marks = None
    for i in range(count):
        x = cx + i * spacing - total / 2.0
        c = Pos(x, cy, z_base + height / 2.0) * Cylinder(radius, height)
        marks = c if marks is None else marks + c
    return marks


def _sweep(lo: float, hi: float, step: float) -> tuple[float, ...]:
    n = int(round((hi - lo) / step))
    return tuple(round(lo + i * step, 3) for i in range(n + 1))


# ⚠️ 扫描范围的设计原则（第一版设计错了，这里是修正后的逻辑）
# ------------------------------------------------------------------------------
# **先问，再测；不要用穷举代替提问。**
#
# 唯一真正需要的未知量是切片器的 **XY 孔补偿值 c**：
#     实际孔径 ≈ 画的直径 + c    →    画的直径 = 目标实际孔径 − c
# 而 c 是**已知信息**（老师设好的，问一句就有），不需要打一堆试件去反推。
#
# 本批试件的作用降到最小：**只用于确认"问到的 c 是否真的能用"**，
# 以及回答"竖直孔与水平孔是否一致"这类无法从 c 推出的问题。
#
# 因此范围只需覆盖**配合类别**，不需要覆盖补偿的不确定性：
#     Ø6 轴承：压入 ≈5.95 / 转动 ≈6.10 / 松配 ≈6.25
#     → 画的范围取 5.85–6.25（步长 0.10）即可，5 档足够定配合。
#     Ø3 轴：转动 ≈3.15 → 取 3.05–3.25，3 档。
#     水平孔：只需知道"与竖直孔差多少" → 2 档（一个偏紧、一个偏松）。
#
# 若老师答不出 c，再退回到宽范围兜底（见 WIDE_DELTAS_*）。
DELTAS_D6 = (-0.15, -0.05, 0.05, 0.15, 0.25)     # 5 档 → 画 5.85–6.25
DELTAS_D3 = (0.05, 0.15, 0.25)                    # 3 档 → 画 3.05–3.25
DELTAS_D6_H = (-0.05, 0.15)                       # 2 档 → 画 5.95 / 6.15

# 兜底：只有"问不到 c 且没有卡尺"时才用（不要默认用这个）
WIDE_DELTAS_D6 = _sweep(-0.40, 0.40, 0.05)        # 17 档
WIDE_DELTAS_D3 = _sweep(-0.30, 0.40, 0.05)        # 15 档
WIDE_DELTAS_D6_H = _sweep(-0.30, 0.40, 0.10)      # 8 档


# ==============================================================================
# T1-a：孔轴配合板（竖直孔 · 平躺打印）
# ==============================================================================
def hole_fit_plate(
    nominal_d: float = 6.0,
    deltas: Sequence[float] | None = None,
    plate_t: float = 4.0,
    cols: int = 6,
    pitch: float = 15.0,
    margin_left: float = 12.0,
    margin_top: float = 12.0,
    margin_right: float = 6.0,
    margin_bottom: float = 6.0,
):
    """阶梯孔径板 —— 测「竖直孔（Z 向）画多大装得进」的映射。

    读法：**顶部一排点 = 列号，左侧一排点 = 行号**，
    孔位按「行-列」在 ``READING_TABLE.md`` 里对应到具体直径。

    为什么扫描范围这么宽（而不是围绕 nominal 取窄带）
    ------------------------------------------------
    实际孔径 ≈ 画的直径 + c + δ，其中 c 是**切片软件的 XY 孔补偿**（实验室
    当前设定，通常未知），δ 是机器固有偏差。范围必须覆盖 c + δ 的不确定性，
    否则可能出现"整盘打完却找不到合格档"的情况。详见 ``DELTAS_D6`` 的注释。

    也因此：**不需要要求任何人把孔补偿设为 0**——只要保证它在标定件与
    正式件之间**不变**，扫描结果就直接可用（因为它测的就是"画多大装得进"）。
    """
    if deltas is None:
        deltas = DELTAS_D3 if abs(nominal_d - 3.0) < 1e-9 else DELTAS_D6

    _require()
    from build123d import Box, Cylinder, Pos

    n = len(deltas)
    rows = math.ceil(n / cols)
    w = margin_left + cols * pitch + margin_right
    h = margin_top + rows * pitch + margin_bottom

    part = Pos(w / 2, h / 2, plate_t / 2) * Box(w, h, plate_t)

    def _hole_xy(r: int, c: int) -> tuple[float, float]:
        return margin_left + pitch * (c + 0.5), margin_bottom + pitch * (r + 0.5)

    holes = None
    for i, d in enumerate(deltas):
        r, c = divmod(i, cols)
        x, y = _hole_xy(r, c)
        tool = Pos(x, y, -0.5) * Cylinder((nominal_d + d) / 2.0, plate_t + 1.0)
        holes = tool if holes is None else holes + tool
    part -= holes

    # --- 行列标（凸点，免字体）---------------------------------------------
    marks = None
    # 列标：顶部，第 c 列标 c+1 个点
    for c in range(cols):
        x, _ = _hole_xy(0, c)
        m = _dot_row(c + 1, x, h - margin_top / 2.0, plate_t)
        if m is not None:
            marks = m if marks is None else marks + m
    # 行标：左侧，第 r 行标 r+1 个点
    for r in range(rows):
        _, y = _hole_xy(r, 0)
        m = _dot_row(r + 1, margin_left / 2.0, y, plate_t)
        if m is not None:
            marks = m if marks is None else marks + m
    if marks is not None:
        part += marks
    return part


# ==============================================================================
# T1-b：水平孔块（水平孔 · 平躺打印，孔轴沿 X）
# ==============================================================================
def horizontal_hole_block(
    nominal_d: float = 6.0,
    deltas: Sequence[float] | None = None,
    block_h: float = 12.0,
    block_w: float = 14.0,
    block_t: float = 8.0,
    pitch: float = 15.0,
    margin: float = 8.0,
):
    """水平孔块 —— 测「孔轴在 XY 平面内」时的孔径偏差。

    ⚠️ 为什么水平孔必须单独测
    ------------------------
    FDM 打印水平孔时，孔的**顶部悬垂**会下垂（"teardrop" 效应），使孔的有效
    截面变成非圆，且在 X 方向测量时表现为截面偏小。这与竖直孔的误差规律
    **完全不同**（竖直孔主要受孔补偿量与层间收缩影响）。工程上把竖直孔的
    补偿量套到水平孔上，是装配失败的常见根因。

    读法：**孔上方一排点 = 序号**（第 i 孔标 i 个点，共 8 档）。
    """
    if deltas is None:
        deltas = DELTAS_D6_H

    _require()
    from build123d import Box, Cylinder, Pos, Rot

    n = len(deltas)
    w = n * pitch + 2 * margin

    part = Pos(w / 2, block_w / 2, block_t / 2) * Box(w, block_w, block_t)

    # 孔轴沿 Y（水平），贯穿 block_w
    holes = None
    for i, d in enumerate(deltas):
        x = margin + pitch * (i + 0.5)
        tool = Pos(x, 0.0, 0.0) * Rot(90, 0, 0) * Cylinder((nominal_d + d) / 2.0, block_w + 2.0)
        tool = Pos(0, block_w / 2.0, block_t / 2.0) * tool
        holes = tool if holes is None else holes + tool
    part -= holes

    # 序号凸点（孔上方）
    marks = None
    for i in range(n):
        x = margin + pitch * (i + 0.5)
        m = _dot_row(i + 1, x, 2.0, block_t)
        if m is not None:
            marks = m if marks is None else marks + m
    if marks is not None:
        part += marks
    return part


# ==============================================================================
# T1-c：细节能力板（薄壁 / 桥接 / 最小柱 / 圆角）
# ==============================================================================
def detail_plate(
    wall_thicknesses: Sequence[float] = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5),
    bridge_spans: Sequence[float] = (5.0, 8.0, 10.0, 15.0, 20.0),
    pin_diameters: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    base_t: float = 4.0,
):
    """细节能力板 —— 一次性摸清本机的工艺下限。

    测三组能力：

    1. **薄壁**：竖直薄片，看哪些厚度能立住、有没有翘曲
    2. **桥接**：两侧支墩之间的水平跨度，看哪些能不打支撑自桥成型
    3. **最小柱**：细圆柱，看哪些直径能打出来且不易断

    这三组数据直接决定 ``printenv.yaml`` 里 ``min_wall_mm`` /
    ``max_bridge_mm`` / ``min_pin_dia_mm`` 该怎么填——**目前那些值是经验默认，
    本试件的作用就是把它们换成实测值**。
    """
    _require()
    from build123d import Box, Cylinder, Pos

    part = None

    # --- 薄壁组（左起第 i 段 = 第 i 个厚度，用序号凸点标记）---
    x = 6.0
    wall_h, wall_len = 25.0, 12.0
    for i, t in enumerate(wall_thicknesses):
        base = Pos(x + wall_len / 2 - wall_len / 2, 10.0, base_t / 2) * Box(wall_len, wall_len, base_t)
        wall = Pos(x + wall_len / 2 - wall_len / 2, 10.0, base_t + wall_h / 2) * Box(wall_len, t, wall_h)
        seg = base + wall
        m = _dot_row(i + 1, x, 10.0 - wall_len / 2 + 1.5, base_t)
        if m is not None:
            seg += m
        part = seg if part is None else part + seg
        x += wall_len + 4.0

    # --- 桥接组 ---
    x = 6.0
    y2 = 42.0
    for i, span in enumerate(bridge_spans):
        pier = 6.0
        base = Pos(x + span / 2 + pier, y2, base_t / 2) * Box(span + 2 * pier, 12.0, base_t)
        bridge = Pos(x + span / 2 + pier, y2, base_t + 2.0) * Box(span, 4.0, 4.0)
        pillar_l = Pos(x + pier / 2, y2, base_t + 6.0) * Box(pier, 4.0, 12.0)
        pillar_r = Pos(x + span + pier * 1.5, y2, base_t + 6.0) * Box(pier, 4.0, 12.0)
        seg = base + bridge + pillar_l + pillar_r
        m = _dot_row(i + 1, x + span / 2 + pier, y2 - 4.5, base_t)
        if m is not None:
            seg += m
        part = seg if part is None else part + seg
        x += span + 2 * pier + 4.0

    # --- 最小柱组 ---
    x = 6.0
    y3 = 62.0
    for i, d in enumerate(pin_diameters):
        base = Pos(x + 3.0, y3, base_t / 2) * Box(6.0, 6.0, base_t)
        pin = Pos(x + 3.0, y3, base_t + 12.0) * Cylinder(d / 2.0, 24.0)
        seg = base + pin
        m = _dot_row(i + 1, x + 3.0, y3 - 4.0, base_t)
        if m is not None:
            seg += m
        part = seg if part is None else part + seg
        x += 9.0

    return part


# ==============================================================================
# T1-d：尺寸基准件（收缩与累积误差）
# ==============================================================================
def dimension_coupon(cube_mm: float = 20.0, bar_len: float = 100.0, bar_w: float = 10.0):
    """尺寸基准件 —— 测 XY 收缩率与长距离累积误差。

    * 立方体：用于三维卡尺测量（X/Y/Z 三向收缩率可能不同：XY 受挤出量补偿影响，
      Z 受层高与层间挤压影响）
    * 长条：用于测**长距离累积误差**——这对"零件要拼装"的场景至关重要，
      因为 100 mm 上的 0.3% 收缩 = 0.3 mm，足以让拼接孔对不上。
    """
    _require()
    from build123d import Box, Pos

    bar_t = 5.0
    cube = Pos(cube_mm / 2 + 5.0, bar_w / 2, cube_mm / 2) * Box(cube_mm, cube_mm, cube_mm)
    bar = Pos(cube_mm + 10.0 + bar_len / 2, bar_w / 2, bar_t / 2) * Box(bar_len, bar_w, bar_t)
    return cube + bar


# ==============================================================================
# T2-a：ASTM D638 Type V 拉伸试件
# ==============================================================================
D638_TYPE_V = dict(
    l_narrow=9.53, w_narrow=3.18, l_overall=63.5, w_grip=9.53, fillet_r=12.7, thickness=3.2
)


def tensile_type_v(
    thickness: float = D638_TYPE_V["thickness"],
    l_narrow: float = D638_TYPE_V["l_narrow"],
    w_narrow: float = D638_TYPE_V["w_narrow"],
    l_overall: float = D638_TYPE_V["l_overall"],
    w_grip: float = D638_TYPE_V["w_grip"],
    fillet_r: float = D638_TYPE_V["fillet_r"],
    grip_holes: bool = False,
):
    """ASTM D638 Type V 拉伸试件（默认平躺打印）。

    **本试件要打两个方向**：

    * ``flat``（平躺）→ 拉伸轴在层平面内 → 测 ``E_xy`` / ``σy_xy``
    * ``upright``（竖立）→ 拉伸轴沿打印 Z 向 → 测 ``E_z`` / ``σy_z``（层间）

    两者的比值就是 ``printenv.yaml`` 里 ``E_z_mpa / E_xy_mpa`` 和
    ``yield_z_mpa / yield_xy_mpa`` 的实测依据——**这是本通道各向异性功能的
    数据来源**，也是相对竞品的差异化基础。

    ⚠️ 竖立打印提示：63.5 mm 高、3.2 mm 厚的细条，需开 brim（≥5 mm）并降低
    首层外壁速度；若仍失败，退化为自设计的短标距 Z 向试件（另见 ``z_tensile_stub``）。
    """
    pts = dogbone_outline(l_narrow, w_narrow, l_overall, w_grip, fillet_r)
    part = _extrude_profile(pts, thickness)

    if grip_holes:
        _require()
        from build123d import Cylinder, Pos, Rot

        r = 2.0
        for sx in (-1, 1):
            tool = Pos(sx * (l_overall / 2.0 - 5.0), 0.0, -1.0) * Cylinder(r, thickness + 2.0)
            part -= tool
    return part


def z_tensile_stub(
    gauge_len: float = 6.0,
    width: float = 6.0,
    thickness: float = 6.0,
    grip_len: float = 8.0,
):
    """Z 向（层间）拉伸**备用**试件 —— 矮粗设计，规避竖立打印失稳。

    立式细长试件在 FDM 上极易被喷嘴推倒。本试件把标距段缩短、截面加粗，
    用「高厚比 ≈ 4」换取打印成功率，代价是标距短、应变测量精度略低。

    仅当 ``tensile_type_v`` 竖立打印反复失败时启用。
    截面 6×6 mm 便于用卡尺精确测量实际尺寸（打印后按实测截面算应力）。
    """
    _require()
    from build123d import Box, Pos

    total_h = grip_len + gauge_len + grip_len
    gauge = Pos(0, 0, grip_len + gauge_len / 2.0) * Box(width, thickness, gauge_len)
    g1 = Pos(0, 0, grip_len / 2.0) * Box(width + 4.0, thickness + 4.0, grip_len)
    g2 = Pos(0, 0, grip_len + gauge_len + grip_len / 2.0) * Box(
        width + 4.0, thickness + 4.0, grip_len
    )
    return gauge + g1 + g2


# ==============================================================================
# T2-b：三点弯曲试件
# ==============================================================================
def bend_coupon(length: float = 80.0, width: float = 10.0, thickness: float = 4.0):
    """三点弯曲试件（参考 ASTM D790 常用尺寸，跨距 64 mm）。

    价值：**短粗件打印成功率高**，且弯曲试件竖放时能有效暴露层间结合强度，
    可作为竖立拉伸失败时的替代方案。

    弯曲模量 :math:`E_f = \\dfrac{L^3 m}{4 b h^3}`，抗弯强度
    :math:`\\sigma_f = \\dfrac{3FL}{2bh^2}`（L 为跨距）。
    """
    _require()
    from build123d import Box, Pos

    return Pos(0, 0, thickness / 2.0) * Box(length, width, thickness)


# ==============================================================================
# 试件全集
# ==============================================================================
def build_all(env: PrintEnv | None = None, full: bool = False) -> list[Coupon]:
    """返回标定试件清单。

    Parameters
    ----------
    full : ``False``（默认）= **最小必要集**，只含"必须实测"的件；
           ``True`` = 含可选的工艺能力板与尺寸基准件。

    设计原则：**先问，再测**。切片器孔补偿值是已知信息（问老师即可），
    所以配合试件的作用仅是**确认**，不是穷举测绘。工艺能力（薄壁/桥接）与
    尺寸收缩量可以通过**保守设计**规避（壁厚统一 ≥1.2、桥接 ≤8、拼接孔留余量），
    故列为可选。
    """
    items = [
        Coupon(
            name="T1a_fit_plate_d6",
            purpose="Ø6 竖直孔配合确认（压入 / 转动 / 松配），并校验问到的补偿值",
            build=lambda: hole_fit_plate(nominal_d=6.0, deltas=DELTAS_D6, cols=5, pitch=13.0),
            quantity=1,
            orientation="flat",
            measures="fits.press_fit_mm / rotating_clearance_mm / calibrated",
            note="画 5.85/5.95/6.05/6.15/6.25 共 5 档；列标凸点",
        ),
        Coupon(
            name="T1b_fit_plate_d3",
            purpose="Ø3 钢轴孔配合确认",
            build=lambda: hole_fit_plate(nominal_d=3.0, deltas=DELTAS_D3, cols=3, pitch=11.0),
            quantity=1,
            orientation="flat",
            measures="fits.sliding_clearance_mm",
            note="画 3.05/3.15/3.25 共 3 档",
        ),
        Coupon(
            name="T1c_horizontal_holes",
            purpose="水平孔与竖直孔的差异（无法从补偿值推出，必须实测）",
            build=lambda: horizontal_hole_block(nominal_d=6.0, deltas=DELTAS_D6_H, pitch=13.0),
            quantity=1,
            orientation="flat",
            measures="水平孔是否需单独补偿",
            note="画 5.95/6.15 共 2 档；与 T1a 同孔径档对比手感",
        ),
        Coupon(
            name="T2a_tensile_type_v_flat",
            purpose="ASTM D638 Type V，平躺 → 层内力学性能 E_xy / σy_xy",
            build=lambda: tensile_type_v(),
            quantity=5,
            orientation="flat",
            measures="material.E_xy_mpa / yield_xy_mpa / ultimate_xy_mpa",
            note="5 个取统计；100% 填充、锁定层高",
        ),
        Coupon(
            name="T2b_tensile_type_v_upright",
            purpose="同款**竖立**打印 → 层间力学性能 E_z / σy_z（各向异性的核心数据）",
            build=lambda: tensile_type_v(),
            quantity=5,
            orientation="upright",
            measures="material.E_z_mpa / yield_z_mpa  → 各向异性比",
            note="⚠️ 需 brim ≥5 mm；已预旋转 90°（长度沿 Z），导入切片即可",
            pre_rotate_deg=(0.0, 90.0, 0.0),
        ),
        Coupon(
            name="T2c_bend_coupon",
            purpose="三点弯曲（短粗件，打印成功率高）；竖放可暴露层间结合强度",
            build=bend_coupon,
            quantity=5,
            orientation="flat",
            measures="弯曲模量/强度，与拉伸互为校核",
            note="80×10×4，跨距 64 mm",
        ),
    ]

    if full:
        items.extend([
            Coupon(
                name="T1d_detail_plate",
                purpose="[可选] 薄壁下限 / 桥接跨度上限 / 最小可打印柱径",
                build=detail_plate,
                quantity=1,
                orientation="flat",
                measures="design_rules.min_wall_mm / max_bridge_mm / min_pin_dia_mm",
                note="可用保守设计规避；仅在需要极限减重时打",
            ),
            Coupon(
                name="T1e_dimension_coupon",
                purpose="[可选] XY/Z 收缩率 + 长距离累积误差",
                build=dimension_coupon,
                quantity=1,
                orientation="flat",
                measures="收缩率（拼接孔可留余量规避）",
                note="20 mm 立方 + 100×10×5 长条",
            ),
        ])
    return items


# ==============================================================================
# 导出与读数表
# ==============================================================================
def export_all(
    env: PrintEnv,
    out_dir: str | Path,
    only: Sequence[str] | None = None,
    full: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """导出试件为 STEP + STL，并返回汇总。"""
    from . import geom as G

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for c in build_all(env, full=full):
        if only and c.name not in only:
            continue
        try:
            part = c.build()
            if any(abs(a) > 1e-9 for a in c.pre_rotate_deg):
                from build123d import Rot

                rx, ry, rz = c.pre_rotate_deg
                part = Rot(rx, ry, rz) * part
            metrics = G.measure(part, env, name=c.name)
            art = G.export(part, out_dir, stem=c.name)
            total_g = metrics.mass_g * c.quantity
            rows.append(
                dict(
                    name=c.name, orientation=c.orientation, quantity=c.quantity,
                    unit_mass_g=round(metrics.mass_g, 2), total_mass_g=round(total_g, 2),
                    bbox=tuple(round(b, 1) for b in metrics.bbox_mm),
                    fits=metrics.fits_in_printer(env).ok,
                    volume_mm3=round(metrics.volume_mm3, 1),
                    step=art["step"], stl=art["stl"],
                    purpose=c.purpose, measures=c.measures, note=c.note,
                )
            )
            if verbose:
                print(f"  OK  {c.name:32s} x{c.quantity}  {metrics.mass_g:6.2f} g/件  "
                      f"合计 {total_g:6.2f} g  bbox={tuple(round(b,1) for b in metrics.bbox_mm)}")
        except Exception as e:
            rows.append(dict(name=c.name, error=f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"  FAIL {c.name}: {type(e).__name__}: {e}")

    total = sum(r.get("total_mass_g", 0) or 0 for r in rows)
    return {"rows": rows, "total_mass_g": round(total, 2), "out_dir": str(out_dir)}


def reading_table(env: PrintEnv) -> str:
    """生成实测记录表（Markdown），供打印后手填。"""

    def _grid_table(nominal: float, deltas: Sequence[float], cols: int, unit: str) -> list[str]:
        rows = math.ceil(len(deltas) / cols)
        out = [
            f"| 孔位 | 画出的孔径 (mm) | 标准件 | 能否插入 | 手感（插不进/紧/顺滑/旷） | 备注 |",
            f"| :-: | ---: | :-: | :--: | :--- | :--- |",
        ]
        if rows == 1:
            for i, d in enumerate(deltas):
                out.append(f"| 列{i + 1} | **{nominal + d:.2f}** | {unit} | | | |")
            return out
        for r in range(rows - 1, -1, -1):      # 从最上面一行往下打印，与实物一致
            for c in range(cols):
                i = r * cols + c
                if i >= len(deltas):
                    continue
                out.append(
                    f"| 行{r + 1}-列{c + 1} | **{nominal + deltas[i]:.2f}** | {unit} | | | |"
                )
        return out

    L = [
        "# 标定试件 · 实测读数表",
        "",
        "> **先记录打印参数**（否则数据不可用）：层高 ______ / 填充 ______ / 墙数 ______ /",
        "> 喷嘴 ______ / 打印板 ______ / **切片器 XY 孔补偿设定 ______** / 切片预设名 ______",
        "",
        "> ⚠️ **孔补偿不需要改**——本表测的是「画多大能装」，在任何补偿设定下都成立。",
        "> 但务必**记下它的值**，并保证标定件与正式作品使用同一设定。",
        "",
        "**核心用途**：由下表反推 →",
        "",
        "```",
        "    画的直径 = 目标实际孔径 − c        （c = 切片器孔补偿）",
        "```",
        "",
        "拿到「压入 / 转动」两档的画法后，以后所有 Ø6 孔直接照此画，**不必再试**。",
        "",
        "---",
        "",
        "## T1a 竖直孔 · Ø6 基准（配 MR63ZZ 轴承，外径 6.00 mm）",
        "",
        f"画出的孔径范围 **6.00−0.40 → 6.00+0.40 mm**，共 {len(DELTAS_D6)} 档，步长 0.05。",
        "",
    ]
    L += _grid_table(6.0, DELTAS_D6, 6, "MR63ZZ")
    L += [
        "",
        "**结论填写**：",
        "",
        "| 目标 | 需要的实际孔径 | 找到的档位（行-列） | 画出的孔径 (mm) |",
        "| :--- | ---: | :--- | ---: |",
        "| 压入配合（轴承压进去不掉） | ≈5.95 | | |",
        "| 转动配合（轴承能自由转） | ≈6.20 | | |",
        "| 松配合（可轻松插拔） | ≈6.40 | | |",
        "",
        "---",
        "",
        "## T1b 竖直孔 · Ø3 基准（配 Ø3 钢轴）",
        "",
        f"画出的孔径范围 **3.00−0.30 → 3.00+0.40 mm**，共 {len(DELTAS_D3)} 档，步长 0.05。",
        "",
    ]
    L += _grid_table(3.0, DELTAS_D3, 5, "Ø3 轴")
    L += [
        "",
        "**结论填写**：",
        "",
        "| 目标 | 需要的实际孔径 | 找到的档位（行-列） | 画出的孔径 (mm) |",
        "| :--- | ---: | :--- | ---: |",
        "| 转动配合（轴能自由转） | ≈3.15 | | |",
        "| 滑动配合（能推拉滑动） | ≈3.10 | | |",
        "",
        "---",
        "",
        "## T1c 水平孔 · Ø6 基准（孔轴沿 Y 轴，水平方向）",
        "",
        f"画出的孔径范围 **6.00−0.30 → 6.00+0.40 mm**，共 {len(DELTAS_D6_H)} 档，步长 0.10。",
        "读法：**孔上方一排点 = 序号**（从左往右，1 个点 = 第 1 档）。",
        "",
        "| 序号 | 画出的孔径 (mm) | 标准件 | 能否插入 | 手感 | 与 T1a 同孔径档相比 |",
        "| :-: | ---: | :-: | :--: | :--- | :--- |",
    ]
    for i, d in enumerate(DELTAS_D6_H):
        L.append(f"| {i + 1} | **{6.0 + d:.2f}** | MR63ZZ | | | 更松 / 一样 / 更紧 |")

    L += [
        "",
        "> 🎯 **T1c 的核心价值**：找出「同样画 6.10 mm，水平孔和竖直孔差多少」。",
        "> 如果差得明显（通常水平孔更紧），说明以后所有**水平孔都要单独补偿**——",
        "> 这是把竖直孔的经验套到水平孔上、导致装配失败的根因。",
        "",
        "---",
        "",
        "## T1d 工艺能力（读法：各组从左往右依次为第 1、2、3… 档，用序号凸点标记）",
        "",
        "### 薄壁组（1=0.4 / 2=0.6 / 3=0.8 / 4=1.0 / 5=1.2 / 6=1.5 mm）",
        "",
        "| 档位 | 壁厚 (mm) | 结果（立住/歪斜/塌陷） | 备注 |",
        "| :-: | ---: | :--- | :--- |",
    ]
    for i, t in enumerate((0.4, 0.6, 0.8, 1.0, 1.2, 1.5)):
        L.append(f"| {i + 1} | {t} | | |")

    L += [
        "",
        "### 桥接组（1=5 / 2=8 / 3=10 / 4=15 / 5=20 mm）",
        "",
        "| 档位 | 跨度 (mm) | 结果（完好/下垂/断裂） | 备注 |",
        "| :-: | ---: | :--- | :--- |",
    ]
    for i, s in enumerate((5.0, 8.0, 10.0, 15.0, 20.0)):
        L.append(f"| {i + 1} | {s} | | |")

    L += [
        "",
        "### 最小柱组（1=Ø1.0 / 2=Ø1.5 / 3=Ø2.0 / 4=Ø2.5 / 5=Ø3.0 / 6=Ø4.0）",
        "",
        "| 档位 | 柱径 (mm) | 结果（完好/弯曲/断裂） | 备注 |",
        "| :-: | ---: | :--- | :--- |",
    ]
    for i, d in enumerate((1.0, 1.5, 2.0, 2.5, 3.0, 4.0)):
        L.append(f"| {i + 1} | {d} | | |")

    L += [
        "",
        "---",
        "",
        "## T1e 尺寸基准",
        "",
        "| 测量项 | 标称 (mm) | 实测 (mm) | 偏差 (mm) | 相对偏差 |",
        "| :--- | ---: | ---: | ---: | ---: |",
        "| 立方 X | 20.00 | | | |",
        "| 立方 Y | 20.00 | | | |",
        "| 立方 Z | 20.00 | | | |",
        "| 长条长度 | 100.00 | | | |",
        "| 长条宽度 | 10.00 | | | |",
        "| 长条厚度 | 5.00 | | | |",
        "",
        "---",
        "",
        "## T2 材料试件",
        "",
        "### T2a 平躺（层内）",
        "",
        "| 编号 | 窄段宽 b (mm) | 厚度 h (mm) | 截面 A (mm²) | 断裂载荷 F (N) | σ = F/A (MPa) | 断裂位置 |",
        "| :-: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for i in range(1, 6):
        L.append(f"| flat-{i} | | | | | | 窄段内 / 端部（无效） |")

    L += [
        "",
        "### T2b 竖立（层间）—— 关键数据",
        "",
        "| 编号 | 窄段宽 b (mm) | 厚度 h (mm) | 截面 A (mm²) | 断裂载荷 F (N) | σ = F/A (MPa) | 断裂位置 |",
        "| :-: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for i in range(1, 6):
        L.append(f"| up-{i} | | | | | | 窄段内 / 端部（无效） |")

    L += [
        "",
        "**各向异性比** = 竖立平均 σ ÷ 平躺平均 σ = ______  （这是仿真里 `E_z / E_xy` 的依据）",
        "",
        "### T2c 三点弯曲（跨距 64 mm）",
        "",
        "| 编号 | 宽 b | 厚 h | 最大力 F (N) | σ_f = 3FL/(2bh²) (MPa) | 备注 |",
        "| :-: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for i in range(1, 6):
        L.append(f"| bend-{i} | | | | | |")

    L += [
        "",
        "---",
        "",
        "## 数据回填",
        "",
        "实测完成后把结果写回 `config/printenv.yaml`：",
        "",
        "```yaml",
        "meta:",
        "  calibrated: true            # 材料参数标定完成",
        "fits:",
        "  calibrated: true            # 公差标定完成",
        "  rotating_clearance_mm:      # ← T1a「转动配合」档：实际孔径 − 6.00，再 /2（单边）",
        "  press_fit_mm:               # ← T1a「压入配合」档：实际孔径 − 6.00（过盈为负）",
        "  sliding_clearance_mm:       # ← T1b「滑动配合」档",
        "material:",
        "  E_xy_mpa:                   # ← T2a",
        "  E_z_mpa:                    # ← T2b",
        "  yield_xy_mpa:               # ← T2a",
        "  yield_z_mpa:                # ← T2b",
        "design_rules:",
        "  min_wall_mm:                # ← T1d 薄壁组最小成功厚度",
        "  max_bridge_mm:              # ← T1d 桥接组最大成功跨度",
        "  min_pin_dia_mm:             # ← T1d 最小柱组",
        "printer:",
        "  xy_hole_compensation_mm:    # ← 记录当时设定值（勿设 0，如实填写即可）",
        "```",
        "",
    ]
    return "\n".join(L)
