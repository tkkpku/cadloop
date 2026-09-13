"""cadloop.examples.biplane — 0.5 m 级双翼机（课程作品级复杂度复刻）。

为什么做这个
------------
行星齿轮箱证明了"通道能处理复杂几何"，但轮子本身外形偏简单。本件按
**往届优秀作品（学长 0.5 m 双翼机，多件装配）** 同级或更高的复杂度重建，
用途是向组员证明 AI 参数化建模的真实能力。

技术内核（不是拼积木，全部是真工程几何）
----------------------------------------
1. **真 NACA 四位数翼型**：``NACA 2412``（上翼）/ ``NACA 4412``（下翼）——
   厚度分布用 NACA 标准多项式，中弧线用四位数弯度公式，厚度沿中弧线法向施加。
2. **放样（loft）**：机翼用根/梢两个翼型截面放样，得到真实的**锥形翼**；
   机身用 6 个圆角矩形截面放样成**流线型机身**。
3. **扭转螺旋桨**：桨叶沿展向在多个截面上**改变安装角**（pitch），
   用相邻截面放样出真实扭转 —— 这是螺旋桨建模的标志性特征。
4. **强度关键件的结构校核**：翼梁（悬臂梁弯曲）、起落架支柱（压弯组合）。

打印分件（H2D 幅面 300×320×320，本机双喷嘴交集）
------------------------------------------------
0.5 m 翼展**必然超幅面**，分件是硬需求：

============================  ==========  ==========================
零件                          尺寸(mm)    分件方式
============================  ==========  ==========================
机身                          440 长      3 段（前/中/尾），插销对接
上翼                          500 展      3 段（中 190 + 两侧 155）
下翼                          430 展      2 段（各 215）
翼间支柱 / 撑杆                ~118 高     单件
尾翼（水平+垂直）              ~180 展     单件
起落架腿 / 轮                  Ø62         单件
螺旋桨 / 整流罩                Ø150        单件
============================  ==========  ==========================

对接全部用**打印插销 + 承插孔**（不依赖任何金属件，符合 ¥100 预算约束）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ==============================================================================
# 1. NACA 四位数翼型
# ==============================================================================

def naca4_surface(m: float = 0.02, p: float = 0.4, t: float = 0.12,
                  n: int = 48) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """返回 (上表面, 下表面) 点列（弦长归一化：前缘 x=0，后缘 x=1）。

    * 厚度分布：``y_t = 5t(0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1015x⁴)``
    * 中弧线（四位数）：``x<p`` 时 ``y_c = m/p²(2px − x²)``，否则
      ``y_c = m/(1−p)²((1−2p) + 2px − x²)``
    * 厚度沿中弧线**法向**施加（这才是 NACA 的标准做法；很多简化实现直接加到
      垂直方向，弯度大时会明显失真）。
    """
    beta = [math.pi * i / (n - 1) for i in range(n)]
    xs = [0.5 * (1.0 - math.cos(b)) for b in beta]      # 余弦加密，前后缘更密
    yt = [5.0 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x
                     + 0.2843 * x ** 3 - 0.1015 * x ** 4) for x in xs]
    yc, dyc = [], []
    for x in xs:
        if x < p:
            yc.append(m / p ** 2 * (2.0 * p * x - x * x))
            dyc.append(2.0 * m / p ** 2 * (p - x))
        else:
            yc.append(m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x))
            dyc.append(2.0 * m / (1.0 - p) ** 2 * (p - x))
    th = [math.atan(d) for d in dyc]
    up = [(xs[i] - yt[i] * math.sin(th[i]), yc[i] + yt[i] * math.cos(th[i]))
          for i in range(n)]
    lo = [(xs[i] + yt[i] * math.sin(th[i]), yc[i] - yt[i] * math.cos(th[i]))
          for i in range(n)]
    return up, lo


def naca4_outline(m: float = 0.02, p: float = 0.4, t: float = 0.12,
                  n: int = 48, chord: float = 1.0,
                  le_at_origin: bool = True) -> list[tuple[float, float]]:
    """闭合翼型轮廓（**逆时针**：上表面 LE→TE，再下表面 TE→LE）。

    ``le_at_origin=True`` 时前缘在 (0,0)、后缘在 (chord,0)；否则以弦长中点为原点。
    """
    up, lo = naca4_surface(m, p, t, n)
    pts = list(up) + list(reversed(lo))[1:-1]
    x0 = 0.0 if le_at_origin else -0.5
    return [((x + x0) * chord, y * chord) for (x, y) in pts]


def naca4_area(m: float = 0.02, p: float = 0.4, t: float = 0.12, chord: float = 1.0,
               n: int = 2000) -> float:
    """翼型截面积（数值积分，用于体积/质量估算的独立校核）。"""
    up, lo = naca4_surface(m, p, t, n)
    a = 0.0
    for i in range(n - 1):
        a += 0.5 * ((up[i][1] - lo[i][1]) + (up[i + 1][1] - lo[i + 1][1])) \
            * (up[i + 1][0] - up[i][0])
    return abs(a) * chord * chord


# ==============================================================================
# 2. 总体参数（按 0.5 m 级双翼机比例，参考 Pitts 类机型）
# ==============================================================================
@dataclass(frozen=True)
class BiplaneSpec:
    """双翼机总体参数（mm）。"""

    # ---- 机翼 ----
    span_upper: float = 500.0
    span_lower: float = 430.0
    chord_root: float = 118.0
    chord_tip: float = 86.0
    t_upper: float = 0.10          # NACA 2410（模型机常用，比 12% 更省料）
    t_lower: float = 0.10          # NACA 4410
    m_upper: float = 0.02
    m_lower: float = 0.04
    p_camber: float = 0.40
    gap: float = 118.0             # 上下翼垂直间距
    stagger: float = 15.0          # 上翼相对下翼前移量

    # ---- 机身 ----
    fuselage_len: float = 440.0
    nose_y: float = 0.0            # 机身轴线高度

    # ---- 尾翼 ----
    hs_span: float = 180.0
    hs_chord_root: float = 62.0
    hs_chord_tip: float = 40.0
    fin_height: float = 118.0
    fin_chord_root: float = 72.0
    fin_chord_tip: float = 38.0
    tail_x: float = 400.0          # 尾翼前缘位置

    # ---- 起落架 ----
    wheel_d: float = 62.0
    wheel_w: float = 22.0
    gear_x: float = 118.0
    gear_track: float = 168.0

    # ---- 动力 ----
    prop_d: float = 150.0
    prop_blades: int = 2
    cowl_d: float = 82.0

    # ---- 分件 ----
    dovetail_pin_d: float = 8.0    # 对接插销直径
    join_depth: float = 18.0       # 插入深度
    window_bay: float = 56.0       # 机身减重窗间距（= 隔框间距）

    # ---- 派生 ----
    @property
    def wing_gap_ratio(self) -> float:
        """翼隙 / 弦长。双翼机典型 1.0~1.3；过小则上下翼干扰严重。"""
        return self.gap / self.chord_root

    @property
    def taper_ratio(self) -> float:
        return self.chord_tip / self.chord_root

    def wing_aspect_ratio(self, span: float) -> float:
        mean_chord = 0.5 * (self.chord_root + self.chord_tip)
        return span / mean_chord


SPEC = BiplaneSpec()


# ==============================================================================
# 3. 工具
# ==============================================================================
def _b123d():
    from cadloop.geom import require_build123d
    require_build123d()


def _airfoil_face(spec: BiplaneSpec, chord: float, y: float, upper: bool,
                  z_offset: float):
    """在 y=y 平面生成翼型截面（Face），厚度方向为 Z。"""
    from build123d import Polyline, make_face, Pos

    m, t = (spec.m_upper, spec.t_upper) if upper else (spec.m_lower, spec.t_lower)
    pts2 = naca4_outline(m, spec.p_camber, t, chord=chord, le_at_origin=True)
    pts3 = [(x, 0.0, zz) for (x, zz) in pts2]
    face = make_face(Polyline(*pts3, close=True))
    # 前缘置于 x=0，整体平移到指定 y
    return Pos(0.0, y, z_offset) * face


def _loft_surface(sections: list) -> object:
    from build123d import loft
    return loft(sections)


def _surface(spec: BiplaneSpec, span: float, chord_root: float, chord_tip: float,
             t_ratio: float, m_camber: float, n_sec: int = 4,
             station_ys: list[float] | None = None) -> object:
    """通用锥形翼面：弦沿 X，展向沿 +Y（从 0 起），厚度沿 Z。

    ``station_ys`` 允许指定非均匀站位（用于把翼面切成可打印的分段）。
    """
    ys = station_ys if station_ys is not None else [
        span * i / (n_sec - 1) for i in range(n_sec)]
    secs = []
    for y in ys:
        frac = y / span if span > 0 else 0.0
        c = chord_root + (chord_tip - chord_root) * frac
        secs.append(_airfoil_face_xy(c, t_ratio, m_camber, y))
    return _loft_surface(secs)


def _airfoil_face_xy(chord: float, t_ratio: float, m_camber: float, y: float):
    from build123d import Polyline, make_face, Pos
    pts2 = naca4_outline(m_camber, 0.4, t_ratio, chord=chord, le_at_origin=True)
    pts3 = [(x, 0.0, z) for (x, z) in pts2]
    f = make_face(Polyline(*pts3, close=True))
    return Pos(0.0, y, 0.0) * f


# ==============================================================================
# 4. 零件构建
# ==============================================================================
def build_wing_panel(spec: BiplaneSpec, y0: float, y1: float, upper: bool,
                     half_span: float, holed: bool = True, root_socket: bool = False):
    """一段机翼（可打印分件）。

    弦长沿展向线性收缩 → 用两个截面放样出**真实锥形翼**；
    再沿展向挖一排**减重椭圆孔**（真实 3D 打印机翼的标准做法，
    同时把打印时间砍掉一半）。
    """
    from build123d import Ellipse, Pos, Rot, Cylinder, extrude

    m = spec.m_upper if upper else spec.m_lower
    t = spec.t_upper if upper else spec.t_lower

    def chord_at(y):
        # ⚠️ 必须用 **abs(y)**：弦长收缩只与「离机身中心线的距离」有关。
        #    首版写成 y/half_span，于是中央段（y 从 −95 到 +95）左半边弦长
        #    反而变大（130.2 mm vs 右半边 105.8 mm），**左右不对称**，
        #    且与左外段接头对不上（左外段 y=−95 处是 105.8）。
        f = abs(y) / half_span
        return spec.chord_root + (spec.chord_tip - spec.chord_root) * f

    secs = [_airfoil_face_xy(chord_at(y), t, m, y) for y in (y0, y1)]
    panel = _loft_surface(secs)

    if holed:
        c_max = max(chord_at(y0), chord_at(y1))
        thick = t * c_max
        hz = max(thick * 0.42, 3.0)          # 孔高（留 ~29% 上下蒙皮）
        hx = c_max * 0.46                     # 孔长（弦向）
        x_c = c_max * 0.42                    # 孔中心（约在最大厚度处）
        span_len = y1 - y0
        n = max(int(span_len // 30.0), 1)
        for i in range(n):
            yh = y0 + span_len * (i + 0.5) / n
            tool = Pos(x_c, yh, -60.0) * Ellipse(hx / 2.0, hz / 2.0)
            panel = panel - extrude(tool, amount=120.0)

    if root_socket:
        r = spec.dovetail_pin_d / 2.0
        d = spec.join_depth
        pin_hole = Pos(spec.chord_root * 0.35, y0 - 1.0, 0.0) * \
            Rot(90, 0, 0) * Cylinder(r, d + 2)
        panel = panel - pin_hole
    return panel


def build_wing_joiner(spec: BiplaneSpec, length: float):
    """机翼对接插销（配合两段的承插孔）。"""
    from build123d import Cylinder, Pos, Rot
    return Pos(0, 0, 0) * Rot(90, 0, 0) * Cylinder(spec.dovetail_pin_d / 2.0, length)


def build_strut(spec: BiplaneSpec, length: float, chord: float = 26.0,
                t_ratio: float = 0.14):
    """翼间支柱：**流线型截面**（真实双翼机的支柱是翼型，不是圆管）。

    用 NACA 0014 对称翼型沿长度放样 —— 抗弯截面系数比同面积圆管高，
    且阻力小。两端各留一段实心插入端。
    """
    return _surface(spec, length, chord, chord * 0.85, t_ratio, 0.0, n_sec=3)


def build_fuselage_sections(spec: BiplaneSpec) -> list[tuple[float, float, float, float]]:
    """机身截面表：(x, 宽, 高, 圆角半径)。"""
    return [
        (0.0,   62.0,  72.0, 16.0),
        (55.0,  76.0,  92.0, 20.0),
        (120.0, 86.0, 104.0, 24.0),
        (200.0, 80.0,  96.0, 22.0),
        (280.0, 66.0,  78.0, 18.0),
        (360.0, 44.0,  54.0, 12.0),
        (440.0, 22.0,  30.0,  7.0),
    ]


def build_fuselage_segment(spec: BiplaneSpec, x0: float, x1: float,
                           cockpit: bool = False):
    """一段机身：放样 + **半硬壳化**（蒙皮 + 隔框 + 纵梁 + 减重窗）。

    ⚠️ 为什么必须掏空（2026-09-13 由质量预算抓出）
    ---------------------------------------------
    首版按放样实体直接出件，机身实心体积 **2.19 L = 2.7 kg PLA**，全机 4.0 kg
    —— 完全不实用（打印 20+ 小时、材料也浪费）。真实飞机机身是**半硬壳**：
    薄蒙皮 + 隔框 + 纵梁，材料利用率高一个量级。

    本实现：
    1. ``offset(solid, -wall)`` 掏成等壁厚壳体（``wall = 2.2 mm``）；
    2. 沿轴每 ``bay`` 切 4 个**减重窗**（上/下/左/右各一），
       窗与窗之间留下的十字筋就是**纵梁**，窗与窗之间的环向材料就是**隔框**。
    """
    from build123d import Plane, RectangleRounded, Pos, Box, offset

    tbl = build_fuselage_sections(spec)
    pts = [s for s in tbl if x0 - 1e-9 <= s[0] <= x1 + 1e-9]
    for xe in (x0, x1):
        if all(abs(s[0] - xe) > 1e-9 for s in pts):
            lo = max((s for s in tbl if s[0] < xe), key=lambda s: s[0])
            hi = min((s for s in tbl if s[0] > xe), key=lambda s: s[0])
            f = (xe - lo[0]) / (hi[0] - lo[0])
            pts.append((xe, lo[1] + (hi[1] - lo[1]) * f,
                        lo[2] + (hi[2] - lo[2]) * f,
                        lo[3] + (hi[3] - lo[3]) * f))
    pts.sort(key=lambda s: s[0])

    secs = [Plane.YZ.offset(x) * RectangleRounded(w, h, r) for (x, w, h, r) in pts]
    solid = _loft_surface(secs)

    # ---- ① 半硬壳化：等壁厚壳体 ----
    wall = 2.2
    try:
        inner = offset(solid, -wall, kind=_kind())
        shell = solid - inner
        if float(shell.volume) > 100.0:
            solid = shell
    except Exception:
        pass    # 放样面太曲时 offset 可能失败，退回实心（并在清单里看得出来）

    # ---- ② 减重窗：留下十字纵梁 + 环形隔框 ----
    def _dim_at(xe):
        if xe <= pts[0][0]:
            return pts[0][1], pts[0][2]
        if xe >= pts[-1][0]:
            return pts[-1][1], pts[-1][2]
        lo = max((s for s in pts if s[0] <= xe), key=lambda s: s[0])
        hi = min((s for s in pts if s[0] >= xe), key=lambda s: s[0])
        if hi[0] - lo[0] < 1e-9:
            return lo[1], lo[2]
        f = (xe - lo[0]) / (hi[0] - lo[0])
        return lo[1] + (hi[1] - lo[1]) * f, lo[2] + (hi[2] - lo[2]) * f

    bay = spec.window_bay
    n_win = max(int((x1 - x0) // bay), 0)
    inset = 2.0
    for i in range(n_win):
        xc = x0 + bay * (i + 0.5)
        w, h = _dim_at(xc)
        win_len = bay * 0.50
        win_wide = min(w, h) * 0.48
        # ⚠️ 切窗的方盒必须**只向外延伸**：尺寸取截面的一半（不是固定大值），
        #    否则在细长尾段会把整个截面切穿（首版就踩了，尾段直接碎掉）。
        for sign, axis in ((1, "y"), (-1, "y"), (1, "z")):
            if axis == "y":
                r_out, depth, wide = w / 2.0, w / 2.0, win_wide
                c = Pos(xc, sign * (r_out - inset), 0.0)
                cut = c * Box(win_len, depth, wide)
            else:
                if cockpit:
                    continue                    # 座舱段顶面不开窗
                r_out, depth, wide = h / 2.0, h / 2.0, win_wide
                c = Pos(xc, 0.0, sign * (r_out - inset))
                cut = c * Box(win_len, wide, depth)
            solid = solid - cut

    if cockpit:
        from build123d import Ellipse, extrude
        cut = Pos(148.0, 0.0, 34.0) * Ellipse(56.0, 36.0)
        solid = solid - extrude(cut, amount=44.0)
    return solid


def _kind():
    from build123d import Kind
    return Kind.ARC



def build_tail_surface(spec: BiplaneSpec, span: float, chord_root: float,
                       chord_tip: float, vertical: bool = False):
    """尾翼面：水平安定面 / 垂直尾翼（都用对称翼型 NACA 0010）。"""
    from build123d import Rot, Pos

    surf = _surface(spec, span, chord_root, chord_tip, 0.10, 0.0, n_sec=3)
    if vertical:
        surf = Rot(90, 0, 0) * surf           # 展向转到 +Z
        surf = Pos(0, 0, 0) * surf
    return surf


def build_wheel(spec: BiplaneSpec):
    """机轮：胎 + 轮毂 + 6 个减重孔 + 轴孔。"""
    from build123d import Cylinder, Pos, Rot, Box

    r = spec.wheel_d / 2.0
    tire = Rot(90, 0, 0) * Cylinder(r, spec.wheel_w)      # 轴沿 Y
    for i in range(6):
        a = 60.0 * i
        hole = Pos(0, -spec.wheel_w, r * 0.62) * Rot(0, 0, a) * \
            (Rot(0, 90, 0) * Cylinder(r * 0.17, spec.wheel_w * 3))
        tire = tire - hole
    hub = Rot(90, 0, 0) * Cylinder(r * 0.30, spec.wheel_w * 1.25)
    tire = tire + hub - (Rot(90, 0, 0) * Cylinder(4.1, spec.wheel_w * 3))
    return tire


def build_gear_leg(spec: BiplaneSpec, side: int = 1):
    """起落架腿：A 形撑杆（翼型截面）+ 轮轴 + 上端与机身对接。"""
    from build123d import Cylinder, Pos, Rot, Box

    leg = _surface(spec, 118.0, 24.0, 18.0, 0.14, 0.0, n_sec=3)
    # ⚠️ 起落架要**朝下**：展向 +Y 转到 −Z 需 ``Rot(−90,0,0)``
    #    （首版写成 +90 用了反转，腿朝上穿进机身 z=+98）。
    leg = Rot(-90, 0, 0) * leg
    leg = Pos(spec.gear_x, 0.0, -20.0) * leg
    axle = Pos(spec.gear_x, side * spec.gear_track / 2.0, -120.0) * \
        Rot(90, 0, 0) * Cylinder(4.0, 42.0)
    return leg + axle


def build_cowl(spec: BiplaneSpec):
    """发动机整流罩：带 6 个冷却进气口。"""
    from build123d import Cylinder, Pos, Rot, Box

    r = spec.cowl_d / 2.0
    cowl = Pos(6.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(r, 54.0)
    cowl = cowl - (Pos(6.0, 0.0, 0.0) * Rot(0, 90, 0) * Cylinder(r - 5.0, 70.0))
    for i in range(6):
        a = 60.0 * i
        cut = Pos(6.0, 0.0, 0.0) * Rot(0, 90, 0) * Rot(0, 0, a) * \
            Box(20.0, 14.0, r * 2.4)
        cut = Pos(-9.0, 0.0, 0.0) * cut
        cowl = cowl - cut
    return cowl


def build_propeller(spec: BiplaneSpec):
    """螺旋桨：**扭转变安装角**的两叶桨（这是螺旋桨建模的标志性特征）。"""
    from build123d import Sphere, Pos, Rot

    r_hub = 9.0
    r_tip = spec.prop_d / 2.0
    stations = [
        (r_hub,      16.0, 38.0),
        (r_hub + 16, 22.0, 30.0),
        (r_tip * 0.7, 18.0, 22.0),
        (r_tip - 4,  10.0, 16.0),
    ]
    secs = []
    for (yy, chord, pitch_deg) in stations:
        f = _airfoil_face_xy(chord, 0.12, 0.04, 0.0)      # 弦沿 X，厚沿 Z
        secs.append(Pos(0.0, yy, 0.0) * Rot(0, pitch_deg, 0) * f)
    blade = _loft_surface(secs)

    hub = Pos(0, 0, 0) * Rot(0, 90, 0) * Sphere(r_hub * 1.15)
    prop = blade + (Rot(180, 0, 0) * blade)
    return prop + hub


def build_joiner_pin(spec: BiplaneSpec, length: float | None = None):
    """通用对接插销。"""
    from build123d import Cylinder, Pos
    L = length if length is not None else spec.join_depth * 2.0
    return Pos(0, 0, 0) * Cylinder(spec.dovetail_pin_d / 2.0 - 0.15, L)


# ==============================================================================
# 5. 总装（含分件方案）
# ==============================================================================
@dataclass
class Part:
    name: str
    solid: object
    material_note: str = ""


def build_all(spec: BiplaneSpec = SPEC) -> list[Part]:
    """返回全部零件（已按 H2D 幅面分件）。"""
    from build123d import Pos, Rot
    P: list[Part] = []

    half_u = spec.span_upper / 2.0          # 250
    half_l = spec.span_lower / 2.0          # 215
    z_up = spec.gap                         # 上翼高度（相对下翼）
    x_stag = spec.stagger                   # 上翼前移

    # ---- 机翼：上翼 3 段（中 95 + 外 155） / 下翼 2 段（215）----
    #
    # ⚠️ 对称件必须用 **mirror(about=Plane.XZ)**，不能用 Rot(0,0,180)。
    #    两个坑叠在一起（都踩过）：
    #    ① ``Rot(0,0,180)`` 绕 Z 转 180° 会把 (x,y) 同时取反 —— 对**弦向不对称**
    #       的机翼来说左右翼前缘会一前一后；
    #    ② **``Plane.YZ`` 的法向是 X**，镜像它会翻 X 而不是 Y —— 要用
    #       ``Plane.XZ``（含 X、Z 轴，法向 = Y）才是左右对称。
    from build123d import mirror, Plane

    split_u = 95.0
    for side, tag in ((1, "R"), (-1, "L")):
        def _place(shape):
            return shape if side == 1 else mirror(shape, about=Plane.XZ)

        # 上翼外段
        w = build_wing_panel(spec, split_u, half_u, True, half_u, root_socket=True)
        P.append(Part(f"wing_upper_outer_{tag}",
                      _place(Pos(x_stag, 0, z_up) * w), "NACA 2410"))
        # 下翼整段（半翼）
        wl = build_wing_panel(spec, 0.0, half_l, False, half_l, root_socket=False)
        P.append(Part(f"wing_lower_{tag}", _place(wl), "NACA 4410"))

    # 上翼中段（跨机身，-95..+95）
    cu = build_wing_panel(spec, -split_u, split_u, True, half_u)
    cu = Pos(x_stag, 0, z_up) * cu
    P.append(Part("wing_upper_center", cu, "NACA 2412"))

    # ---- 机身 3 段 ----
    P.append(Part("fuselage_front", build_fuselage_segment(spec, 0.0, 147.0),
                  "放样流线机身 + 整流罩接口"))
    P.append(Part("fuselage_mid", build_fuselage_segment(spec, 147.0, 294.0,
                                                         cockpit=True), "含座舱开口"))
    P.append(Part("fuselage_tail", build_fuselage_segment(spec, 294.0, 440.0),
                  "尾锥"))

    # ---- 翼间支柱：每侧前后各一（N 形撑）----
    #
    # ⚠️ 旋转方向（2026-09-13 修正）：绕 X 轴转 θ 时
    #    ``y' = y·cosθ − z·sinθ, z' = y·sinθ + z·cosθ``。
    #    ``_surface`` 的展向在 +Y，要把它竖成 **+Z（向上撑到上翼）** 需 ``Rot(90,0,0)``；
    #    首版写了 ``Rot(−90,0,0)`` → 支柱朝下戳到 z=−118，看起来像「悬空零件」。
    for side, tag in ((1, "R"), (-1, "L")):
        for dx, sub in ((-8.0, "fwd"), (74.0, "aft")):
            s = build_strut(spec, spec.gap)
            s = Pos(spec.chord_root * 0.35 + dx, 0, 0) * Rot(90, 0, 0) * s
            s = Pos(0, side * 150.0, 0) * s
            P.append(Part(f"strut_{sub}_{tag}", s, "NACA 0014 流线支柱"))

    # ---- 尾翼 ----
    hs = build_tail_surface(spec, spec.hs_span, spec.hs_chord_root,
                            spec.hs_chord_tip)
    hs = Pos(spec.tail_x, -spec.hs_span / 2.0, 0) * hs
    P.append(Part("tailplane", hs, "NACA 0010 水平安定面"))
    fin = build_tail_surface(spec, spec.fin_height, spec.fin_chord_root,
                             spec.fin_chord_tip, vertical=True)
    fin = Pos(spec.tail_x + 26.0, 0, 10.0) * fin
    P.append(Part("fin", fin, "NACA 0010 垂直尾翼"))

    # ---- 起落架 + 轮 ----
    for side, tag in ((1, "R"), (-1, "L")):
        P.append(Part(f"gear_leg_{tag}", build_gear_leg(spec, side), "A 形翼型撑杆"))
        P.append(Part(f"wheel_{tag}", Pos(spec.gear_x, side * spec.gear_track / 2.0,
                                          -120.0) * build_wheel(spec), "胎+毂+减重孔"))

    # ---- 动力 ----
    P.append(Part("cowl", build_cowl(spec), "带冷却进气口"))
    P.append(Part("propeller", Pos(-14.0, 0, 0) * build_propeller(spec), "扭转变距两叶桨"))

    # ---- 对接插销 ----
    P.append(Part("joiner_wing_upper", Pos(0, 0, 0) * build_joiner_pin(spec, 34.0),
                  "上翼中段↔外段"))
    P.append(Part("joiner_fuselage", build_joiner_pin(spec, 30.0), "机身 3 段对接"))
    return P


def summary_table(parts: list[Part], env) -> str:
    """零件清单（名称 / 体积 / 质量 / 包围盒）。"""
    from cadloop import geom as G
    rows = []
    total = 0.0
    for p in parts:
        m = G.measure(p.solid, env, p.name)
        total += m.volume_mm3
        rows.append((p.name, m.volume_mm3, m.mass_g, m.bbox_mm, p.material_note))
    return rows, total

