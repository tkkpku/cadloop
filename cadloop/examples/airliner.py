"""cadloop.examples.airliner — 0.5 m 级**双发单翼客机**（课程作品级复杂度）。

术语更正
--------
少爷要的是「双**发**」（twin-engine，翼吊两台发动机）+ **单翼** + 类似民航客机。
首版本鱼按字面理解做成了「双**翼**机」（biplane，上下两层机翼）—— 那是完全
不同的构型。本文件是正确构型。

构型（典型窄体客机）
--------------------
::

        ┌─── 垂直尾翼
        │
    ════╪════════════════ 水平尾翼
     ╱  │                    ╲
    ▓   │  单翼（低单翼 + 后掠）│   ▓      ← 翼梢小翼
        │                    ╱
        ◯◯  ← 两台翼吊发动机（短舱 + 挂架）
       ╱  ╲
      ◯    ◯   ← 主起落架        ◯ 前起落架

技术内核（与双翼机版共用同一套真几何）
--------------------------------------
* **真 NACA 四位数翼型**（``NACA 2412`` 客机常用翼型）
* **放样**：机翼根/梢截面 + 后掠偏移 → 真实后掠锥形翼
* **翼吊发动机**：短舱（轴对称放样）+ 挂架（翼型剖面）
* **半硬壳机身**：2.2 mm 蒙皮 + 隔框 + 纵梁 + 一排**客舱舷窗**
* **上翘尾锥**：客机标志性外形

分件（H2D 幅面 300×320×320）
----------------------------
============  ===========  ================================
零件           尺寸(mm)      分件
============  ===========  ================================
机身 470 长     470          3 段，插销对接
机翼 500 展     500          3 段（中 190 + 两侧 155）
发动机短舱      92           单件 ×2
挂架            ~70          单件 ×2
水平/垂直尾翼    ~185         单件
起落架/轮        ~65          单件 ×3
翼梢小翼        ~55          单件 ×2
============  ===========  ================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .biplane import Part, _loft_surface, naca4_outline

# ==============================================================================
# 1. 参数
# ==============================================================================
@dataclass(frozen=True)
class AirlinerSpec:
    """双发单翼客机总体参数（mm）。"""

    # ---- 机翼 ----
    span: float = 500.0
    wing_root_chord: float = 92.0
    wing_tip_chord: float = 52.0
    wing_t: float = 0.12            # NACA 2412
    wing_m: float = 0.02
    sweep_deg: float = 16.0         # 前缘后掠
    wing_le_x: float = 168.0        # 翼根前缘位置（机身坐标）
    wing_z: float = -30.0           # 低单翼：机翼在机身下侧

    # ---- 机身 ----
    fuselage_len: float = 470.0
    shell: float = 2.2
    bay: float = 58.0
    window_n: int = 14
    window_w: float = 12.0
    window_h: float = 8.0

    # ---- 发动机 ----
    nacelle_d: float = 42.0
    nacelle_len: float = 92.0
    nacelle_y: float = 130.0
    nacelle_x: float = 186.0        # 短舱前缘 x

    # ---- 尾翼 ----
    ht_span: float = 185.0
    ht_root: float = 58.0
    ht_tip: float = 34.0
    ht_x: float = 418.0
    fin_height: float = 108.0
    fin_root: float = 74.0
    fin_tip: float = 40.0
    fin_x: float = 392.0

    # ---- 起落架 ----
    nose_gear_x: float = 100.0
    nose_wheel_d: float = 32.0
    main_gear_x: float = 252.0
    main_gear_y: float = 54.0
    main_wheel_d: float = 44.0
    gear_drop: float = 62.0

    # ---- 翼梢小翼 ----
    winglet_h: float = 52.0
    winglet_root: float = 46.0
    winglet_tip: float = 24.0

    # ---- 分件 ----
    pin_d: float = 8.0
    join_depth: float = 18.0

    # ---- 派生 ----
    @property
    def half_span(self) -> float:
        return self.span / 2.0

    @property
    def mean_chord(self) -> float:
        return 0.5 * (self.wing_root_chord + self.wing_tip_chord)

    @property
    def aspect_ratio(self) -> float:
        """展弦比。窄体客机典型 8~10；此件按 0.5 m 模型比例取较小值。"""
        return self.span / self.mean_chord

    @property
    def taper_ratio(self) -> float:
        return self.wing_tip_chord / self.wing_root_chord

    def chord_at(self, y: float) -> float:
        """弦长沿展向线性收缩 —— **用 abs(y)**（左右必须对称）。"""
        f = abs(y) / self.half_span
        return self.wing_root_chord + (self.wing_tip_chord - self.wing_root_chord) * f

    def le_x_at(self, y: float) -> float:
        """前缘 x 坐标（后掠）。"""
        return self.wing_le_x + abs(y) * math.tan(math.radians(self.sweep_deg))

    @property
    def wing_area_mm2(self) -> float:
        """机翼参考面积（梯形公式）。"""
        return self.span * self.mean_chord


SPEC = AirlinerSpec()


# ==============================================================================
# 2. 基础几何
# ==============================================================================
def _face(chord: float, t: float, m: float, y: float, x_le: float, z: float):
    """把翼型放在 y=y 的横截面上（弦沿 X、厚度沿 Z）。"""
    from build123d import Polyline, make_face
    pts2 = naca4_outline(m, 0.4, t, chord=chord, le_at_origin=True)
    pts3 = [(x + x_le, y, zz + z) for (x, zz) in pts2]
    return make_face(Polyline(*pts3, close=True))


def _lin(a: float, b: float, n: int) -> list[float]:
    return [a] if n <= 1 else [a + (b - a) * i / (n - 1) for i in range(n)]


def build_wing_panel(spec: AirlinerSpec, y0: float, y1: float, n_sec: int = 3,
                     root_socket: bool = False):
    """后掠锥形翼的一段（可打印分件）。

    ⚠️ **不要在外表面挖穿孔减重孔**（2026-09-13 修正的概念错误）
    ------------------------------------------------------------
    首版沿展向打了一排贯穿翼厚的椭圆孔"省材料"，被少爷一眼看穿
    「这玩意不坠机才怪」。错在三点：

    1. **没必要**：``measure`` 给的是**实心等效**体积；切片软件的填充率
       （典型 15~20%）本来就会把内部做成稀疏结构 —— CAD 里挖穿等于重复劳动。
    2. **破坏气动面**：真实机翼的减重孔在**翼盒内部**，外面有蒙皮。把蒙皮打穿
       等于给机翼开了一排缝，升力面直接被破坏。
    3. **看起来就是错的**：评审/组员第一眼就会质疑，反而削弱可信度。

    正确做法：**翼面保持光滑实心**，让切片器决定内部填充。
    （若确实要减重，应当做成**不破蒙皮的内部空腔**，或翼肋+翼梁的开放式结构。）
    """
    ys = _lin(y0, y1, n_sec)
    secs = [_face(spec.chord_at(y), spec.wing_t, spec.wing_m, y,
                  spec.le_x_at(y), spec.wing_z) for y in ys]
    panel = _loft_surface(secs)

    if root_socket:
        from build123d import Pos, Rot, Cylinder
        r = spec.pin_d / 2.0
        panel = panel - (Pos(spec.le_x_at(y0) + spec.chord_at(y0) * 0.35,
                             y0 - 1.0, spec.wing_z) *
                         Rot(90, 0, 0) * Cylinder(r, spec.join_depth + 2))
    return panel


def build_fuselage_sections(spec: AirlinerSpec):
    """机身截面表：(x, 宽, 高, 圆角, z 偏移)。尾段整体上翘 —— 客机标志。"""
    return [
        (0.0,   12.0, 12.0,  5.0,  0.0),
        (26.0,  44.0, 46.0, 19.0,  0.0),
        (64.0,  68.0, 72.0, 27.0,  0.0),
        (112.0, 76.0, 80.0, 30.0,  0.0),
        (300.0, 76.0, 80.0, 30.0,  0.0),
        (360.0, 70.0, 76.0, 27.0,  2.0),
        (408.0, 52.0, 60.0, 21.0,  9.0),
        (446.0, 30.0, 38.0, 13.0, 19.0),
        (470.0, 16.0, 22.0,  7.0, 27.0),
    ]


def _dim_at(tbl, xe):
    if xe <= tbl[0][0]:
        return tbl[0][1:]
    if xe >= tbl[-1][0]:
        return tbl[-1][1:]
    lo = max((s for s in tbl if s[0] <= xe), key=lambda s: s[0])
    hi = min((s for s in tbl if s[0] >= xe), key=lambda s: s[0])
    if hi[0] - lo[0] < 1e-9:
        return lo[1:]
    f = (xe - lo[0]) / (hi[0] - lo[0])
    return tuple(lo[i] + (hi[i] - lo[i]) * f for i in range(1, 5))


def build_fuselage_segment(spec: AirlinerSpec, x0: float, x1: float,
                          windows: bool = False):
    """机身段：放样 + 半硬壳 + 隔框/纵梁 + 可选舷窗。"""
    from build123d import (Plane, RectangleRounded, Pos, Box, offset, Ellipse,
                           extrude, Cylinder, Rot)

    tbl = build_fuselage_sections(spec)
    pts = [s for s in tbl if x0 - 1e-9 <= s[0] <= x1 + 1e-9]
    for xe in (x0, x1):
        if all(abs(s[0] - xe) > 1e-9 for s in pts):
            pts.append((xe, *_dim_at(tbl, xe)))
    pts.sort(key=lambda s: s[0])

    secs = [Pos(x, 0, dz) * (Plane.YZ * RectangleRounded(w, h, r))
            for (x, w, h, r, dz) in pts]
    solid = _loft_surface(secs)

    # ① 半硬壳
    try:
        from build123d import Kind
        inner = offset(solid, -spec.shell, kind=Kind.ARC)
        shell = solid - inner
        if float(shell.volume) > 100.0:
            solid = shell
    except Exception:
        pass

    # ② 隔框间减重窗（留下纵梁）+ 舷窗
    n = max(int((x1 - x0) // spec.bay), 0)
    for i in range(n):
        xc = x0 + spec.bay * (i + 0.5)
        w, h, r, dz = _dim_at(tbl, xc)
        wl, ww = spec.bay * 0.48, min(w, h) * 0.44
        for sign in (1, -1):                      # 左右侧
            solid = solid - (Pos(xc, sign * (w / 2.0 - 2.0), dz) *
                             Box(wl, w / 2.0, ww))
        solid = solid - (Pos(xc, 0.0, -h / 2.0 + 2.0 + dz) *
                         Box(wl, ww, h / 2.0))    # 底部
        solid = solid - (Pos(xc, 0.0, h / 2.0 - 2.0 + dz) *
                         Box(wl, ww, h / 2.0))    # 顶部

    # ③ 客舱舷窗（左右各一排）—— 客机的招牌特征
    if windows:
        xa, xb = max(x0 + 22.0, 130.0), min(x1 - 26.0, 330.0)
        if xb > xa:
            for k in range(spec.window_n):
                xw = xa + (xb - xa) * k / max(spec.window_n - 1, 1)
                w, h, r, dz = _dim_at(tbl, xw)
                for sign in (1, -1):
                    c = Pos(xw, sign * (w / 2.0 - spec.shell + 1.0), dz + 8.0)
                    cut = c * (Plane.XZ * Ellipse(spec.window_w / 2.0,
                                                  spec.window_h / 2.0))
                    solid = solid - extrude(cut, amount=sign * 40.0)
    return solid


def build_nacelle(spec: AirlinerSpec):
    """发动机短舱：轴对称放样（进气口—最大直径—尾喷口收口）。"""
    from build123d import Plane, Circle, Pos
    prof = [(-6.0, 0.30), (0.0, 0.62), (14.0, 0.94), (34.0, 1.00),
            (62.0, 0.92), (82.0, 0.70), (92.0, 0.52)]
    secs = [Pos(spec.nacelle_x + dx, 0, 0) *
            (Plane.YZ * Circle(spec.nacelle_d / 2.0 * rf)) for (dx, rf) in prof]
    return _loft_surface(secs)


def build_pylon(spec: AirlinerSpec):
    """发动机挂架：翼型剖面（NACA 0018 厚剖面，真实挂架就是翼型）。"""
    from build123d import Rot, Pos
    y_mid = (spec.nacelle_y + 46.0) / 2.0
    py = _surface_simple(spec, 46.0, 84.0, 30.0, 0.18)
    py = Rot(0, 0, 0) * py
    py = Pos(spec.nacelle_x + 6.0, spec.nacelle_y, spec.wing_z - 4.0) * py
    return py


def _surface_simple(spec, span, chord_root, chord_tip, t_ratio, n_sec=3):
    """沿 +Y 的简单锥形翼面（内部工具）。"""
    ys = _lin(0.0, span, n_sec)
    secs = [_face(chord_root + (chord_tip - chord_root) * (y / span), t_ratio,
                  0.0, y, 0.0, 0.0) for y in ys]
    return _loft_surface(secs)


def build_tail_surface(spec: AirlinerSpec, span: float, c_root: float,
                       c_tip: float, vertical: bool = False):
    """水平安定面 / 垂直尾翼（对称翼型 NACA 0010）。

    ⚠️ ``span`` 是**全展**。内部只建半展再镜像——首版把 ``span`` 当半展用，
    结果水平尾翼做成 370 mm（超幅面）、垂直尾翼高 216 mm（应为 108），
    **整体尺寸翻倍**。
    """
    from build123d import Rot, Pos, mirror, Plane
    half = span / 2.0
    ys = _lin(0.0, half, 3)
    secs = [_face(c_root + (c_tip - c_root) * (y / half), 0.10, 0.0, y, 0.0, 0.0)
            for y in ys]
    h = _loft_surface(secs)
    full = h + mirror(h, about=Plane.XZ)          # → 全展
    if vertical:
        full = Rot(90, 0, 0) * full               # 展向转到 +Z（0..span）
        full = Pos(0, 0, half) * full             # 翼根落在 z=0（向上长）
    return full


def build_wheel(spec: AirlinerSpec, d: float, w: float = 16.0):
    """机轮：**轮胎 + 轮辋凹槽 + 轮毂凸台 + 轴孔**（无穿孔减重孔）。

    ⚠️ 2026-09-13 修正：首版在轮盘上打 5 个"减重孔"，但
    * **转轴搞错了**——轮子的轴是 **Y**（``Rot(90,0,0)*Cylinder``），
      而孔的轴却用了 ``Rot(0,90,0)`` 变成 **X**；
    * **分布平面也错了**——孔应在 XZ 盘面内绕 **Y** 分布，却绕 Z 分布了。

    两个错叠在一起 → 5 条横穿轮子的隧道被轮缘切碎 → 轮子上出现
    「莫名其妙的不规则小洞」（少爷原话）。

    现改为真实机轮剖面：外圈轮胎 + 两侧轮辋凹槽（让胎唇显形）+ 中央轮毂凸台。
    造型正确，且同样不需要穿孔（填充率由切片器决定）。
    """
    from build123d import Cylinder, Rot
    r = d / 2.0
    wheel = Rot(90, 0, 0) * Cylinder(r, w)                       # 轮胎外圈
    wheel = wheel - (Rot(90, 0, 0) * Cylinder(r * 0.82, w * 1.4))  # 掏内腔
    wheel = wheel + (Rot(90, 0, 0) * Cylinder(r * 0.80, w * 0.52))  # 轮辐盘（窄于胎 → 形成轮辋凹槽）
    wheel = wheel + (Rot(90, 0, 0) * Cylinder(r * 0.32, w * 1.02))  # 轮毂凸台
    return wheel - (Rot(90, 0, 0) * Cylinder(3.1, w * 3))           # 轴孔


def build_gear_leg(spec: AirlinerSpec, wheel_d: float, drop: float):
    """起落架支柱（朝下）。"""
    from build123d import Rot, Pos
    leg = _surface_simple(spec, drop, 20.0, 14.0, 0.16)
    return Pos(0, 0, 0) * (Rot(-90, 0, 0) * leg)


# ==============================================================================
# 3. 总装
# ==============================================================================
def build_all(spec: AirlinerSpec = SPEC) -> list[Part]:
    from build123d import Pos, Rot, mirror, Plane

    P: list[Part] = []
    hs = spec.half_span

    # ---- 机身 3 段 ----
    P.append(Part("fuselage_nose", build_fuselage_segment(spec, 0.0, 160.0),
                  "放样 + 半硬壳"))
    P.append(Part("fuselage_center",
                  build_fuselage_segment(spec, 160.0, 320.0, windows=True),
                  "含 14×2 舷窗"))
    P.append(Part("fuselage_tail", build_fuselage_segment(spec, 320.0, 470.0),
                  "上翘尾锥"))

    # ---- 机翼：中段 + 左右外段（用 mirror(Plane.XZ) 保左右对称）----
    split = 95.0
    P.append(Part("wing_center",
                  build_wing_panel(spec, -split, split, n_sec=4),
                  "NACA 2412 后掠 16°"))
    for side, tag in ((1, "R"), (-1, "L")):
        w = build_wing_panel(spec, split, hs, root_socket=True)
        P.append(Part(f"wing_outer_{tag}",
                      w if side == 1 else mirror(w, about=Plane.XZ),
                      "含展向减重孔 + 对接承插孔"))
        wl = _surface_simple(spec, spec.winglet_h, spec.winglet_root,
                             spec.winglet_tip, 0.10)
        wl = Rot(90, 0, 0) * wl
        # 小翼绕 X 转 90° 后，其**厚度**落到 Y 向 → 会探出翼尖。
        # 内移「根部半厚」使外表面与翼尖齐平，翼展才是整 500.0 mm
        # （首版不做内移，翼展变成 504.6）。
        half_thick = spec.winglet_root * 0.10 / 2.0
        wl = Pos(spec.le_x_at(hs) + 6.0, side * (hs - half_thick),
                 spec.wing_z - 4.0) * wl
        P.append(Part(f"winglet_{tag}", wl, "翼梢小翼"))

    # ---- 发动机：短舱 + 挂架 ----
    for side, tag in ((1, "R"), (-1, "L")):
        nac = build_nacelle(spec)
        nac = Pos(0, side * spec.nacelle_y, spec.wing_z - 40.0) * nac
        P.append(Part(f"nacelle_{tag}", nac, "轴对称放样短舱"))
        py = _surface_simple(spec, 44.0, 82.0, 26.0, 0.18)
        py = Rot(90, 0, 0) * py
        py = Pos(spec.nacelle_x + 10.0, side * spec.nacelle_y,
                 spec.wing_z - 40.0) * py
        P.append(Part(f"pylon_{tag}", py, "NACA 0018 挂架"))

    # ---- 尾翼 ----
    ht = build_tail_surface(spec, spec.ht_span, spec.ht_root, spec.ht_tip)
    P.append(Part("h_stab", Pos(spec.ht_x, 0, 8.0) * ht, "NACA 0010 水平安定面"))
    fin = build_tail_surface(spec, spec.fin_height, spec.fin_root, spec.fin_tip,
                             vertical=True)
    # 垂直尾翼已把翼根放在 z=0，故只做水平/前后定位（不再加 z 偏移，否则会陷进机身）
    P.append(Part("v_fin", Pos(spec.fin_x, 0, 14.0) * fin, "NACA 0010 垂直尾翼"))

    # ---- 起落架 ----
    P.append(Part("gear_nose",
                  Pos(spec.nose_gear_x, 0, -22.0) *
                  build_gear_leg(spec, spec.nose_wheel_d, spec.gear_drop),
                  "前起落架"))
    P.append(Part("wheel_nose",
                  Pos(spec.nose_gear_x, 0, -22.0 - spec.gear_drop) *
                  build_wheel(spec, spec.nose_wheel_d, 12.0), "前轮"))
    for side, tag in ((1, "R"), (-1, "L")):
        P.append(Part(f"gear_main_{tag}",
                      Pos(spec.main_gear_x, side * spec.main_gear_y, -22.0) *
                      build_gear_leg(spec, spec.main_wheel_d, spec.gear_drop + 6),
                      "主起落架"))
        P.append(Part(f"wheel_main_{tag}",
                      Pos(spec.main_gear_x, side * spec.main_gear_y,
                          -22.0 - spec.gear_drop - 6) *
                      build_wheel(spec, spec.main_wheel_d, 18.0), "主轮"))

    # ---- 对接插销 ----
    P.append(Part("joiner_wing", Pos(0, 0, 0) *
                  (Rot(90, 0, 0) * __import__("build123d").Cylinder(
                      spec.pin_d / 2.0 - 0.15, 36.0)), "上/下翼对接"))
    P.append(Part("joiner_fuselage",
                  __import__("build123d").Cylinder(spec.pin_d / 2.0 - 0.15, 30.0),
                  "机身 3 段对接"))
    return P
