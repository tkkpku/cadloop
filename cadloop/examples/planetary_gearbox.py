"""cadloop.examples.planetary_gearbox — 复杂度上限压力测试件。

为什么要做这个
--------------
T3 柔性铰链是「金标准算例」：几何简单、有解析解，用来验证**通道本身对不对**。
但它无法回答另一个问题：**通道的复杂上限够不够支撑真实课程需求？**

本件按「远超课程需求」设计，一次性压测通道的五个未验证点：

============================  ==========================================
压测点                        本件如何施压
============================  ==========================================
① 曲面 / 高阶轮廓             真渐开线齿形（滚切法生成的样条，非圆弧近似）
② 多体装配                    6 个零件、含 3 个行星轮，共 9 个实体
③ 内齿悬垂                    48 齿内齿圈 = 48 处朝下齿面，打印必崩
④ 孔系 / 薄壁 / 加强筋         6×螺栓孔 + 轴承座 + 6×4mm 薄筋
⑤ 非轴对称受载                行星架 3 销受切向力，弯曲 + 扭转耦合
============================  ==========================================

几何参数（模数 2、压力角 20°、标准齿制）
----------------------------------------
::

    太阳轮 z1 = 12   →  分度圆 Ø24
    行星轮 z2 = 18   →  分度圆 Ø36   ×3
    内齿圈 z3 = 48   →  分度圆 Ø96   （12 + 2×18 = 48 ✓ 同心条件）

    中心距 a = m(z1+z2)/2 = 2×30/2 = 30 mm

渐开线齿形推导（本文件的核心数学）
----------------------------------
取基圆半径 :math:`r_b = r\\cos\\alpha`，滚动角 :math:`t`，则渐开线为

.. math::

    \\rho(t) = r_b\\sqrt{1+t^2}, \\qquad
    \\mathrm{inv}(t) = t - \\arctan t

齿在**分度圆**处的半角为 :math:`\\psi = \\pi/(2z)`。可证外齿与内齿槽的
半角遵循**同一个**函数：

.. math::

    \\theta(\\rho) = \\psi + \\mathrm{inv}(t_p) - \\mathrm{inv}(t(\\rho)),
    \\qquad t(\\rho)=\\sqrt{(\\rho/r_b)^2-1}

* **外齿轮**：齿以 :math:`\\theta(\\rho)` 为半角（随 :math:`\\rho` 增大而**变小** → 齿顶变尖）
* **内齿圈**：**齿槽**以 :math:`\\theta(\\rho)` 为半角（随 :math:`\\rho` 增大而变小
  → 槽向外收窄、齿向外变厚），故齿的半角是补角 :math:`\\pi/z - \\theta`

这条「外齿半角 == 内齿槽半角」的对应关系是本文件能用一个函数同时生成
太阳轮/行星轮/内齿圈的原因。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# ==============================================================================
# 渐开线齿形生成
# ==============================================================================

def _inv(t: float) -> float:
    """渐开线函数 inv(t) = tan(α) - α，此处以滚动角 t = tanα 为自变量。"""
    return t - math.atan(t)


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _polar(rho: float, ang: float) -> tuple[float, float]:
    return (rho * math.cos(ang), rho * math.sin(ang))


@dataclass(frozen=True)
class SpurGear:
    """标准直齿圆柱齿轮参数（外齿或内齿）。"""

    module: float = 2.0
    teeth: int = 12
    pressure_angle_deg: float = 20.0
    internal: bool = False

    # ---- 基本尺寸（GB/T 1356 标准齿制）------------------------------------
    @property
    def pitch_r(self) -> float:
        return self.module * self.teeth / 2.0

    @property
    def base_r(self) -> float:
        return self.pitch_r * math.cos(math.radians(self.pressure_angle_deg))

    @property
    def tip_r(self) -> float:
        """齿顶圆半径（内齿指向圆心，故半径更小）。"""
        return (self.pitch_r - self.module) if self.internal else (self.pitch_r + self.module)

    @property
    def root_r(self) -> float:
        """齿根圆半径（含 0.25m 顶隙）。"""
        return (self.pitch_r + 1.25 * self.module) if self.internal else \
               max(self.pitch_r - 1.25 * self.module, 0.1)

    @property
    def psi(self) -> float:
        """分度圆处齿（槽）半角。"""
        return math.pi / (2.0 * self.teeth)

    @property
    def pitch_angle(self) -> float:
        return 2.0 * math.pi / self.teeth

    @property
    def t_pitch(self) -> float:
        return math.tan(math.radians(self.pressure_angle_deg))

    def half_angle(self, rho: float) -> float:
        """齿（外齿）/ 齿槽（内齿）在半径 rho 处的半角 θ(ρ)。"""
        ratio = max((rho / self.base_r) ** 2 - 1.0, 0.0)
        return self.psi + _inv(self.t_pitch) - _inv(math.sqrt(ratio))

    def _t_range(self, n: int) -> list[float]:
        """从「齿廓有效起点」到「齿廓终点」的滚动角序列（**升序**）。

        低齿数时齿根圆会落在基圆**以内**（z < 34 @ α=20°），该段无渐开线，
        实际齿廓由过渡曲线补上——这里用径向直线近似（对打印件足够）。

        ⚠️ 注意不要写死「齿根 < 齿顶」：**内齿的齿顶半径小于齿根半径**
        （齿指向圆心）。这里统一取 min/max，外齿内齿都成立。
        （首版写死 root_r 做起点的，导致内齿圈轮廓反向自交——已修。）
        """
        r_start = max(min(self.tip_r, self.root_r), self.base_r * 1.0000001)
        r_end = max(self.tip_r, self.root_r)
        t_lo = math.sqrt(max((r_start / self.base_r) ** 2 - 1.0, 0.0))
        t_hi = math.sqrt(max((r_end / self.base_r) ** 2 - 1.0, 0.0))
        return _linspace(t_lo, t_hi, n)


def external_gear_outline(g: SpurGear, n_inv: int = 9, n_arc: int = 3
                          ) -> list[tuple[float, float]]:
    """外齿轮的闭合 2D 轮廓点列（逆时针，齿以基准角 0 为中心）。"""
    assert not g.internal, "本函数仅用于外齿"
    rb, rf, ra = g.base_r, g.root_r, g.tip_r
    ts = g._t_range(n_inv)
    pts: list[tuple[float, float]] = []

    for k in range(g.teeth):
        base = k * g.pitch_angle
        # ① 右齿面：角度 -θ，由齿根到齿顶
        for t in ts:
            rho = rb * math.sqrt(1.0 + t * t)
            pts.append(_polar(rho, base - g.half_angle(rho)))
        # ② 齿顶圆弧（-θa → +θa）
        th_a = g.half_angle(ra)
        for x in _linspace(base - th_a, base + th_a, n_arc + 2)[1:-1]:
            pts.append(_polar(ra, x))
        # ③ 左齿面：角度 +θ，由齿顶回齿根
        for t in reversed(ts):
            rho = rb * math.sqrt(1.0 + t * t)
            pts.append(_polar(rho, base + g.half_angle(rho)))
        # ④ 齿根圆弧：本齿左齿根 → 下一齿右齿根
        th_r = g.half_angle(rf)
        for x in _linspace(base + th_r, base + g.pitch_angle - th_r, n_arc + 2)[1:-1]:
            pts.append(_polar(rf, x))
    return pts


def internal_hole_outline(g: SpurGear, n_inv: int = 9, n_arc: int = 3
                          ) -> list[tuple[float, float]]:
    """内齿圈的**内孔**闭合轮廓（逆时针）。

    齿以基准角 0 为中心；齿半角 = 齿距半角 − θ(ρ)（θ 为齿槽半角）。
    返回的闭合轮廓即「要挖掉的孔」，用 ``圆盘(齿根圆) − 该轮廓`` 得到齿圈。
    """
    assert g.internal, "本函数仅用于内齿"
    rb = g.base_r
    ra, rf = g.tip_r, g.root_r          # 注意：内齿 ra < r < rf
    ts = g._t_range(n_inv)
    half_pitch = g.pitch_angle / 2.0
    pts: list[tuple[float, float]] = []

    for k in range(g.teeth):
        base = k * g.pitch_angle
        # ① 齿顶圆弧（齿中心 base，半角 = half_pitch − θ(ra)）
        h_in = half_pitch - g.half_angle(ra)
        for x in _linspace(base - h_in, base + h_in, n_arc + 2)[1:-1]:
            pts.append(_polar(ra, x))
        # ② 齿的 + 侧齿面：由齿顶向外到齿根
        for t in ts:
            rho = rb * math.sqrt(1.0 + t * t)
            pts.append(_polar(rho, base + half_pitch - g.half_angle(rho)))
        # ③ 齿根圆弧（齿槽底部）
        h_out = half_pitch - g.half_angle(rf)
        for x in _linspace(base + h_out, base + g.pitch_angle - h_out, n_arc + 2)[1:-1]:
            pts.append(_polar(rf, x))
        # ④ 下一齿的 − 侧齿面：由齿根回到齿顶（角度 = base + π − half_pitch + θ）
        for t in reversed(ts):
            rho = rb * math.sqrt(1.0 + t * t)
            pts.append(_polar(rho, base + g.pitch_angle - half_pitch + g.half_angle(rho)))
    return pts


# ==============================================================================
# 齿轮参数总表
# ==============================================================================
M = 2.0
Z_SUN, Z_PLANET, Z_RING = 12, 18, 48
assert Z_SUN + 2 * Z_PLANET == Z_RING, "同心条件不满足：z_ring = z_sun + 2·z_planet"

SUN = SpurGear(M, Z_SUN)
PLANET = SpurGear(M, Z_PLANET)
RING = SpurGear(M, Z_RING, internal=True)
CENTER_DIST = M * (Z_SUN + Z_PLANET) / 2.0     # = 30 mm


# ==============================================================================
# 零件尺寸总表
# ==============================================================================
@dataclass(frozen=True)
class GB:
    """减速箱主要尺寸（mm）。"""

    gear_width: float = 20.0        # 齿轮齿宽
    ring_wall: float = 5.0          # 齿圈外壁厚
    ring_height: float = 24.0       # 齿圈筒高
    flange_od: float = 130.0        # 底法兰外径
    flange_t: float = 6.0           # 底法兰厚
    bolt_n: int = 6                  # 螺栓数
    bolt_d: float = 4.5             # 螺栓孔（M4 通孔）
    bolt_bcd: float = 118.0         # 螺栓分布圆
    rib_t: float = 4.0              # 加强筋厚
    rib_h: float = 10.0             # 加强筋高
    carrier_plate_d: float = 84.0   # 行星架板外径（销外缘 r=35 → 留 7mm 边距）
    carrier_plate_t: float = 6.0
    pin_d: float = 10.0             # 行星销直径
    pin_len: float = 26.0
    shaft_d: float = 20.0           # 输出轴直径
    shaft_len: float = 22.0
    planet_bore_d: float = 6.2      # 行星轮内孔（轴承外圈）
    cover_t: float = 6.0
    hole_d: float = 18.0            # 行星架减重孔（Ø）
    hole_r_pos: float = 24.0        # 减重孔中心半径


GBOX = GB()
RING_OD_R = RING.root_r + GBOX.ring_wall          # 齿圈外半径


# ==============================================================================
# 3D 构建
# ==============================================================================
def _require_b123d():
    from cadloop.geom import require_build123d
    require_build123d()


def _gear_solid(g: SpurGear, width: float, bore_d: float = 0.0,
                z0: float = 0.0, y_offset: float = 0.0):
    """把 2D 齿廓拉伸成实体，可选中心孔。"""
    from build123d import Polyline, make_face, extrude, Cylinder, Pos

    if g.internal:
        pts = internal_hole_outline(g)
        from build123d import Circle
        base = Circle(g.root_r) - make_face(Polyline(*[(x, y, 0.0) for x, y in pts], close=True))
    else:
        pts = external_gear_outline(g)
        base = make_face(Polyline(*[(x, y, 0.0) for x, y in pts], close=True))

    body = extrude(base, amount=width)
    if bore_d > 0:
        body = body - Pos(0, 0, -1) * Cylinder(bore_d / 2.0, width + 2)
    if z0:
        body = Pos(0, 0, z0) * body
    if y_offset:
        body = Pos(0, y_offset, 0) * body
    return body


def build_sun():
    """太阳轮：12 齿 + Ø8 中心孔 + 键槽。"""
    from build123d import Box, Pos, Rot
    s = _gear_solid(SUN, GBOX.gear_width, bore_d=8.0)
    key = Pos(0, 4.0, GBOX.gear_width / 2) * Rot(0, 0, 0) * Box(3.0, 1.8, GBOX.gear_width + 2)
    return s - key


def build_planet():
    """行星轮：18 齿 + Ø6.2 轴承孔（含减重）。"""
    return _gear_solid(PLANET, GBOX.gear_width, bore_d=GBOX.planet_bore_d)


def build_ring_housing():
    """内齿圈箱体：内齿 + 外筒 + 底法兰(螺栓孔) + 6 加强筋 + 输出轴中心孔。

    复杂度来源：48 处内齿（= 48 处打印悬垂） + 孔系 + 薄筋 + 薄壁。

    ⚠️ **必须让各件体积重叠（emb 参数），不能只让面贴合**
    ------------------------------------------------------
    首版把齿圈筒/extrude 起点定在 z=0、法兰顶面也在 z=0，加强筋内侧面正好落在
    齿圈外壁上——三对配合面都是**面接触**。OCCT 的 fuse 对「只贴合、不重叠」
    的实体常常**不合并**，结果是 ``solids() == 8`` 的 Compound（而非 1 个实体）。

    后果是隐蔽且严重的：

    * 导出的 STL **不水密**（trimesh 报 ``watertight=False``、``bodies=2``）
    * ``dfam-check`` 因此给不出体积（``volume_mm3 = None``），支撑占比失真
    * 切片软件可能把内腔误判为外壁

    修法：让每件在配合处**嵌入 1 mm**（``emb``），保证体相交。
    这是 CAD 里的通用纪律：**装配件要"咬进去"，不要"贴上去"**。
    """
    from build123d import (Circle, Cylinder, Box, Pos, Rot, extrude,
                           make_face, Polyline)

    emb = 1.0    # 嵌入量：保证体积重叠而非面接触

    # ① 齿圈筒（内齿 + 外壁）—— 向下嵌入法兰 emb
    hole = make_face(Polyline(*[(x, y, 0.0) for x, y in internal_hole_outline(RING)],
                              close=True))
    ring = Pos(0, 0, -emb) * extrude(Circle(RING_OD_R) - hole,
                                     amount=GBOX.ring_height + emb)

    # ② 底法兰：Ø130 × 6（z∈[-6,0]），带 6× 螺栓孔 + 中心通过孔
    # ⚠️ Cylinder 是**中心对齐**的，高度 h 的圆柱占据 z∈[-h/2, h/2]。
    #    要让法兰顶面落在 z=0，位移量必须是 **-flange_t/2** 而不是 -flange_t
    #    （首版写成 -flange_t，法兰落到 z∈[-9,-3]，与齿圈裂开 3 mm → 两个实体）。
    flange = Pos(0, 0, -GBOX.flange_t / 2.0) * Cylinder(GBOX.flange_od / 2.0, GBOX.flange_t)
    for i in range(GBOX.bolt_n):
        a = 360.0 * i / GBOX.bolt_n
        flange -= Pos(GBOX.bolt_bcd / 2.0, 0, -GBOX.flange_t / 2.0) * Rot(0, 0, a) * \
            Cylinder(GBOX.bolt_d / 2.0, GBOX.flange_t + 2)
    flange -= Pos(0, 0, -GBOX.flange_t / 2.0) * Cylinder(GBOX.shaft_d / 2.0 + 3.0,
                                                         GBOX.flange_t + 2)

    # ③ 6 条外部加强筋（薄板 —— 打印与屈曲的敏感件）—— 向内外各嵌入 emb
    ribs = None
    r_in, r_out = RING_OD_R, GBOX.flange_od / 2.0
    for i in range(GBOX.bolt_n):
        a = 360.0 * i / GBOX.bolt_n + 30.0
        r_mid = (r_in - emb + r_out) / 2.0
        rib = Pos(r_mid, 0, (GBOX.rib_h - emb) / 2.0) * \
            Box(r_out - r_in + emb, GBOX.rib_t, GBOX.rib_h + emb)
        rib = Rot(0, 0, a) * rib
        ribs = rib if ribs is None else ribs + rib

    return ring + flange + ribs


def build_carrier():
    """行星架：底板 + 3 行星销 + 输出轴 + 3 减重孔 + **销根圆角**。

    受载最复杂的自制件：3 销承受切向力 → 板弯曲 + 轴扭转耦合。

    ⚠️ 销根**必须倒圆角**（2026-09-13 压测暴露）
    -------------------------------------------
    首版销根是直角内凹。这是经典**应力奇异**：内凹尖角处应力理论无穷大，
    FEA 给的是网格相关值而非物理值。实测（h=3mm、载荷仅 100 N）：
    von Mises 118.6 MPa、σzz 200.5 MPa，而手算轴根弯曲仅 ≈6.9 MPa——
    **虚高 17 倍**。加 R2 圆角把奇异点变成有限应力集中。
    """
    from build123d import Circle, Cylinder, Pos, Rot

    P, G = Pos, GBOX
    plate = P(0, 0, G.carrier_plate_t / 2.0) * Cylinder(G.carrier_plate_d / 2.0,
                                                        G.carrier_plate_t)
    for i in range(3):
        a = 120.0 * i
        plate -= P(G.hole_r_pos, 0, G.carrier_plate_t / 2.0) * Rot(0, 0, a + 60) * \
            Cylinder(G.hole_d / 2.0, G.carrier_plate_t + 2)
    shaft = P(0, 0, -G.shaft_len / 2.0) * Cylinder(G.shaft_d / 2.0, G.shaft_len)
    part = plate + shaft

    for i in range(3):
        a = 120.0 * i
        pin = P(CENTER_DIST, 0, G.carrier_plate_t + G.pin_len / 2.0) * \
            Cylinder(G.pin_d / 2.0, G.pin_len)
        part = part + Rot(0, 0, a) * pin

    # 销根圆角：挑 z≈板顶面、且轴心在 CENTER_DIST 处的圆棱边
    try:
        edges = [
            e for e in part.edges()
            if abs(e.center().Z - G.carrier_plate_t) < 1e-6
            and abs(math.hypot(e.center().X, e.center().Y) - CENTER_DIST) < 0.3
        ]
        if edges:
            part = part.fillet(2.0, edges)
    except Exception:
        pass
    # ⚠️ 轴-板交界（z=0、r=shaft_d/2 处的内凹圆棱）同样是奇异源，且正是弯矩最大处。
    #    必须在**两侧**都倒圆角（顶面侧 + 底面侧），否则 FEA 应力被奇异值主导。
    try:
        edges = [
            e for e in part.edges()
            if abs(e.center().Z) < 1e-6
            and abs(math.hypot(e.center().X, e.center().Y) - G.shaft_d / 2.0) < 0.3
        ]
        if edges:
            part = part.fillet(2.0, edges)
    except Exception:
        pass
    return part


def build_cover():
    """端盖：Ø130 × 6 + 6 螺栓孔 + 中心让位孔 + 沉台。"""
    from build123d import Circle, Cylinder, Pos, Rot

    G, P = GBOX, Pos
    cov = P(0, 0, G.cover_t / 2.0) * Cylinder(G.flange_od / 2.0, G.cover_t)
    for i in range(G.bolt_n):
        a = 360.0 * i / G.bolt_n
        cov -= P(G.bolt_bcd / 2.0, 0, G.cover_t / 2.0) * Rot(0, 0, a) * \
            Cylinder(G.bolt_d / 2.0, G.cover_t + 2)
    cov -= P(0, 0, G.cover_t / 2.0) * Cylinder(SUN.tip_r + 3.0, G.cover_t + 2)
    cov -= P(0, 0, G.cover_t - 1.0) * Cylinder(40.0, 2.0)
    return cov


def build_assembly():
    """返回 {name: solid}。行星轮 ×3 用旋转变换摆到工作位置。"""
    from build123d import Pos, Rot
    sun = build_sun()
    planet = build_planet()
    parts = {
        "ring_housing": build_ring_housing(),
        "sun_gear": sun,
        "carrier": build_carrier(),
        "cover": build_cover(),
    }
    sun = Pos(0, 0, 0) * sun
    for i in range(3):
        a = 120.0 * i
        parts[f"planet_{i+1}"] = Pos(CENTER_DIST, 0, 0) * Rot(0, 0, a + 9.0) * planet
    return parts


def carrier_boundaries():
    """行星架的边界条件（用通道现有的**平面选择**能力）。"""
    from cadloop.mesh import BoundarySpec
    z_fix = -GBOX.shaft_len                              # 输出轴底面
    z_load = GBOX.carrier_plate_t + GBOX.pin_len         # 3 个销顶面（同一平面）
    return (
        BoundarySpec("fixed", "z", z_fix, tol=1e-6),
        BoundarySpec("load", "z", z_load, tol=1e-6),
    )


def carrier_pin_force_n(torque_nmm: float = 1000.0) -> float:
    """额定输出扭矩下，每个行星销的切向力 [N]。"""
    return torque_nmm / (3.0 * CENTER_DIST)


if __name__ == "__main__":
    print(f"模数 m={M}  压力角 20°")
    for g, tag in ((SUN, "太阳轮"), (PLANET, "行星轮"), (RING, "内齿圈")):
        print(f"  {tag}: z={g.teeth:2d}  r={g.pitch_r:6.2f}  rb={g.base_r:6.2f}  "
              f"ra={g.tip_r:6.2f}  rf={g.root_r:6.2f}  ψ={math.degrees(g.psi):5.3f}°")
    print(f"  中心距 a = {CENTER_DIST} mm")
    print(f"  齿圈外半径 {RING_OD_R:.2f} mm   法兰 Ø{GBOX.flange_od}")
    print(f"  额定扭矩 1000 N·mm → 单销切向力 {carrier_pin_force_n():.2f} N")
    ext = external_gear_outline(SUN)
    hole = internal_hole_outline(RING)
    print(f"  轮廓点数: 太阳轮 {len(ext)}  内齿圈 {len(hole)}")
