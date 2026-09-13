"""cadloop.geom — 几何构建工具与标准件库。

为什么把标准件做成函数
----------------------
装配失败的头号原因是**孔的公差在多个零件之间不一致**。若让 LLM 每次现写
"Φ6 轴承孔"，30 个零件可能得到 30 种写法（有的减去 0.05，有的不减）。
把它们函数化后：

* 公差补偿只在一个地方定义（``PrintEnv.fits`` → 配合类型 → 实际孔径）
* 调用方只写 ``bearing_bore("MR63ZZ", fit="press")``，无需记忆数字
* 换打印机 / 换耗材 / 标定完成后，只改 ``printenv.yaml``

坐标与单位约定
--------------
* 单位 **mm**；全局 Z 轴向上；**打印摆放方向 = +Z**（即零件的 Z 向对应层间法向）。
* 因此："受力沿 Z" ≡ "层间受力"，强度取 ``interlayer``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import Check, PrintEnv

# build123d 采用延迟导入：即使未安装，本模块的纯数据部分（PartSpec 等）仍可被引用，
# 便于在没装 CAD 内核的环境里做配置/判据单元测试。
try:  # pragma: no cover
    from build123d import (  # type: ignore
        Align,
        Box,
        Cylinder,
        Part,
        Pos,
        Rot,
        export_step,
        export_stl,
    )

    _HAS_B123D = True
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
    Align = Box = Cylinder = Part = Pos = Rot = None  # type: ignore
    export_step = export_stl = None  # type: ignore
    _HAS_B123D = False
    _IMPORT_ERROR = _e


class GeometryError(RuntimeError):
    """几何构建失败（布尔运算失败、圆角失败、参数非法等）。"""


def require_build123d() -> None:
    if not _HAS_B123D:
        raise GeometryError(
            f"build123d 不可用（{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}）。"
            "请确认已激活 cadloop 环境：conda activate cadloop"
        )


# ==============================================================================
# 1. 通用刀具（布尔减）
# ==============================================================================
def hole_tool(
    diameter_mm: float,
    depth_mm: float,
    at: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "Z",
    start_at_origin: bool = True,
):
    """生成用于布尔减的圆柱刀具。

    Parameters
    ----------
    diameter_mm : 刀具直径（**已是最终孔径**，公差补偿由调用者通过 fits.hole_for 算好）
    depth_mm    : 刀具长度。做贯穿孔时请传 ``穿透长度 + 余量``，不要恰好等于壁厚，
                  否则会留下零厚度面导致布尔失败或网格退化。
    at          : 刀具起始点（若 ``start_at_origin``，此点为圆柱底面中心）
    axis        : ``"X"`` / ``"Y"`` / ``"Z"``，圆柱轴线方向
    """
    require_build123d()
    if diameter_mm <= 0 or depth_mm <= 0:
        raise GeometryError(f"刀具尺寸非法: d={diameter_mm}, depth={depth_mm}")
    if axis not in ("X", "Y", "Z"):
        raise GeometryError(f"axis 必须是 X/Y/Z，收到 {axis!r}")

    align = (Align.CENTER, Align.CENTER, Align.MIN) if start_at_origin else (Align.CENTER, Align.CENTER, Align.CENTER)
    tool = Cylinder(diameter_mm / 2.0, depth_mm, align=align)
    if axis == "X":
        tool = Rot(0, 90, 0) * tool
    elif axis == "Y":
        tool = Rot(90, 0, 0) * tool
    return Pos(*at) * tool


def through_hole(
    diameter_mm: float,
    thickness_mm: float,
    at: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "Z",
    clearance_mm: float = 1.0,
):
    """贯穿孔刀具：自动两侧各留 ``clearance_mm`` 余量。

    避免"刀具长度恰好等于壁厚"这一常见陷阱（会生成退化面）。
    """
    total = thickness_mm + 2.0 * clearance_mm
    start = list(at)
    idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    start[idx] -= clearance_mm
    return hole_tool(diameter_mm, total, tuple(start), axis, start_at_origin=True)


# ==============================================================================
# 2. 标准件库（公差在本模块内统一施加）
# ==============================================================================
def bearing_bore(env: PrintEnv, bearing_name: str, fit: str = "press", depth_mm: float | None = None):
    """按标准件手册生成轴承座孔刀具。

    Parameters
    ----------
    bearing_name : printenv.yaml ``hardware.bearings`` 中的 name（如 ``MR63ZZ``）
    fit          : ``press``（过盈压入，孔略小于外径）/ ``rotating``（轴在孔内转）
                   / ``sliding``
    depth_mm     : 座孔深度；``None`` 时取轴承宽度
    """
    require_build123d()
    spec = _find_hardware(env, "bearings", bearing_name)
    od = float(spec["od_mm"])
    width = float(spec["width_mm"])
    bore_d = env.fits.hole_for(od, fit)
    depth = width if depth_mm is None else float(depth_mm)
    return hole_tool(bore_d, depth, start_at_origin=True)


def screw_hole(env: PrintEnv, size: str = "M3", clearance: bool = True, depth_mm: float = 10.0):
    """螺丝孔刀具。

    ``clearance=True``  → 通孔（孔径 = 名义 + 间隙）
    ``clearance=False`` → 底孔（用于自攻，孔径 ≈ 0.8 × 名义）
    """
    require_build123d()
    nominal = _screw_nominal(size)
    d = nominal + 0.2 if clearance else nominal * 0.8
    return hole_tool(d, depth_mm, start_at_origin=True)


def heat_insert_pocket(env: PrintEnv, size: str = "M3", depth_mm: float = 6.0):
    """热熔铜嵌件（heat-set insert）座孔刀具。

    经验：座孔直径 ≈ 嵌件外径 − 0.1 mm，深度 ≈ 嵌件长度 + 0.3 mm。
    这里按 M3 常见嵌件（外径 4.6 mm，长 5.7 mm）计算，可被 printenv 覆盖。
    """
    require_build123d()
    ins = env.hardware.get("heat_insert_specs", {}) or {}
    spec = ins.get(size) or {"od_mm": 4.6, "length_mm": 5.7}
    d = float(spec["od_mm"]) - 0.1
    h = depth_mm if depth_mm else float(spec["length_mm"]) + 0.3
    return hole_tool(d, h, start_at_origin=True)


def magnet_pocket(env: PrintEnv, name: str = "D6x2"):
    """磁铁座孔刀具（过盈 0.05 mm）。"""
    require_build123d()
    spec = _find_hardware(env, "magnets", name)
    od = float(spec["od_mm"])
    th = float(spec["thickness_mm"])
    return hole_tool(od - 0.05, th + 0.2, start_at_origin=True)


def _find_hardware(env: PrintEnv, group: str, name: str) -> dict:
    items = env.hardware.get(group) or []
    for it in items:
        if str(it.get("name", "")).lower() == name.lower():
            return it
    raise GeometryError(
        f"标准件 {name!r} 不在 hardware.{group} 中；可选 "
        f"{[i.get('name') for i in items]}"
    )


def _screw_nominal(size: str) -> float:
    s = size.strip().upper().lstrip("M")
    try:
        return float(s)
    except ValueError as e:
        raise GeometryError(f"无法解析螺丝规格 {size!r}（期望形如 'M3'）") from e


# ==============================================================================
# 3. 度量与导出
# ==============================================================================
@dataclass
class PartMetrics:
    """零件的量化指标（几何侧，不含力学）。"""

    name: str
    volume_mm3: float
    mass_g: float
    bbox_mm: tuple[float, float, float]
    n_solids: int

    def fits_in_printer(self, env: PrintEnv) -> Check:
        return env.printer.check_build_volume(self.bbox_mm)

    def needs_split(self, env: PrintEnv) -> bool:
        return env.printer.needs_split(self.bbox_mm, env.rules.split_threshold_mm)

    def estimated_print_hours(self, env: PrintEnv, efficiency_mm3_s: float = 15.0,
                              overhead_factor: float = 1.4) -> float:
        """打印时长估算。

        t ≈ V / v_eff × overhead / 3600
        ``v_eff`` 为实际平均体积流量（峰值 40 mm³/s，实际受冷却/换向/外墙减速限制，
        典型 10–20），``overhead`` 补偿行程空耗。本机标定后应替换为实测值。
        """
        return self.volume_mm3 / efficiency_mm3_s * overhead_factor / 3600.0

    def __str__(self) -> str:
        return (
            f"{self.name}: V={self.volume_mm3:.1f} mm³  质量={self.mass_g:.2f} g  "
            f"包围盒={tuple(round(b, 2) for b in self.bbox_mm)} mm  solids={self.n_solids}"
        )


def measure(part: Any, env: PrintEnv, name: str = "part") -> PartMetrics:
    """提取零件几何指标。"""
    require_build123d()
    bb = part.bounding_box()
    size = (bb.size.X, bb.size.Y, bb.size.Z)
    vol = float(part.volume)
    solids = part.solids() if hasattr(part, "solids") else []
    return PartMetrics(
        name=name,
        volume_mm3=vol,
        mass_g=vol * env.material.density_g_mm3(),
        bbox_mm=size,
        n_solids=len(solids) if solids is not None else 1,
    )


def export(part: Any, out_dir: str | Path, stem: str = "part") -> dict[str, str]:
    """导出 STEP（CAD 交换）与 STL（切片/打印）。

    返回 ``{"step": path, "stl": path}``。
    """
    require_build123d()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step_p, stl_p = out / f"{stem}.step", out / f"{stem}.stl"
    try:
        export_step(part, str(step_p))
    except Exception as e:  # pragma: no cover
        raise GeometryError(f"STEP 导出失败: {type(e).__name__}: {e}") from e
    try:
        export_stl(part, str(stl_p))
    except Exception as e:  # pragma: no cover
        raise GeometryError(f"STL 导出失败: {type(e).__name__}: {e}") from e
    return {"step": str(step_p), "stl": str(stl_p)}


# ==============================================================================
# 4. 打印摆放分析（差异化的起点）
# ==============================================================================
@dataclass
class OrientationCandidate:
    """一种打印摆放方案。"""

    name: str
    rotation_deg: tuple[float, float, float]
    note: str = ""


DEFAULT_ORIENTATIONS: tuple[OrientationCandidate, ...] = (
    OrientationCandidate("flat", (0, 0, 0), "平放：零件主平面 = 打印 XY 面（层内受力）"),
    OrientationCandidate("upright", (90, 0, 0), "竖放：零件主法向 = 打印 Z 向（层间受力）"),
    OrientationCandidate("side", (0, 90, 0), "侧躺：绕 Y 轴翻转"),
)


def apply_orientation(part: Any, cand: OrientationCandidate):
    """按候选方案旋转零件（用于评估不同摆放下的支撑需求与截面积）。"""
    require_build123d()
    rx, ry, rz = cand.rotation_deg
    return Rot(rx, ry, rz) * part


def bed_contact_area(part: Any, tol_mm: float = 0.05) -> float:
    """零件与打印板（Z=0 平面）的接触面积近似 [mm²]。

    实现方式：统计所有面中「法向朝下且位于最低处」的面的面积总和。
    接触面积过小 → 首层附着不足 → 打印失败风险。
    """
    require_build123d()
    zmin = part.bounding_box().min.Z
    total = 0.0
    for f in part.faces():
        try:
            if abs(f.center().Z - zmin) > tol_mm:
                continue
            n = f.normal_at(f.center())
            if n.Z < -0.9:  # 朝下的水平面
                total += float(f.area)
        except Exception:
            continue
    return total
