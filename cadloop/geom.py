"""cadloop.geom — 几何度量与导出（FEA 管线的几何边界）。

2026-09-13 收缩后的职责
-----------------------
本模块只保留 FEA 管线**真正需要**的两件事：

* ``measure`` —— 从 build123d 零件提取体积/质量/包围盒（用于报告与打印估算）
* ``export``  —— 导出 STEP（CAD 交换 / 送 Fusion 查看）与 STL（切片、dfam-check）

原来还有两块内容，**已归档到 ``archive/geom_standard_parts.py``**：

* **标准件库**（``bearing_bore`` / ``screw_hole`` / ``heat_insert_pocket`` /
  ``magnet_pocket`` / ``hole_tool``）—— 被 text-to-cad 的 **``step-parts``**
  技能取代：那个技能能从 step.parts 目录直接下载厂商 STEP 模型（螺丝、轴承、
  电机、连接器），比我们按手册尺寸手搓圆柱精确得多，也省掉维护职责。
* **打印摆放分析**（``OrientationCandidate`` / ``apply_orientation`` /
  ``bed_contact_area``）—— 被 text-to-cad 的 **``dfam-check``** 取代：
  它给六个轴对齐候选 + 各自的支撑面积，是**射线投射实测**而非几何估计。

数据仍然以名义尺寸 ± printenv.yaml 为准的
-----------------------------------------
**建模一律按名义尺寸**（Ø6.00 就是 Ø6.00），公差补偿交给切片软件的
「XY 孔补偿」参数统一处理——这样 AI 参数化脚本不必为每个孔写补偿逻辑。

坐标与单位约定
--------------
* 单位 **mm**；全局 Z 轴向上。
* **FEA 永远在零件坐标系里算**；"层间方向"是打印坐标系的概念，
  由 ``fem.solve_static`` 的 ``layer_normal`` 参数映射（见 ``dfam.py`` 的说明）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Check, PrintEnv

# build123d 采用延迟导入：即使未安装，本模块的纯数据部分（PartMetrics）仍可被
# 引用，便于在没装 CAD 内核的环境里做配置/判据单元测试。
try:  # pragma: no cover
    from build123d import export_step, export_stl  # type: ignore

    _HAS_B123D = True
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
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
# 度量与导出
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

        ⚠️ 这是**粗略估算**（可信度层级 ``estimate``）；要准确值请走
        text-to-cad 的 ``gcode`` 技能用真实切片器出 G-code。
        """
        return self.volume_mm3 / efficiency_mm3_s * overhead_factor / 3600.0

    def __str__(self) -> str:
        return (
            f"{self.name}: V={self.volume_mm3:.1f} mm³  质量={self.mass_g:.2f} g  "
            f"包围盒={tuple(round(b, 2) for b in self.bbox_mm)} mm  solids={self.n_solids}"
        )


def measure(part: Any, env: PrintEnv, name: str = "part") -> PartMetrics:
    """提取零件几何指标。

    与 ``dfam.cross_check_volume`` 配套使用：trimesh 读 STL 算出的体积
    应与本函数从实体算出的体积一致，否则说明导出或网格化环节出了问题。
    """
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
    """导出 STEP（CAD 交换 / Fusion 查看）与 STL（切片、dfam-check）。

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
