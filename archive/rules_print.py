"""cadloop.rules_print — 可打印性检查。

.. deprecated:: 2026-09-13
    **本模块已停用，不要在新代码中 import。** 可打印性检查的职责已移交
    `text-to-cad <https://github.com/earthtojake/text-to-cad>`_ 的
    **``dfam-check``** 技能，理由：

    ==================  ==========================  ==========================
    维度                本模块（旧）                dfam-check（现行）
    ==================  ==========================  ==========================
    壁厚测量            **参数级**：调用方自报数字    **射线投射实测**：报 min / p05
                                                    / p25 / median 分位数
    悬垂判定            仅统计朝下面面积             同样统计，且给角度直方图 +
                                                    最大违规面坐标（可直接指导改设计）
    摆放优化            固定 2 个候选                 六向轴对齐候选 + 支撑面积对比
    装配体（多体）       不支持                       逐体拆分（避免把配合间隙误判成薄壁）
    工艺覆盖            仅 FDM                        FDM / SLS / SLA / 金属 PBF / MJF
    ==================  ==========================  ==========================

    现行做法：用 dfam-check 生成 JSON，再由 :mod:`cadloop.dfam` 翻译成判据，
    与 FEA 结论合流（见 ``pipeline.run_pipeline(dfam_json=...)``）。

    **本文件保留仅为历史参考**——它记录了「角度约定」等有用结论，
    且 FDM 工艺限值已对齐到 dfam-check 的 ISO/ASTM 引用表。
    物理删除需少爷批准（文件操作纪律：移动/删除先报计划）。

---

原模块说明（历史）
------------------
这些检查的作用是**在仿真之前**就把"物理上印不出来"的设计筛掉，
避免把算力花在无法制造的几何上。

角度定义（避免歧义，务必按此理解）
----------------------------------
对每个面取其法向 ``n`` 与打印方向 ``+Z``：

* ``nz = n·Z > 0`` → 面朝上，自支撑，无需支撑
* ``nz = 0``       → 竖直墙面，理想情况
* ``nz < 0``       → 面朝下，**可能需要支撑**

定义 **面-水平夹角 β**：

.. math::

    \\beta = \\arccos(-n_z) \\in [0°, 90°)

* β = 0°  → 完全水平朝下（悬垂最严重）
* β → 90° → 接近竖直（免支撑）

判据：``β < max_overhang_deg`` 时该面需要支撑。这与 PrusaSlicer / Bambu Studio
的「overhang threshold」语义一致；**限值已于 2026-09-13 由 60° 收紧到 45°**
（对齐 dfam-check 的 ISO/ASTM 引用值）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Check, PrintEnv
from .geom import GeometryError, apply_orientation, bed_contact_area, require_build123d

# ⚠️ 弃用标记必须放在 ``from __future__`` **之后**——``__future__`` 导入必须是
#    文件首个语句（docstring 除外），否则整个模块 SyntaxError 无法导入。
__deprecated__ = True
__deprecated_reason__ = "已被 text-to-cad 的 dfam-check 技能取代（2026-09-13 收缩决策）"
__replacement__ = "cadloop.dfam"


@dataclass
class FaceGroup:
    """一类面的统计。"""

    count: int = 0
    area_mm2: float = 0.0

    def add(self, area: float) -> None:
        self.count += 1
        self.area_mm2 += area


@dataclass
class PrintabilityReport:
    """单个摆放方向下的可打印性统计。"""

    orientation: str
    total_faces: int
    downward_faces: int
    overhang_faces: FaceGroup          # 需要支撑的面
    horizontal_faces: FaceGroup        # 水平朝下（β≈0，最难）
    vertical_faces: FaceGroup
    upward_faces: FaceGroup
    bed_contact_area_mm2: float
    bbox_mm: tuple[float, float, float]
    support_area_ratio: float          # 需支撑面积 / 零件总表面积
    max_beta_deg_needing_support: float
    note: str = ""

    def checks(self, env: PrintEnv, part_volume_mm3: float | None = None) -> list[Check]:
        out: list[Check] = []
        # 支撑面积占比：经验上超过 30% 意味着材料与时间成本激增
        out.append(
            Check(
                name="support_area_ratio",
                ok=self.support_area_ratio <= 0.30,
                value=self.support_area_ratio * 100.0,
                limit=30.0,
                unit="%",
                note="需支撑面积占零件总表面积的比例（经验阈值 30%）",
            )
        )
        # 床接触：接触面积过小 → 首层附着不足
        if part_volume_mm3:
            bbox_xy = max(self.bbox_mm[0] * self.bbox_mm[1], 1e-9)
            bed_ratio = self.bed_contact_area_mm2 / bbox_xy
            out.append(
                Check(
                    name="bed_adhesion",
                    ok=self.bed_contact_area_mm2 >= 25.0,
                    value=self.bed_contact_area_mm2,
                    limit=25.0,
                    unit="mm²",
                    comparison=">=",
                    note=f"接触面积 / 包围盒底面积 = {bed_ratio * 100:.1f}%",
                )
            )
        # 幅面
        out.append(env.printer.check_build_volume(self.bbox_mm))
        return out

    def summary(self) -> str:
        return (
            f"[{self.orientation}] 面数={self.total_faces}  朝下={self.downward_faces}  "
            f"需支撑={self.overhang_faces.count} 面 / {self.overhang_faces.area_mm2:.1f} mm²  "
            f"(占比 {self.support_area_ratio * 100:.1f}%)  "
            f"床接触={self.bed_contact_area_mm2:.1f} mm²"
        )


def analyze_orientation(
    part: Any,
    env: PrintEnv,
    orientation_name: str = "flat",
    rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> PrintabilityReport:
    """分析某一种打印摆放的可打印性。"""
    require_build123d()
    from build123d import Rot  # 延迟导入

    rx, ry, rz = rotation_deg
    oriented = Rot(rx, ry, rz) * part if any(abs(a) > 1e-9 for a in (rx, ry, rz)) else part

    thr = env.rules.max_overhang_deg
    overhang, horizontal, vertical, upward = FaceGroup(), FaceGroup(), FaceGroup(), FaceGroup()
    downward = 0
    total_area = 0.0
    total_faces = 0
    worst_beta = 90.0

    for f in oriented.faces():
        try:
            n = f.normal_at(f.center())
            area = float(f.area)
        except Exception:
            continue
        total_faces += 1
        total_area += area
        nz = float(n.Z)

        if nz > 1e-6:
            upward.add(area)
            continue

        if abs(nz) <= 0.05:
            # 接近竖直的墙面：β ≈ 90°，免支撑
            vertical.add(area)
            continue

        downward += 1
        beta = math.degrees(math.acos(max(0.0, min(1.0, -nz))))
        if beta < thr:
            overhang.add(area)
            worst_beta = min(worst_beta, beta)
        if beta < 10.0:
            horizontal.add(area)

    bb = oriented.bounding_box().size
    bbox = (float(bb.X), float(bb.Y), float(bb.Z))
    ratio = (overhang.area_mm2 / total_area) if total_area > 0 else 0.0

    orient_note = ""
    if worst_beta < 90.0:
        orient_note = f"最严峻悬垂面 β={worst_beta:.1f}°（阈值 {thr}°）"

    return PrintabilityReport(
        orientation=orientation_name,
        total_faces=total_faces,
        downward_faces=downward,
        overhang_faces=overhang,
        horizontal_faces=horizontal,
        vertical_faces=vertical,
        upward_faces=upward,
        bed_contact_area_mm2=bed_contact_area(oriented),
        bbox_mm=bbox,
        support_area_ratio=ratio,
        max_beta_deg_needing_support=worst_beta,
        note=orient_note,
    )


def compare_orientations(
    part: Any,
    env: PrintEnv,
    candidates: Sequence[tuple[str, Sequence[float]]] | None = None,
) -> list[PrintabilityReport]:
    """对比多种摆放方案，返回按"需支撑面积比例"升序排列的报告。

    这是**打印方向优化**的第一步：先做几何筛选，把支撑需求大的方向淘汰，
    再对剩下的候选做力学评估（见 fem 层）。
    """
    if candidates is None:
        candidates = (
            ("flat", (0.0, 0.0, 0.0)),
            ("upright", (90.0, 0.0, 0.0)),
            ("side", (0.0, 90.0, 0.0)),
        )
    reports = [analyze_orientation(part, env, name, rot) for name, rot in candidates]
    return sorted(reports, key=lambda r: r.support_area_ratio)


# ==============================================================================
# 薄壁 / 最小特征检查（参数级，几何级的厚度分析留作后续）
# ==============================================================================
def check_wall_thicknesses(env: PrintEnv, walls: dict[str, float]) -> list[Check]:
    """按配置检查一组壁厚参数。

    Parameters
    ----------
    walls : ``{"hinge_thickness": 0.8, "web": 1.2, ...}``
    """
    return [env.rules.check_wall(t, where=name) for name, t in walls.items()]


def check_features(env: PrintEnv, features: dict[str, float]) -> list[Check]:
    """按配置检查最小特征尺寸（孔/销/柱）。"""
    out: list[Check] = []
    for name, d in features.items():
        if name.startswith("hole"):
            out.append(env.rules.check_hole(d, where=name))
        elif name.startswith("pin"):
            out.append(env.rules.check_pin(d, where=name))
        else:
            out.append(env.rules.check_feature(d, where=name))
    return out
