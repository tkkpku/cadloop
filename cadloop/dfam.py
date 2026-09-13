"""cadloop.dfam — 消费 text-to-cad 的 ``dfam-check`` 输出。

定位（2026-09-13 收缩后）
------------------------
cadloop **不再自己实现可打印性检查**。理由：

* ``earthtojake/text-to-cad`` 的 ``dfam-check`` 用**射线投射实测**壁厚
  （报 ``min`` / ``p05`` / ``p25`` / ``median`` 分位数），而本仓库原先的
  ``rules_print.py`` 只是**参数级**检查（调用方自报壁厚数字），信息量差一个量级；
* 它的悬垂/支撑体积分析、六个轴向摆放候选、逐体（assembly）拆分都已成熟；
* 重复实现等于维护两份会漂移的判据。

所以本模块只做一件事：**把 dfam-check 的 JSON 翻译成 cadloop 的 ``Check``**，
让「可制造性结论」与「结构强度结论」合流进同一份报告。

数据来源（实测 schema，2026-09-13 跑 ``dfam_tool.py measure`` 得到）
-------------------------------------------------------------------
::

    {
      "file": "...",
      "mesh":   {"bbox_mm": [x,y,z], "volume_mm3", "surface_area_mm2",
                 "triangle_count", "watertight", "euler_number", "body_count"},
      "scale":  {"bbox_diagonal_mm", "units_suspect", "note"},
      "overhangs": {"angle_limit_used_deg", "down_facing_area_below_limit_mm2",
                    "down_facing_area_below_limit_pct", "face_count_below_limit",
                    "down_facing_angle_histogram_mm2", "largest_faces_below_limit"},
      "wall_thickness": {"samples_valid", "min_mm", "p05_mm", "p25_mm",
                         "median_mm", "max_mm", "thinnest_samples", ["per_body"]},
      "support_volume": {"angle_limit_used_deg", "estimated_support_volume_mm3",
                         "part_volume_mm3", "support_to_part_ratio_pct", "method"}
    }

判定策略（遵循 dfam-check 的 SKILL.md 约定）
--------------------------------------------
* ``mesh.watertight == false`` → **一票否决**（任何工艺都无法切片），排在最前。
* ``scale.units_suspect == true`` → **一票否决**：单位存疑时所有尺寸都无意义，
  且"朝下面全贴在板上"会让悬垂/支撑读数退化为 0。
* 壁厚：``p05_mm`` 低于限值即视为违规（``min_mm`` 可能是采样离群值），
  两者都要报出来。
* 支撑体积比：**只作成本信号**，不作硬失败（其文档明言是粗略上界估计），
  除非调用方给了显式预算。
* 粉末工艺（SLS/MJF）不适用悬垂判据，改为"困粉逃逸"检查——本工具不测，
  应报 ``need more info`` 而非推断。本模块只服务 FDM。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import Check, PrintEnv

# 支撑体积比的经验阈值：超过此值通常值得重新定向或改设计（dfam-check 文档）
SUPPORT_RATIO_GUIDELINE_PCT = 30.0


class DfamError(RuntimeError):
    """dfam-check 报告缺失、格式不符或与当前工艺不匹配。"""


@dataclass
class DfamReport:
    """解析后的 dfam-check 报告。"""

    path: str
    raw: dict
    file: str = ""
    bbox_mm: tuple[float, ...] = ()
    volume_mm3: float | None = None
    watertight: bool = True
    body_count: int = 1
    units_suspect: bool = False
    wall_min_mm: float | None = None
    wall_p05_mm: float | None = None
    wall_median_mm: float | None = None
    wall_samples: int = 0
    overhang_area_mm2: float = 0.0
    overhang_pct: float = 0.0
    angle_limit_deg: float = 45.0
    support_ratio_pct: float = 0.0
    support_volume_mm3: float = 0.0

    # ---- 派生 ---------------------------------------------------------------
    @property
    def is_fdm_report(self) -> bool:
        """报告是否看起来是按 FDM 参数生成的（有悬垂区块即算）。"""
        return "overhangs" in self.raw

    def limit_mismatch(self, env: PrintEnv, tol: float = 1.0) -> str | None:
        """检查生成报告时用的自支撑角是否与当前配置一致。

        不一致意味着报告是拿另一套工艺参数测的，判据不可直接采信。
        """
        want = env.rules.max_overhang_deg
        got = self.angle_limit_deg
        if abs(got - want) > tol:
            return (f"dfam 报告用 --angle-limit {got:g}° 生成，"
                    f"当前配置 max_overhang_deg={want:g}°；请用 {want:g}° 重跑 dfam_tool")
        return None


def load_report(path: str | Path) -> DfamReport:
    """读取 ``dfam_tool.py measure`` 产出的 JSON。"""
    p = Path(path)
    if not p.exists():
        raise DfamError(f"dfam 报告不存在: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise DfamError(f"dfam 报告解析失败（{p}）: {type(e).__name__}: {e}") from e
    if not isinstance(raw, dict) or "mesh" not in raw:
        raise DfamError(f"dfam 报告缺少 mesh 区块，疑似格式不符: {p}")

    mesh = raw.get("mesh") or {}
    scale = raw.get("scale") or {}
    ov = raw.get("overhangs") or {}
    wt = raw.get("wall_thickness") or {}
    sv = raw.get("support_volume") or {}

    return DfamReport(
        path=str(p),
        raw=raw,
        file=str(raw.get("file", "")),
        bbox_mm=tuple(float(v) for v in (mesh.get("bbox_mm") or ())),
        volume_mm3=(None if mesh.get("volume_mm3") is None else float(mesh["volume_mm3"])),
        watertight=bool(mesh.get("watertight", False)),
        body_count=int(mesh.get("body_count", 1)),
        units_suspect=bool(scale.get("units_suspect", False)),
        wall_min_mm=(None if wt.get("min_mm") is None else float(wt["min_mm"])),
        wall_p05_mm=(None if wt.get("p05_mm") is None else float(wt["p05_mm"])),
        wall_median_mm=(None if wt.get("median_mm") is None else float(wt["median_mm"])),
        wall_samples=int(wt.get("samples_valid", 0)),
        overhang_area_mm2=float(ov.get("down_facing_area_below_limit_mm2", 0.0) or 0.0),
        overhang_pct=float(ov.get("down_facing_area_below_limit_pct", 0.0) or 0.0),
        angle_limit_deg=float(ov.get("angle_limit_used_deg", 45.0) or 45.0),
        support_ratio_pct=float(sv.get("support_to_part_ratio_pct", 0.0) or 0.0),
        support_volume_mm3=float(sv.get("estimated_support_volume_mm3", 0.0) or 0.0),
    )


def checks_from_report(rep: DfamReport, env: PrintEnv) -> tuple[list[Check], list[str]]:
    """把 dfam 报告翻译成 cadloop 判据。

    Returns
    -------
    (checks, notes)
        ``checks`` 是会进报告判据表的条目；``notes`` 是只展示不判定的信息
        （例如支撑体积比这类"成本信号"）。
    """
    checks: list[Check] = []
    notes: list[str] = []

    # --- 1. 一票否决项（排最前）--------------------------------------------
    checks.append(
        Check(
            "dfam_watertight",
            rep.watertight,
            float(rep.watertight),
            1.0,
            "",
            ">=",
            "非水密网格无法切片（dfam-check 实测）",
        )
    )
    checks.append(
        Check(
            "dfam_units_ok",
            not rep.units_suspect,
            float(not rep.units_suspect),
            1.0,
            "",
            ">=",
            "单位存疑时所有尺寸与悬垂读数都不可信；先确认 STL 单位",
        )
    )

    # --- 2. 幅面 -------------------------------------------------------------
    if rep.bbox_mm:
        checks.append(env.printer.check_build_volume(rep.bbox_mm))

    # --- 3. 壁厚（p05 为主判据，min 仅供参考）-------------------------------
    r = env.rules
    if rep.wall_p05_mm is not None:
        checks.append(
            Check(
                "dfam_wall_p05",
                rep.wall_p05_mm >= r.min_wall_mm,
                rep.wall_p05_mm,
                r.min_wall_mm,
                "mm",
                ">=",
                f"射线投射实测壁厚 p05（样本 {rep.wall_samples}；"
                f"min={rep.wall_min_mm}，median={rep.wall_median_mm}）",
            )
        )
    elif rep.wall_min_mm is not None:
        checks.append(
            Check(
                "dfam_wall_min",
                rep.wall_min_mm >= r.min_wall_mm,
                rep.wall_min_mm,
                r.min_wall_mm,
                "mm",
                ">=",
                "仅有 min 值（无分位数），可能是采样离群值",
            )
        )
    else:
        notes.append("dfam 报告未给出壁厚样本（可能网格退化），壁厚未判")

    # --- 4. 悬垂/支撑（信息项，不作硬失败）----------------------------------
    notes.append(
        f"悬垂：{rep.overhang_area_mm2:.1f} mm² 朝下面积低于自支撑角 "
        f"{rep.angle_limit_deg:g}°（占表面积 {rep.overhang_pct:.1f}%）"
    )
    notes.append(
        f"支撑体积：{rep.support_volume_mm3:.1f} mm³（占零件 "
        f"{rep.support_ratio_pct:.1f}%，估算方式：面心到打印板棱柱上界）"
    )
    if rep.support_ratio_pct > SUPPORT_RATIO_GUIDELINE_PCT:
        notes.append(
            f"⚠️ 支撑占比超过经验阈值 {SUPPORT_RATIO_GUIDELINE_PCT:.0f}%"
            "——属**成本信号**（dfam-check 文档明确不作硬失败）；"
            "建议跑 `dfam_tool.py orientations` 找更省支撑的摆放"
        )

    # --- 5. 与当前工艺参数的一致性 ------------------------------------------
    mismatch = rep.limit_mismatch(env)
    if mismatch:
        notes.append(f"⚠️ {mismatch}")

    return checks, notes


def summarize(rep: DfamReport) -> str:
    """一行摘要。"""
    bb = "×".join(f"{v:.2f}" for v in rep.bbox_mm) if rep.bbox_mm else "?"
    vol = "?" if rep.volume_mm3 is None else f"{rep.volume_mm3:.1f}"
    return (
        f"dfam-check: {Path(rep.path).name}  bbox={bb} mm  V={vol} mm³  "
        f"watertight={rep.watertight}  bodies={rep.body_count}  "
        f"wall p05={rep.wall_p05_mm} / min={rep.wall_min_mm} mm  "
        f"support={rep.support_ratio_pct:.1f}%"
    )


def cross_check_volume(rep: DfamReport, volume_mm3: float, tol: float = 0.01) -> Check:
    """用 dfam 的网格体积交叉校验 cadloop 的几何体积。

    两个独立工具（trimesh 读 STL vs build123d 算实体）给出同一体积，
    说明**几何建模 → 导出 → 网格化**这一段没有丢材料或串坐标。
    这是本仓库"音叉"精神在几何侧的延伸。
    """
    if rep.volume_mm3 is None or volume_mm3 <= 0:
        return Check("dfam_volume_crosscheck", False, 0.0, tol, "", "<=",
                     "缺少可比体积（dfam 未给 volume 或几何体积非法）")
    rel = abs(rep.volume_mm3 - volume_mm3) / volume_mm3
    return Check(
        "dfam_volume_crosscheck",
        rel <= tol,
        rel * 100.0,
        tol * 100.0,
        "%",
        "<=",
        f"dfam(trimesh)={rep.volume_mm3:.2f} vs cadloop(build123d)={volume_mm3:.2f} mm³",
    )


if __name__ == "__main__":  # python -m cadloop.dfam <report.json>
    import sys

    from .config import load_env

    if len(sys.argv) < 2:
        print("用法: python -m cadloop.dfam <dfam_report.json>")
        raise SystemExit(2)
    _env = load_env()
    r = load_report(sys.argv[1])
    print(summarize(r))
    print()
    cs, ns = checks_from_report(r, _env)
    for c in cs:
        print(" ", c)
    print()
    for n in ns:
        print("  ·", n)
