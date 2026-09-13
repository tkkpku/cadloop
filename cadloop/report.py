"""cadloop.report — 结果报告与可信度分级。

可信度分级（借鉴 cad-cae-copilot 的做法）
---------------------------------------
每个数字都必须能回答"这句话凭什么可信"。本模块给每个指标打一个层级标签，
低层级结果**不得**冒充高层级：

============ ================================================
层级          含义
============ ================================================
``solver``    求解器实际算出来的场量（位移、应力）
``analytic``  与闭式解析解的对照（音叉、梁理论）
``geometry``  纯几何量（体积、包围盒、悬垂面积）
``estimate``  经验公式估算（打印时长、质量、成本）
``screening`` 各向同性筛选级判断（非认证级）
============ ================================================

另外，当 ``PrintEnv.calibrated`` 为假（材料/公差未实测标定）时，
报告强制标注"仅供参照"，且 ``allow_conclusive_claims`` 相关的结论措辞降级。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Check, PrintEnv

CREDIBILITY_SOLVER = "solver"
CREDIBILITY_ANALYTIC = "analytic"
CREDIBILITY_GEOMETRY = "geometry"
CREDIBILITY_ESTIMATE = "estimate"
CREDIBILITY_SCREENING = "screening"

TIER_ORDER = {
    CREDIBILITY_SOLVER: 4,
    CREDIBILITY_ANALYTIC: 3,
    CREDIBILITY_GEOMETRY: 2,
    CREDIBILITY_ESTIMATE: 1,
    CREDIBILITY_SCREENING: 0,
}


@dataclass
class ReportedValue:
    """带可信度标签的单个指标。"""

    name: str
    value: float
    unit: str
    tier: str
    note: str = ""

    def __str__(self) -> str:
        return f"{self.name} = {self.value:.4g} {self.unit} [{self.tier}]" + (
            f"  ({self.note})" if self.note else ""
        )


@dataclass
class Report:
    """一次运行的结构化报告。"""

    title: str
    env: PrintEnv
    part_name: str = "part"
    status: str = "unknown"
    values: list[ReportedValue] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ---- 汇总 ---------------------------------------------------------------
    @property
    def all_checks_passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def verdict(self, need_conclusive: bool = True) -> str:
        """结论措辞。

        三种状态必须说三种话，不能混（2026-09-13 修正：此前把「保守兜底」
        也说成「标定完备」，属于**措辞失真**——保守值不是实测值）：

        * ``measured``             → 材料与公差实测 → 结论可作为设计依据
        * ``conservative_fallback`` → 参数被**故意低估**（注塑值×0.8，层间×0.5，
          SF=2.0）→ 结论偏安全、可用，但**必须标注"非实测"**
        * 两者皆非                 → 只有参照价值，不得作为制造依据
        """
        env = self.env
        strategy = env.calibration_strategy
        if self.all_checks_passed:
            if strategy == "measured" and env.allow_conclusive_claims:
                return "全部判据通过（材料与公差均已实测标定，结论可作为设计依据）"
            if strategy == "conservative_fallback":
                return ("全部判据通过 —— ⚠️ 材料参数为**保守兜底值（非实测）**："
                        "注塑值×0.8、层间再×0.5、SF=2.0；结论偏安全，"
                        "但裕度落在敏感区(S<2.5)的零件仍建议实测复核")
            if not need_conclusive:
                return "全部判据通过（未启用结论性判定，仅作参考）"
            return "全部判据通过 ⚠️ 但材料/公差尚未标定，结论仅供**参照**，不得作为制造依据"
        names = ", ".join(c.name for c in self.failed_checks)
        return f"有 {len(self.failed_checks)} 项判据未通过: {names}"

    # ---- 渲染 ---------------------------------------------------------------
    def to_markdown(self) -> str:
        L: list[str] = []
        L.append(f"# {self.title}")
        L.append("")
        L.append(f"- 零件：`{self.part_name}`")
        L.append(f"- 时间：{self.timestamp}")
        L.append(f"- 状态：**{self.status}**")
        L.append(f"- 边界条件：`{self.env.meta.get('name', '?')}` "
                 f"v{self.env.meta.get('version', '?')}（calibrated={self.env.calibrated}）")
        L.append("")

        L.append("## 结论")
        L.append("")
        L.append(self.verdict())
        L.append("")

        if self.values:
            L.append("## 指标（含可信度层级）")
            L.append("")
            L.append("| 指标 | 数值 | 单位 | 可信度 | 说明 |")
            L.append("| :--- | ---: | :--- | :--- | :--- |")
            for v in sorted(self.values, key=lambda x: -TIER_ORDER.get(x.tier, 0)):
                L.append(f"| {v.name} | {v.value:.4g} | {v.unit} | `{v.tier}` | {v.note} |")
            L.append("")

        if self.checks:
            n_ok = sum(1 for c in self.checks if c.ok)
            L.append(f"## 判据（{n_ok}/{len(self.checks)} 通过）")
            L.append("")
            L.append("| 判据 | 结果 | 实测 | 限值 | 方向 | 说明 |")
            L.append("| :--- | :--: | ---: | ---: | :--: | :--- |")
            for c in self.checks:
                flag = "✅" if c.ok else "❌"
                L.append(
                    f"| {c.name} | {flag} | {c.value:.4g} {c.unit} | "
                    f"{c.limit:.4g} {c.unit} | {c.comparison} | {c.note} |"
                )
            L.append("")

        for head, body in self.sections:
            L.append(f"## {head}")
            L.append("")
            L.append(body)
            L.append("")

        gaps = self.env.blocking_gaps()
        if gaps:
            L.append("## ⚠️ 未标定项（影响结论强度）")
            L.append("")
            for g in gaps:
                L.append(f"- {g}")
            L.append("")

        if self.notes:
            L.append("## 备注")
            L.append("")
            for n in self.notes:
                L.append(f"- {n}")
            L.append("")

        if self.artifacts:
            L.append("## 产物")
            L.append("")
            L.append("| 类型 | 路径 |")
            L.append("| :--- | :--- |")
            for k, v in self.artifacts.items():
                L.append(f"| {k} | `{v}` |")
            L.append("")

        return "\n".join(L)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        return p


# ==============================================================================
# 便捷构造
# ==============================================================================
def value(name: str, v: float, unit: str, tier: str, note: str = "") -> ReportedValue:
    return ReportedValue(name=name, value=float(v), unit=unit, tier=tier, note=note)


def checks_table(checks: Sequence[Check]) -> str:
    """把判据列表渲染成 Markdown 表格（用于 section 内嵌）。"""
    if not checks:
        return "_（无）_"
    lines = ["| 判据 | 结果 | 实测 | 限值 | 说明 |", "| :--- | :--: | ---: | ---: | :--- |"]
    for c in checks:
        lines.append(
            f"| {c.name} | {'✅' if c.ok else '❌'} | {c.value:.4g} {c.unit} | "
            f"{c.limit:.4g} {c.unit} | {c.note} |"
        )
    return "\n".join(lines)
