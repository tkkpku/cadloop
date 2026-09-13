"""cadloop.robustness — 结论稳健性分析与屈曲检查。

本模块把四轮踩坑的教训固化成**通道的自动判据**，而不是文档里的一句话。

教训一：材料参数对结果到底有多敏感？
------------------------------------
判据不是「准不准」，而是**「估计错了会不会翻转结论」**：

.. math::
    S = \\frac{\\sigma_{\\text{allow}}}{\\sigma_{\\text{work}}}
    \\quad\\Longrightarrow\\quad
    \\text{可容忍的参数偏差} = \\frac{S}{S_{\\text{target}}}

* 实测数据表明：**块状结构**（机架、底板、齿轮）工作应力常只有 1–6 MPa，
  相对许用 20 MPa 有 3–20 倍裕度 → 材料参数估错 2 倍都不影响结论。
* 而**薄壁 / 细杆 / 柔性件**裕度常在 1–2 倍 → 此时参数估计必须可靠。

所以流程是：**默认用保守兜底值跑一遍 → 看裕度落在哪个区 → 只有敏感区才触发实测**。
这把「要不要打试件」从主观判断变成了机械判据。

教训二：细长受压件的真杀手是屈曲，不是强度
------------------------------------------
本机验算：一根 100 mm 长、5×5 mm 截面的立柱，**受压 31 N 就屈曲**，
而按强度算能承几百牛。若只做强度校核，会得出「安全」的错误结论。
故本模块提供欧拉屈曲的解析速查（4 种常见约束组合）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .config import Check, PrintEnv

# ==============================================================================
# 一、结论稳健性
# ==============================================================================
# 裕度分区（S = 许用 / 工作）
ZONE_ROBUST = "robust"            # S ≥ 4：材料参数完全不敏感
ZONE_ACCEPTABLE = "acceptable"    # 3 ≤ S < 4：稳健
ZONE_WATCH = "watch"              # 2.5 ≤ S < 3：可接受，留意薄壁件
ZONE_SENSITIVE = "sensitive"      # 1.5 ≤ S < 2.5：**敏感 → 建议实测复核**
ZONE_CRITICAL = "critical"        # S < 1.5：**必须实测 或 改设计**

ZONE_THRESHOLDS = (
    (4.0, ZONE_ROBUST),
    (3.0, ZONE_ACCEPTABLE),
    (2.5, ZONE_WATCH),
    (1.5, ZONE_SENSITIVE),
    (0.0, ZONE_CRITICAL),
)

ZONE_ADVICE = {
    ZONE_ROBUST: "材料参数不敏感：即使实际强度仅为估计的 1/4 仍安全，无需实测",
    ZONE_ACCEPTABLE: "稳健：参数估计误差不改变结论",
    ZONE_WATCH: "可接受，处于保守兜底策略的设计目标区（SM ≈ 2）",
    ZONE_SENSITIVE: "⚠️ 敏感区：建议打材料试件实测该方向强度，或加大截面",
    ZONE_CRITICAL: "🔴 临界：必须实测材料参数，或重新设计（加厚/改结构/降载）",
}


@dataclass
class RobustnessVerdict:
    """单个判据的稳健性结论。"""

    name: str
    work_value: float
    allow_value: float
    unit: str
    margin_factor: float                 # S = allow / work
    zone: str
    tolerance_ratio: float               # 可容忍的参数偏差倍数（相对当前估计值）
    advice: str = ""

    @property
    def needs_measurement(self) -> bool:
        return self.zone in (ZONE_SENSITIVE, ZONE_CRITICAL)

    def as_check(self) -> Check:
        """转成通用 Check（敏感区即视为"需要注意"，但不算失败）。"""
        return Check(
            name=f"robustness[{self.name}]",
            ok=self.zone not in (ZONE_CRITICAL,),
            value=self.margin_factor,
            limit=1.5,
            unit="×",
            comparison=">=",
            note=f"{self.zone}；可容忍参数偏差 {self.tolerance_ratio:.2f}×；{self.advice}",
        )

    def __str__(self) -> str:
        flag = {"robust": "OK  ", "acceptable": "OK  ", "watch": "OK  ",
                "sensitive": "WARN", "critical": "FAIL"}[self.zone]
        return (
            f"[{flag}] {self.name}: S = {self.margin_factor:5.2f}×  "
            f"({self.work_value:.2f} / {self.allow_value:.2f} {self.unit})  "
            f"zone={self.zone}  可容忍参数偏差 {self.tolerance_ratio:.2f}×"
        )


def assess(
    name: str,
    work_value: float,
    allow_value: float,
    unit: str = "MPa",
    design_target: float = 2.0,
) -> RobustnessVerdict:
    """评估单个量的稳健性。

    Parameters
    ----------
    work_value : 工作值（应力等）
    allow_value : 许用值（已含安全系数）
    design_target : 保守兜底策略的设计目标裕度。S 低于此值即落入敏感区。
    """
    if work_value <= 0:
        return RobustnessVerdict(name, work_value, allow_value, unit,
                                 float("inf"), ZONE_ROBUST, float("inf"), ZONE_ADVICE[ZONE_ROBUST])
    S = allow_value / work_value
    zone = next(z for th, z in ZONE_THRESHOLDS if S >= th)
    # 可容忍的参数偏差：只考虑工作值随材料"强度"线性变化的量
    # （强度估计偏高 x 倍 → 工作应力不变但许用变，故 S ∝ 1/x）
    tolerance = S / design_target
    return RobustnessVerdict(name, work_value, allow_value, unit, S, zone, tolerance,
                             ZONE_ADVICE[zone])


def assess_material_sensitivity(
    env: PrintEnv,
    nominal_allow_mpa: float,
    uncertainty_ratio: float = 0.5,
) -> dict[str, float]:
    """材料参数不确定性下的安全系数带（±``uncertainty_ratio``）。

    用途：回答"如果实际材料只有（或有）估计值的 X 倍，结论会怎样"。
    返回 ``{"S_low":..., "S_nominal":..., "S_high":..., "flips_below":...}``。
    """
    S_nom = nominal_allow_mpa
    S_low = nominal_allow_mpa * (1.0 - uncertainty_ratio)
    S_high = nominal_allow_mpa * (1.0 + uncertainty_ratio)
    return {
        "S_low": S_low,
        "S_nominal": S_nom,
        "S_high": S_high,
        "flips_below": 1.0 / max(S_nom, 1e-12),   # 参数偏高多少倍会导致 S<1
    }


# ==============================================================================
# 二、屈曲（细长受压件的真杀手）
# ==============================================================================
# 欧拉屈曲临界载荷：  Pcr = π²EI / (K L)²
# K = 有效长度系数，取决于两端约束：
EULER_K = {
    "pinned-pinned": 1.0,      # 两端铰支
    "fixed-free": 2.0,         # 一端固定一端自由（悬臂柱）—— 最不利
    "fixed-pinned": 0.7,       # 一端固定一端铰支
    "fixed-fixed": 0.5,        # 两端固定
}


def euler_buckling_load(
    length_mm: float,
    I_mm4: float,
    E_mpa: float,
    end_condition: str = "fixed-free",
) -> float:
    """欧拉屈曲临界载荷 [N]。

    .. math::
        P_{cr} = \\frac{\\pi^2 E I}{(K L)^2}

    ⚠️ 只在弹性范围内成立。若屈曲应力超过屈服强度，应改用非弹性屈曲
    （切线模量理论），本函数会通过 ``buckling_stress`` 提示该情形。
    """
    if length_mm <= 0 or I_mm4 <= 0 or E_mpa <= 0:
        raise ValueError("length/I/E 必须为正")
    K = EULER_K.get(end_condition)
    if K is None:
        raise ValueError(f"未知约束 {end_condition!r}；可选 {sorted(EULER_K)}")
    return math.pi**2 * E_mpa * I_mm4 / (K * length_mm) ** 2


def rect_inertia(b_mm: float, h_mm: float) -> float:
    """矩形截面惯性矩（绕 b 方向的中性轴，h 为弯曲方向高度）。"""
    return b_mm * h_mm**3 / 12.0


def buckling_stress(Pcr_n: float, area_mm2: float) -> float:
    """临界屈曲应力 [MPa]。若接近或超过屈服强度，说明欧拉公式已不适用。"""
    return Pcr_n / area_mm2 if area_mm2 > 0 else float("inf")


def check_buckling(
    env: PrintEnv,
    name: str,
    length_mm: float,
    b_mm: float,
    h_mm: float,
    axial_load_n: float,
    end_condition: str = "fixed-free",
    orientation: str = "in_layer",
) -> tuple[Check, dict]:
    """受压杆件屈曲校核。

    Returns
    -------
    (Check, info) — info 含 Pcr、安全系数、以及欧拉适用性提示。
    """
    I = rect_inertia(b_mm, h_mm)
    area = b_mm * h_mm
    E = env.material.E(orientation)
    Pcr = euler_buckling_load(length_mm, I, E, end_condition)
    sig_cr = buckling_stress(Pcr, area)
    sig_y = env.material.yield_strength(orientation)
    sf = env.physics.safety_factor

    info = {
        "Pcr_n": Pcr,
        "sigma_cr_mpa": sig_cr,
        "slenderness": length_mm / math.sqrt(I / area),
        "euler_valid": sig_cr < sig_y,
        "end_condition": end_condition,
    }

    note = f"Pcr={Pcr:.1f} N，屈曲应力 {sig_cr:.1f} MPa"
    if not info["euler_valid"]:
        note += f" ⚠️ 已超屈服({sig_y:.0f} MPa)，欧拉公式不适用，需按强度校核"
    note += f"；约束={end_condition}(K={EULER_K[end_condition]})"

    chk = Check(
        name=f"buckling[{name}]",
        ok=axial_load_n * sf <= Pcr,
        value=axial_load_n * sf,
        limit=Pcr,
        unit="N",
        note=note,
    )
    return chk, info


# ==============================================================================
# 三、汇总：把稳健性 + 屈曲接进报告
# ==============================================================================
@dataclass
class RobustnessReport:
    verdicts: list[RobustnessVerdict] = field(default_factory=list)
    buckling: list[Check] = field(default_factory=list)
    buckling_info: list[dict] = field(default_factory=list)

    @property
    def sensitive_items(self) -> list[RobustnessVerdict]:
        return [v for v in self.verdicts if v.needs_measurement]

    @property
    def needs_any_measurement(self) -> bool:
        return bool(self.sensitive_items) or any(not c.ok for c in self.buckling)

    def checks(self) -> list[Check]:
        return [v.as_check() for v in self.verdicts] + list(self.buckling)

    def summary(self) -> str:
        lines = ["稳健性评估:"]
        for v in sorted(self.verdicts, key=lambda x: x.margin_factor):
            lines.append("  " + str(v))
        for c in self.buckling:
            lines.append("  " + str(c))
        if self.needs_any_measurement:
            lines.append("  → ⚠️ 存在敏感项：建议对上述零件打材料试件实测，或加大截面")
        else:
            lines.append("  → ✅ 全部落在稳健区，材料参数的不确定性不影响结论")
        return "\n".join(lines)
