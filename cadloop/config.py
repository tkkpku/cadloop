"""cadloop.config — 边界条件的读取与施加。

设计原则
--------
1. **单一真相源**：所有打印机 / 耗材 / 公差数字来自 ``config/printenv.yaml``，
   代码里不硬编码任何物理常数。
2. **配置是"活的"**：本模块不只存数据，还提供判据方法
   （``fits_within`` / ``needs_split`` / ``allowable_stress`` / ``check_wall`` ...），
   让"建模阶段主动施加约束"成为可能，而不是画完图再事后检查。
3. **未标定保护**：``meta.calibrated = false`` 时，强度与装配类的**结论性判定**
   被降级为参考值并附警告——避免把典型值当实测值用（本机材料参数尚未标定）。

术语约定
--------
- 长度单位统一 **mm**，力 **N**，应力 **MPa (= N/mm²)**，质量 **g**。
- 打印方向语义：
  * ``"in_layer"``  —— 受力/变形发生在**层平面内**（打印 XY 面）
  * ``"interlayer"`` —— 受力沿**层间法向**（打印 Z 向），FDM 件最弱方向
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

# 默认配置文件位置：<repo>/config/printenv.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "printenv.yaml"

# 层平面内 / 层间 的别名集合，容忍多种写法
_IN_LAYER_ALIASES = {"in_layer", "in-layer", "xy", "flat", "in_plane", "layer_plane"}
_INTERLAYER_ALIASES = {"interlayer", "inter-layer", "z", "upright", "layer_normal", "normal"}


class ConfigError(RuntimeError):
    """配置文件缺失、字段缺失或取值非法。"""


# ------------------------------------------------------------------------------
# 判据结果：统一格式，便于汇总进报告
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    """单条判据的检查结果。"""

    name: str
    ok: bool
    value: float
    limit: float
    unit: str = ""
    comparison: str = "<="  # 判据方向："<=" 表示 value 不得超过 limit
    note: str = ""

    @property
    def margin(self) -> float:
        """相对裕度。

        对 ``<=`` 判据：``(limit - value) / limit``，正值表示通过。
        对 ``>=`` 判据：``(value - limit) / limit``。
        """
        if self.limit == 0:
            return float("inf") if self.ok else float("-inf")
        if self.comparison == "<=":
            return (self.limit - self.value) / abs(self.limit)
        return (self.value - self.limit) / abs(self.limit)

    def __str__(self) -> str:
        flag = "PASS" if self.ok else "FAIL"
        return (
            f"[{flag}] {self.name}: {self.value:.4g} {self.unit} "
            f"{self.comparison} {self.limit:.4g} {self.unit}"
            + (f"  ({self.note})" if self.note else "")
        )


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d or d[key] is None:
        raise ConfigError(f"配置缺少必需字段 {where}.{key}")
    return d[key]


# ------------------------------------------------------------------------------
# 打印机
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Printer:
    model: str
    build_volume_mm: tuple[float, float, float]
    nozzle_diameter_mm: float
    available_nozzles_mm: tuple[float, ...]
    layer_height_mm: dict[str, float]
    max_volumetric_flow_mm3_s: float
    plate: str
    max_chamber_temp_c: float
    xy_hole_compensation_mm: float | None
    max_single_print_hours: float | None
    remote_camera: bool

    # ---- 幅面判据 -----------------------------------------------------------
    def fits_within(self, size_mm: Sequence[float]) -> bool:
        """零件包围盒是否落在可用幅面内。"""
        return all(s <= v for s, v in zip(size_mm, self.build_volume_mm))

    def needs_split(self, size_mm: Sequence[float], threshold_mm: float) -> bool:
        """是否必须分件。取「幅面」与「配置阈值」中更严的一个。"""
        limit = min(min(self.build_volume_mm), threshold_mm)
        return any(s > limit for s in size_mm)

    def check_build_volume(self, size_mm: Sequence[float]) -> Check:
        worst = max(size_mm)
        vol = max(self.build_volume_mm)
        return Check(
            name="build_volume",
            ok=worst <= vol,
            value=worst,
            limit=vol,
            unit="mm",
            note=f"包围盒 {tuple(round(s, 2) for s in size_mm)} vs 幅面 {self.build_volume_mm}",
        )

    # ---- 公差补偿 -----------------------------------------------------------
    @property
    def hole_compensation_mm(self) -> float:
        """Bambu Studio「XY 孔补偿」。未标定时返回 0 并在报告中提示。"""
        return 0.0 if self.xy_hole_compensation_mm is None else float(self.xy_hole_compensation_mm)

    @property
    def hole_compensation_is_calibrated(self) -> bool:
        return self.xy_hole_compensation_mm is not None

    # ---- 打印时长 -----------------------------------------------------------
    def check_print_hours(self, hours: float) -> Check | None:
        """单次打印时长判据。未配置上限时返回 None（信息不足，不做判定）。"""
        if self.max_single_print_hours is None:
            return None
        return Check(
            name="print_duration",
            ok=hours <= self.max_single_print_hours,
            value=hours,
            limit=self.max_single_print_hours,
            unit="h",
            note="超过单次上限需拆分盘次",
        )


# ------------------------------------------------------------------------------
# 材料：各向异性是 FDM 件的第一性事实
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Material:
    name: str
    density_g_cm3: float
    E_xy_mpa: float
    E_z_mpa: float
    poisson_xy: float
    yield_xy_mpa: float
    yield_z_mpa: float
    ultimate_xy_mpa: float
    ultimate_z_mpa: float
    tg_c: float
    thermal_conductivity_w_mk: float
    creep_model: str

    @staticmethod
    def _mode(orientation: str) -> str:
        o = orientation.strip().lower()
        if o in _IN_LAYER_ALIASES:
            return "in_layer"
        if o in _INTERLAYER_ALIASES:
            return "interlayer"
        raise ValueError(
            f"未知受力方向 {orientation!r}；"
            f"应为层平面内 {sorted(_IN_LAYER_ALIASES)} 或层间 {sorted(_INTERLAYER_ALIASES)}"
        )

    def E(self, orientation: str = "in_layer") -> float:
        """杨氏模量 [MPa]。"""
        return self.E_xy_mpa if self._mode(orientation) == "in_layer" else self.E_z_mpa

    def yield_strength(self, orientation: str = "in_layer") -> float:
        """屈服强度 [MPa]。"""
        return self.yield_xy_mpa if self._mode(orientation) == "in_layer" else self.yield_z_mpa

    def ultimate_strength(self, orientation: str = "in_layer") -> float:
        return self.ultimate_xy_mpa if self._mode(orientation) == "in_layer" else self.ultimate_z_mpa

    def allowable_stress(self, orientation: str = "in_layer", safety_factor: float = 1.5) -> float:
        """许用应力 = 屈服强度 / 安全系数 [MPa]。"""
        return self.yield_strength(orientation) / safety_factor

    def density_g_mm3(self) -> float:
        return self.density_g_cm3 * 1e-3

    def orientation_factor(self) -> float:
        """层间强度相对层平面内强度的折减系数（各向异性比）。"""
        return self.E_z_mpa / self.E_xy_mpa

    def check_stress(
        self, sigma_mpa: float, orientation: str = "in_layer", safety_factor: float = 1.5
    ) -> Check:
        allow = self.allowable_stress(orientation, safety_factor)
        return Check(
            name=f"stress_{orientation}",
            ok=sigma_mpa <= allow,
            value=sigma_mpa,
            limit=allow,
            unit="MPa",
            note=f"屈服 {self.yield_strength(orientation):.1f} MPa / SF {safety_factor}",
        )


# ------------------------------------------------------------------------------
# 可打印性红线
# ------------------------------------------------------------------------------
# 取值对齐 earthtojake/text-to-cad 的 `dfam-check/references/process-limits.md`
# （FDM 列，引自 Hubs/Formlabs/EOS + ISO/ASTM 52910 §6.5/§6.7）。
# 该表把「有支撑壁厚」与「无支撑壁厚」分开——无支撑壁更严（1.6 vs 1.2 mm），
# 因为垂直悬空壁没有下方材料支撑，散热与附着都更差。
@dataclass(frozen=True)
class DesignRules:
    min_wall_mm: float
    min_feature_mm: float
    min_hole_dia_mm: float
    min_pin_dia_mm: float
    max_overhang_deg: float
    max_bridge_mm: float
    split_threshold_mm: float
    min_clearance_mm: float
    min_wall_unsupported_mm: float = 0.0   # 0 = 未配置则不检查

    def check_wall(self, t_mm: float, where: str = "", unsupported: bool = False) -> Check:
        """壁厚判据。

        ``unsupported=True`` 时用更严的「无支撑壁厚」限值；未配置则回落到
        有支撑限值（并在 note 里说明）。
        """
        if unsupported and self.min_wall_unsupported_mm > 0:
            limit, name = self.min_wall_unsupported_mm, "min_wall_unsupported"
        else:
            limit, name = self.min_wall_mm, "min_wall"
        return Check(name, t_mm >= limit, t_mm, limit, "mm", ">=", where)

    def check_feature(self, d_mm: float, where: str = "") -> Check:
        return Check("min_feature", d_mm >= self.min_feature_mm, d_mm, self.min_feature_mm, "mm", ">=", where)

    def check_hole(self, d_mm: float, where: str = "") -> Check:
        return Check("min_hole", d_mm >= self.min_hole_dia_mm, d_mm, self.min_hole_dia_mm, "mm", ">=", where)

    def check_pin(self, d_mm: float, where: str = "") -> Check:
        return Check("min_pin", d_mm >= self.min_pin_dia_mm, d_mm, self.min_pin_dia_mm, "mm", ">=", where)

    def check_overhang(self, deg: float, where: str = "") -> Check:
        """悬垂判据。``deg`` 为**面与水平面的夹角**（自水平面量起）：
        小于 ``max_overhang_deg`` 时需要支撑。该约定与 dfam-check、
        PrusaSlicer/Bambu Studio 的 overhang threshold 一致。
        """
        return Check("max_overhang", deg <= self.max_overhang_deg, deg, self.max_overhang_deg, "deg", "<=", where)

    def check_bridge(self, span_mm: float, where: str = "") -> Check:
        return Check("max_bridge", span_mm <= self.max_bridge_mm, span_mm, self.max_bridge_mm, "mm", "<=", where)


# ------------------------------------------------------------------------------
# 装配公差
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Fits:
    calibrated: bool
    rotating_clearance_mm: float
    sliding_clearance_mm: float
    press_fit_mm: float
    snap_fit_clearance_mm: float

    def hole_for(self, nominal_d_mm: float, kind: str = "rotating") -> float:
        """按配合类型计算孔径。

        kind: ``rotating`` / ``sliding`` / ``press`` / ``snap``
        间隙为正（孔大于轴），过盈为负（孔小于轴）。
        """
        table = {
            "rotating": 2 * self.rotating_clearance_mm,
            "sliding": 2 * self.sliding_clearance_mm,
            "press": self.press_fit_mm,
            "snap": 2 * self.snap_fit_clearance_mm,
        }
        if kind not in table:
            raise ValueError(f"未知配合类型 {kind!r}；可选 {sorted(table)}")
        return nominal_d_mm + table[kind]


# ------------------------------------------------------------------------------
# 物理条件
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Physics:
    gravity_mm_s2: float
    safety_factor: float
    ambient_temp_c: float
    friction_pla_pla: float
    motors: tuple[dict, ...]

    @property
    def motor_count(self) -> int:
        return sum(int(m.get("count", 1)) for m in self.motors)

    def total_stall_torque_nmm(self) -> float:
        return sum(float(m.get("stall_torque_nmm", 0.0)) * int(m.get("count", 1)) for m in self.motors)


@dataclass(frozen=True)
class Verification:
    require_bench_pass: bool
    static_tol: float
    modal_tol: float
    buckling_tol: float
    thermal_tol: float
    mesh_convergence_tol: float


# ------------------------------------------------------------------------------
# 顶层对象
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class PrintEnv:
    """完整边界条件集。所有建模/仿真代码的入口。"""

    meta: dict
    printer: Printer
    material: Material
    rules: DesignRules
    fits: Fits
    hardware: dict
    physics: Physics
    verification: Verification

    # ---- 加载 ---------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "PrintEnv":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            raise ConfigError(f"配置文件不存在: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PrintEnv":
        if not isinstance(raw, dict):
            raise ConfigError("配置根节点必须是映射")

        pr = _require(raw, "printer", "root")
        vol_block = _require(pr, "build_volume_mm", "printer")
        active_key = pr.get("active_build_volume", "dual_nozzle_intersect")
        if active_key not in vol_block:
            raise ConfigError(f"printer.active_build_volume={active_key!r} 不在 build_volume_mm 中")

        printer = Printer(
            model=str(_require(pr, "model", "printer")),
            build_volume_mm=tuple(float(v) for v in vol_block[active_key]),  # type: ignore[arg-type]
            nozzle_diameter_mm=float(_require(pr, "nozzle_diameter_mm", "printer")),
            available_nozzles_mm=tuple(float(v) for v in pr.get("available_nozzles_mm", [])),
            layer_height_mm={k: float(v) for k, v in (pr.get("layer_height_mm") or {}).items()},
            max_volumetric_flow_mm3_s=float(pr.get("max_volumetric_flow_mm3_s", 40.0)),
            plate=str(pr.get("plate", "unknown")),
            max_chamber_temp_c=float(pr.get("max_chamber_temp_c", 0.0)),
            xy_hole_compensation_mm=(
                None if pr.get("xy_hole_compensation_mm") is None
                else float(pr["xy_hole_compensation_mm"])
            ),
            max_single_print_hours=(
                None if pr.get("max_single_print_hours") is None
                else float(pr["max_single_print_hours"])
            ),
            remote_camera=bool(pr.get("remote_camera", False)),
        )

        mt = _require(raw, "material", "root")
        material = Material(
            name=str(_require(mt, "name", "material")),
            density_g_cm3=float(_require(mt, "density_g_cm3", "material")),
            E_xy_mpa=float(_require(mt, "E_xy_mpa", "material")),
            E_z_mpa=float(_require(mt, "E_z_mpa", "material")),
            poisson_xy=float(mt.get("poisson_xy", 0.35)),
            yield_xy_mpa=float(_require(mt, "yield_xy_mpa", "material")),
            yield_z_mpa=float(_require(mt, "yield_z_mpa", "material")),
            ultimate_xy_mpa=float(mt.get("ultimate_xy_mpa", mt["yield_xy_mpa"])),
            ultimate_z_mpa=float(mt.get("ultimate_z_mpa", mt["yield_z_mpa"])),
            tg_c=float(mt.get("tg_c", 0.0)),
            thermal_conductivity_w_mk=float(mt.get("thermal_conductivity_w_mk", 0.0)),
            creep_model=str(mt.get("creep_model", "ignored")),
        )

        dr = _require(raw, "design_rules", "root")
        _rule_keys = (
            "min_wall_mm", "min_feature_mm", "min_hole_dia_mm", "min_pin_dia_mm",
            "max_overhang_deg", "max_bridge_mm", "split_threshold_mm", "min_clearance_mm",
        )
        _rule_kwargs = {k: float(_require(dr, k, "design_rules")) for k in _rule_keys}
        # 可选：无支撑壁厚（dfam-check 把有/无支撑分开；缺省沿用有支撑值）
        if dr.get("min_wall_unsupported_mm") is not None:
            _rule_kwargs["min_wall_unsupported_mm"] = float(dr["min_wall_unsupported_mm"])
        rules = DesignRules(**_rule_kwargs)

        ft = _require(raw, "fits", "root")
        fits = Fits(
            calibrated=bool(ft.get("calibrated", False)),
            rotating_clearance_mm=float(ft.get("rotating_clearance_mm", 0.0)),
            sliding_clearance_mm=float(ft.get("sliding_clearance_mm", 0.0)),
            press_fit_mm=float(ft.get("press_fit_mm", 0.0)),
            snap_fit_clearance_mm=float(ft.get("snap_fit_clearance_mm", 0.0)),
        )

        ph = raw.get("physics") or {}
        physics = Physics(
            gravity_mm_s2=float(ph.get("gravity_mm_s2", 9806.65)),
            safety_factor=float(ph.get("safety_factor", 1.5)),
            ambient_temp_c=float(ph.get("ambient_temp_c", 25.0)),
            friction_pla_pla=float(ph.get("friction_pla_pla", 0.4)),
            motors=tuple(ph.get("motors") or ()),
        )

        vf = raw.get("verification") or {}
        verification = Verification(
            require_bench_pass=bool(vf.get("require_bench_pass", True)),
            static_tol=float(vf.get("static_tol", 0.08)),
            modal_tol=float(vf.get("modal_tol", 0.08)),
            buckling_tol=float(vf.get("buckling_tol", 0.10)),
            thermal_tol=float(vf.get("thermal_tol", 0.05)),
            mesh_convergence_tol=float(vf.get("mesh_convergence_tol", 0.02)),
        )

        return cls(
            meta=raw.get("meta") or {},
            printer=printer,
            material=material,
            rules=rules,
            fits=fits,
            hardware=raw.get("hardware") or {},
            physics=physics,
            verification=verification,
        )

    # ---- 便捷属性 -----------------------------------------------------------
    @property
    def calibration_strategy(self) -> str:
        """标定策略：``conservative_fallback``（保守兜底）或 ``measured``（已实测）。"""
        cal = self.meta.get("calibration") or {}
        return str(cal.get("strategy", "conservative_fallback"))

    @property
    def calibration_confidence(self) -> str:
        cal = self.meta.get("calibration") or {}
        return str(cal.get("confidence", "low"))

    @property
    def calibrated(self) -> bool:
        """材料参数是否为**实测**值（等价于 strategy == measured）。"""
        if "calibrated" in self.meta:          # 向后兼容旧格式
            return bool(self.meta["calibrated"])
        return self.calibration_strategy == "measured"

    @property
    def allow_conclusive_claims(self) -> bool:
        """是否允许输出「强度合格 / 可装配」这类结论性判定。

        两种情形均可出结论，但**依据不同、措辞必须不同**：

        * ``measured``（材料+公差都实测）→ 结论可作为设计依据
        * ``conservative_fallback``（保守兜底）→ 结论**也成立**，因为材料参数被
          故意低估（注塑值×0.8，层间再×0.5）且 SF 提到 2.0，双保险；
          但报告须标注「基于保守估计」。

        真正禁止出结论的是「既没实测、又没采用保守策略」的中间态。
        """
        if self.calibration_strategy == "conservative_fallback":
            return True          # 保守值 → 结论偏安全，可用（带标注）
        return self.calibrated and self.fits.calibrated

    def blocking_gaps(self) -> list[str]:
        """尚未标定/未确认的配置项，供报告提示。"""
        gaps: list[str] = []
        cal = self.meta.get("calibration") or {}
        if self.calibration_strategy == "conservative_fallback":
            gaps.append(
                f"材料参数为**保守兜底值**（{cal.get('basis', '注塑值×0.8，层间×0.5')}）"
                "→ 结论偏安全；安全裕度落在敏感区(1.5≤S<2.5)的零件会被要求实测复核"
            )
        elif not self.calibrated:
            gaps.append("材料力学参数未标定且未采用保守策略 → 强度判定不可用")
        if not self.fits.calibrated:
            gaps.append("配合公差未标定（待确认切片器孔补偿值）→ 装配判定仅供参照")
        if self.printer.xy_hole_compensation_mm is None:
            gaps.append("printer.xy_hole_compensation_mm 未填（待确认识别切片器设定）")
        if self.printer.max_single_print_hours is None:
            gaps.append("printer.max_single_print_hours 未填（待确认能否过夜打印）→ 无法做时长判据")
        return gaps

    # ---- 汇总 ---------------------------------------------------------------
    def summary(self) -> str:
        p, m = self.printer, self.material
        lines = [
            f"PrintEnv: {self.meta.get('name', '?')} v{self.meta.get('version', '?')}",
            f"  标定策略          : {self.calibration_strategy} (confidence={self.calibration_confidence})",
            f"  printer           : {p.model} 幅面 {p.build_volume_mm} mm, 喷嘴 {p.nozzle_diameter_mm} mm",
            f"  material          : {m.name}  E_xy={m.E_xy_mpa:.0f} / E_z={m.E_z_mpa:.0f} MPa"
            f"  (各向异性比 {m.orientation_factor():.2f})",
            f"  yield             : in_layer {m.yield_xy_mpa:.0f} / interlayer {m.yield_z_mpa:.0f} MPa",
            f"  许用应力          : in_layer {m.allowable_stress('in_layer', self.physics.safety_factor):.2f}"
            f" / interlayer {m.allowable_stress('interlayer', self.physics.safety_factor):.2f} MPa"
            f"  @SF={self.physics.safety_factor}",
            f"  motors            : {self.physics.motor_count} x "
            f"({', '.join(str(x.get('name', '?')) for x in self.physics.motors) or '-'})",
            f"  budget            : {self.hardware.get('budget_cny', '-')} CNY",
            f"  min wall / feature: {self.rules.min_wall_mm} (unsupported {self.rules.min_wall_unsupported_mm})"
            f" / {self.rules.min_feature_mm} mm",
            f"  overhang / hole   : self-supporting {self.rules.max_overhang_deg}° / min hole Ø{self.rules.min_hole_dia_mm} mm",
        ]
        gaps = self.blocking_gaps()
        if gaps:
            lines.append("  未标定项:")
            lines.extend(f"    - {g}" for g in gaps)
        return "\n".join(lines)


# 模块级便捷入口
def load_env(path: str | Path | None = None) -> PrintEnv:
    """加载边界条件（最常用入口）。"""
    return PrintEnv.load(path)


if __name__ == "__main__":  # 自检：python -m cadloop.config
    env = load_env()
    print(env.summary())
    print()
    print("判据自检:")
    print(" ", env.rules.check_wall(0.8))
    print(" ", env.rules.check_wall(0.6))
    print(" ", env.material.check_stress(23.44, "in_layer", env.physics.safety_factor))
    print(" ", env.material.check_stress(23.44, "interlayer", env.physics.safety_factor))
    print(" ", env.printer.check_build_volume((60, 40, 8)))
    print(" ", env.printer.check_build_volume((400, 40, 8)))
    print("  MR63ZZ 过盈座孔径:", round(env.fits.hole_for(6.0, "press"), 3),
          "| 转动配合座孔径:", round(env.fits.hole_for(6.0, "rotating"), 3))
