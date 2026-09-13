"""T3 golden example — 柔性铰链（叶片式）夹持单元。

用途：验证 cadloop 通道端到端可用，并演示「解析解对照 + 各向异性 + 打印方向」。

几何（单位 mm，Z 向上）
----------------------

::

        z
        ^   ┌──────────────────┐
        │   │      arm         │  ← 顶面加载 F（沿 +X）
        │   └────────┬─────────┘
        │            │
        │   ┌────────┴─────────┐
        │   │     hinge        │  ← 薄片：厚 t（X 向）, 宽 D（Y 向）, 长 L（Z 向）
        │   └────────┬─────────┘
        │   ┌────────┴─────────┐
        │   │      base        │  ← 底面固定
        └──►└──────────────────┘  x

铰链厚度 t 沿 X 方向，因此**弯曲绕 Y 轴**，惯性矩

.. math::
    I_y = \\frac{D\\,t^3}{12}, \\qquad K_\\theta = \\frac{E I_y}{L}

铰链根部最大弯曲应力（矩形截面）

.. math::
    \\sigma_{\\max} = \\frac{6M}{D\\,t^2}, \\qquad M = F \\cdot \\ell

其中力臂 ℓ = 加载面到铰链中心的距离。

取默认参数（t=0.8, D=12, L=10, F=1.5 N）：
ℓ = 8 + 5 = 13 mm，M = 19.5 N·mm，σ ≈ 15.2 MPa，θ ≈ 7.3°，
**落在小变形线弹性适用范围内**（<15°），因此解析对照成立。

用法
----
::

    conda activate cadloop
    cd <repo>
    python -m cadloop.examples.flexure_hinge            # 默认参数
    python -m cadloop.examples.flexure_hinge --t 1.0 --force 2.0
    python -m cadloop.examples.flexure_hinge --no-fea   # 只做几何+可打印性
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from ..config import Check, PrintEnv, load_env
from ..mesh import BoundarySpec
from ..report import CREDIBILITY_ANALYTIC, CREDIBILITY_SCREENING, value
from ..pipeline import run_pipeline


# ==============================================================================
# 参数
# ==============================================================================
@dataclass(frozen=True)
class FlexureHingeParams:
    """柔性铰链单元的设计变量。

    ⚠️ 2026-09-13 按 ISO/ASTM 工艺限值重新设计（见 ``printenv.yaml`` 的
    ``design_rules``）：

    * 原设计 t=0.8 mm **违反**无支撑壁厚下限 1.6 mm（铰链是独立竖壁，
      不是贴着实体的"有支撑壁"），且悬垂限值由 60° 收紧到 45°。
    * 重设计依据两条闭式关系（小变形线弹性）：

      .. math::

          \\sigma_{\\text{nom}} = \\frac{\\theta E t}{2L}
          \\qquad\\text{（给定转角下，应力只由 } t/L \\text{ 决定，与宽度无关）}
          \\\\
          K = \\frac{E D t^3}{12 L}
          \\qquad\\text{（转动刚度）}

      故加厚 t 必须同步加长 L（保持 t/L 不变或更低）才能不增加应力；
      刚度变化再用宽度 D 补偿。
    """

    hinge_thickness: float = 1.6    # t：铰链厚度（X 向）—— ≥ min_wall_unsupported 1.6
    hinge_length: float = 20.0      # L：铰链长度（Z 向）—— 由 10 加倍，抵消 t 加倍带来的应力上升
    depth: float = 8.0              # D：Y 向宽度（铰链宽度）—— 由 12 收到 8，平衡刚度
    block_width: float = 24.0       # 固定块与臂的 X 向宽度
    block_height: float = 8.0       # 固定块高度
    arm_height: float = 8.0         # 臂高度
    load_force_n: float = 1.3       # 端部载荷（沿 +X）—— 对应约 4° 转角
    fillet_radius: float = 2.0      # 铰链根部圆角（R/t=1.25，用于压低应力集中）

    # ---- 派生几何量 ---------------------------------------------------------
    @property
    def z_hinge_center(self) -> float:
        return self.block_height + self.hinge_length / 2.0

    @property
    def z_load_face(self) -> float:
        return self.block_height + self.hinge_length + self.arm_height

    @property
    def lever_arm_mm(self) -> float:
        """加载面相对铰链中心的力臂 [mm]。"""
        return self.z_load_face - self.z_hinge_center

    def bbox_estimate_mm(self) -> tuple[float, float, float]:
        return (
            self.block_width,
            self.depth,
            self.block_height + self.hinge_length + self.arm_height,
        )


# ==============================================================================
# 解析预测
# ==============================================================================
def analytic_predictions(p: FlexureHingeParams, env: PrintEnv) -> dict[str, float]:
    """叶片式柔性铰链的闭式预测（小变形线弹性）。

    注意：这是**只考虑铰链变形**的模型，忽略了固定块与臂自身的弹性柔度。
    因此 FEA 的总位移会略大于本预测（多出的部分来自臂/块的弯曲与剪切）。
    应力预测受此影响较小（铰链是唯一大应力区）。
    """
    E = env.material.E_xy_mpa
    D, t, L = p.depth, p.hinge_thickness, p.hinge_length
    lever = p.lever_arm_mm

    I = D * t**3 / 12.0
    K_theta = E * I / L
    M = p.load_force_n * lever
    sigma = 6.0 * M / (D * t**2)
    theta_rad = M / K_theta
    disp_hinge_only = theta_rad * lever

    return {
        "I_y_mm4": I,
        "K_theta_nmm_per_rad": K_theta,
        "moment_nmm": M,
        "lever_mm": lever,
        "sigma_max_mpa": sigma,
        "theta_deg": math.degrees(theta_rad),
        "disp_hinge_only_mm": disp_hinge_only,
    }


# ==============================================================================
# 几何
# ==============================================================================
def build_part(p: FlexureHingeParams):
    """构建几何（build123d 代数模式）。"""
    from build123d import Box, Pos

    W, D = p.block_width, p.depth
    hb, L, ha = p.block_height, p.hinge_length, p.arm_height

    base = Pos(0, 0, hb / 2.0) * Box(W, D, hb)
    hinge = Pos(0, 0, hb + L / 2.0) * Box(p.hinge_thickness, D, L)
    arm = Pos(0, 0, hb + L + ha / 2.0) * Box(W, D, ha)

    part = base + hinge + arm

    # 圆角：**铰链根部/顶部的应力集中棱边**。
    # ⚠️ 2026-09-13 修正：原先筛的是 |Y|=D/2 的棱边（铰链"前后面"与块的交接），
    #    但本件绕 Y 轴弯曲，危险纤维在 x=±t/2 的**侧面**，应力集中也在那里。
    #    筛错边 = 给不承力的地方倒角，Kt 基本没降。现改为筛 |X|≈t/2 的棱边
    #    （沿 Y 方向、每层 2 条、上下共 4 条）。
    if p.fillet_radius > 0:
        from build123d import Axis

        half_t = p.hinge_thickness / 2.0
        edges = [
            e for e in part.edges()
            if (abs(e.center().Z - hb) < 1e-6 or abs(e.center().Z - (hb + L)) < 1e-6)
            and abs(abs(e.center().X) - half_t) < 1e-6
        ]
        if edges:
            try:
                part = part.fillet(p.fillet_radius, edges)
            except Exception:
                pass

    return part


def boundaries(p: FlexureHingeParams) -> tuple[BoundarySpec, ...]:
    return (
        BoundarySpec("fixed", "z", 0.0, tol=1e-6),
        BoundarySpec("load", "z", p.z_load_face, tol=1e-6),
    )


def hinge_refinement(p: FlexureHingeParams, layers: float = 4.0, transition_mm: float = 1.5):
    """铰链区域的局部网格细化。

    ⚠️ 这是本样例的**关键设置**，不是可选优化：
    P1 四面体在弯曲问题中会人工刚化，若铰链厚度方向只落一层单元，位移会被
    低估数倍（本机实测 h=0.7 vs t=0.8 时低估 3.5 倍）。而全局细化到 h=0.2
    会让单元数涨到 10⁶ 量级，因此必须**只在铰链附近细化**。

    取 ``layers`` 层跨厚度：size = t / layers（弯曲问题建议 ≥3–4 层）。
    """
    from ..mesh import RefinementBox

    half_t = p.hinge_thickness / 2.0
    return RefinementBox(
        name="hinge",
        bounds_mm=(
            -half_t - 0.1, -p.depth / 2.0 - 0.1, p.block_height - 0.5,
            half_t + 0.1, p.depth / 2.0 + 0.1, p.block_height + p.hinge_length + 0.5,
        ),
        size_mm=p.hinge_thickness / layers,
        thickness_mm=transition_mm,
    )


def wall_params(p: FlexureHingeParams) -> dict[str, float]:
    return {
        "hinge_thickness": p.hinge_thickness,
        "block_width": p.block_width,
        "block_height": p.block_height,
    }


def feature_params(p: FlexureHingeParams) -> dict[str, float]:
    return {"pin_dummy": 3.0}  # 占位：本件无标准件特征


# ==============================================================================
# 业务判据（在通道通用判据之外，本件特有的设计判据）
# ==============================================================================
def design_checks(p: FlexureHingeParams, env: PrintEnv, analytic: dict[str, float],
                  measured_disp_mm: float | None = None) -> list[Check]:
    """柔性铰链特有的设计判据。"""
    out: list[Check] = []
    allow = env.material.allowable_stress("in_layer", env.physics.safety_factor)

    # 0. 🆕 无支撑壁厚：铰链是**独立竖壁**，不是贴着实体的"有支撑壁"，
    #    故适用更严的 min_wall_unsupported_mm（1.6 mm）而非 1.2 mm。
    #    这是 2026-09-13 重设计的**首要约束**（原 t=0.8 违反此限）。
    out.append(
        env.rules.check_wall(
            p.hinge_thickness,
            where=f"hinge t={p.hinge_thickness}",
            unsupported=True,
        )
    )
    # 0b. 圆角相对厚度比：R/t 越大应力集中越低（本件目标 R/t ≥ 1.0）
    if p.fillet_radius > 0:
        rt = p.fillet_radius / p.hinge_thickness
        out.append(
            Check(
                name="fillet_ratio",
                ok=rt >= 1.0,
                value=rt,
                limit=1.0,
                unit="R/t",
                comparison=">=",
                note=f"R={p.fillet_radius} / t={p.hinge_thickness}；R/t<1 时 Kt 显著偏高",
            )
        )

    # 1. 铰链根部应力（解析值，screening 级）
    out.append(
        Check(
            name="hinge_bending_stress_analytic",
            ok=analytic["sigma_max_mpa"] <= allow,
            value=analytic["sigma_max_mpa"],
            limit=allow,
            unit="MPa",
            note="解析 σ=θEt/(2L)（等价 6M/(D t²)），筛选级、不含应力集中",
        )
    )

    # 2. 小变形假设有效性：转角须小于 15°，否则线弹性/梁理论失效
    theta = analytic["theta_deg"]
    out.append(
        Check(
            name="small_deformation_validity",
            ok=theta <= 15.0,
            value=theta,
            limit=15.0,
            unit="deg",
            note="超过 15° 需几何非线性（大变形）分析，梁理论失效",
        )
    )

    # 3. 刚度下限：太软则夹持无力（示例阈值：K ≥ 50 N·mm/rad）
    K = analytic["K_theta_nmm_per_rad"]
    out.append(
        Check(
            name="rotational_stiffness",
            ok=K >= 50.0,
            value=K,
            limit=50.0,
            unit="N·mm/rad",
            comparison=">=",
            note="夹持机构需要足够刚度传递力",
        )
    )
    return out


# ==============================================================================
# 主流程
# ==============================================================================
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="cadloop T3 样例：柔性铰链单元")
    ap.add_argument("--t", type=float, default=1.6, help="铰链厚度 [mm]（≥1.6 满足无支撑壁厚下限）")
    ap.add_argument("--L", type=float, default=20.0, help="铰链长度 [mm]")
    ap.add_argument("--D", type=float, default=8.0, help="铰链宽度 [mm]")
    ap.add_argument("--R", type=float, default=2.0, help="铰链根部圆角 [mm]")
    ap.add_argument("--force", type=float, default=1.3, help="端部载荷 [N]")
    ap.add_argument("--dfam", type=str, default=None,
                    help="dfam-check 产出的 JSON 报告路径（可制造性判据来源）")
    ap.add_argument("--mesh-size", type=float, default=1.0,
                    help="全局网格尺寸 [mm]；铰链区自动局部细化（见 --layers）")
    ap.add_argument("--layers", type=float, default=4.0,
                    help="铰链厚度方向的单元层数（弯曲问题需 ≥3–4，否则刚度虚高数倍）")
    ap.add_argument("--no-refine", action="store_true",
                    help="关闭局部细化（不推荐：会让位移被低估数倍）")
    ap.add_argument("--layer-normal", default="z", choices=["x", "y", "z"],
                    help="打印层间方向对应的零件坐标轴。flat→z；绕X侧躺→y；绕Y侧躺→x")
    ap.add_argument("--out", type=str, default="./runs", help="输出根目录")
    ap.add_argument("--config", type=str, default=None, help="printenv.yaml 路径")
    ap.add_argument("--no-fea", action="store_true", help="跳过有限元，只做几何与可打印性")
    ap.add_argument("--no-bench", action="store_true", help="跳过音叉自检（不推荐）")
    args = ap.parse_args(argv)

    env = load_env(args.config)
    p = FlexureHingeParams(
        hinge_thickness=args.t,
        hinge_length=args.L,
        depth=args.D,
        load_force_n=args.force,
        fillet_radius=getattr(args, "R", 2.0),
    )

    print(env.summary())
    print()

    an = analytic_predictions(p, env)
    print("解析预测（小变形线弹性，仅计铰链变形）:")
    print(f"  力臂 ℓ           = {an['lever_mm']:.2f} mm")
    print(f"  惯性矩 I_y       = {an['I_y_mm4']:.5f} mm⁴")
    print(f"  转动刚度 K_θ     = {an['K_theta_nmm_per_rad']:.2f} N·mm/rad")
    print(f"  弯矩 M           = {an['moment_nmm']:.3f} N·mm")
    print(f"  根部应力 σ_max   = {an['sigma_max_mpa']:.3f} MPa")
    print(f"  端点转角 θ       = {an['theta_deg']:.3f} °")
    print(f"  铰链段端点位移   = {an['disp_hinge_only_mm']:.4f} mm")
    print()

    res = run_pipeline(
        build_part=lambda: build_part(p),
        env=env,
        out_root=args.out,
        part_name="flexure_hinge",
        boundaries=boundaries(p),
        refine_in=(() if args.no_refine else (hinge_refinement(p, layers=args.layers),)),
        mesh_size_mm=args.mesh_size,
        fixed_boundary="fixed",
        load_boundary="load",
        load_vector_n=(p.load_force_n, 0.0, 0.0) if not args.no_fea else None,
        exclude_planes=(("z", 0.0, 1.5),) if not args.no_fea else (),
        layer_normal=args.layer_normal,
        dfam_json=args.dfam,
        run_fea=not args.no_fea,
        require_bench=not args.no_bench,
        note=f"T3 golden example, params={asdict(p)}",
        verbose=False,
    )

    # 追加本件特有的设计判据到报告
    extra = design_checks(p, env, an)
    res.checks.extend(extra)
    if res.report is not None:
        res.report.checks = res.checks
        res.report.write(res.report_path)

    print("阶段:", res.stage_summary())
    if res.metrics:
        print("几何:", res.metrics)
    if res.mesh_result:
        print("网格:", res.mesh_result)
    if res.static_result:
        print("力学:", res.static_result.summary())

    print()
    ok = sum(1 for c in res.checks if c.ok)
    print(f"判据: {ok}/{len(res.checks)} 通过")
    for c in res.checks:
        if not c.ok:
            print("  ", c)
    if res.errors:
        print("错误:")
        for e in res.errors:
            print("  -", e.splitlines()[0])
    print()
    print("报告:", res.report_path)
    print("状态:", res.status)
    return 0 if res.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
