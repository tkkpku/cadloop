"""cadloop.bench — 音叉自检（管线健康判据）。

为什么需要它
------------
一个能算出漂亮云图的管线，和一个算错的管线，从图上是**看不出来**的。
所以在信任任何真实零件的结论之前，先用有**闭式解析解**的算例把管线跑一遍：

* 不匹配 → 先修管线（或确认网格分辨率），再谈别的
* 匹配 → 该管线在本问题类型上的可信度才成立

这与 Forge 项目 ``projects/benchmark_cantilever/study.py`` 的做法一致
（其注释明言："If any of these MISMATCH, fix the install before trusting any real part"）。

⚠️ 关于三维固支梁与欧拉-伯努利解的系统偏差
------------------------------------------
本机实测（悬臂梁，L/H=10，P1 单元）：

===========  ==========  ================
网格 n       单元数       与解析解之比
===========  ==========  ================
4            960         0.428
8            7 680       0.743
16           61 440      0.919
24           207 360     0.962
===========  ==========  ================

比值**二阶收敛**趋近 1（收敛阶实测 1.156 → 1.658 → 1.867），
Richardson 外推（p=2）得 −3.9855 vs 解析 −4.0000，**残差仅 0.36%**。

结论：偏差是**网格离散误差**，不是物理效应。因此：
* 音叉必须用**足够细的网格**（或做网格收敛检查），否则会误判为"管线有 bug"
* 容差取 ``verification.static_tol``（默认 8%），并额外做剪切修正
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import PrintEnv
from .mesh import BoundarySpec, MeshResult, build_mesh, read_into_dolfinx

# 矩形截面梁的剪切修正系数 k = 5/6（Timoshenko 梁标准值）
SHEAR_CORRECTION_K = 5.0 / 6.0


@dataclass
class BenchCase:
    """单个基准算例的结果。"""

    name: str
    expected: float
    measured: float
    tolerance: float
    unit: str
    formula: str = ""
    note: str = ""

    @property
    def rel_error(self) -> float:
        if self.expected == 0:
            return float("inf")
        return (self.measured - self.expected) / self.expected

    @property
    def passed(self) -> bool:
        return abs(self.rel_error) <= self.tolerance

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return (
            f"[{flag}] {self.name}: 实测 {self.measured:.6f} {self.unit} "
            f"vs 解析 {self.expected:.6f} {self.unit}  "
            f"(误差 {self.rel_error * 100:+.2f}%, 容差 ±{self.tolerance * 100:.0f}%)"
        )


@dataclass
class BenchReport:
    """音叉总报告。"""

    cases: list[BenchCase] = field(default_factory=list)
    mesh_info: dict = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return bool(self.cases) and all(c.passed for c in self.cases)

    def summary(self) -> str:
        head = "音叉自检: " + ("PASS ✅" if self.all_passed else "FAIL ❌")
        body = "\n".join("  " + str(c) for c in self.cases)
        return head + "\n" + body if body else head


# ==============================================================================
# 解析解
# ==============================================================================
def cantilever_tip_deflection_eb(L: float, W: float, H: float, F: float, E: float) -> float:
    """欧拉-伯努利悬臂梁端部挠度 [mm]。"""
    I = W * H**3 / 12.0
    return F * L**3 / (3.0 * E * I)


def cantilever_tip_deflection_timoshenko(L: float, W: float, H: float, F: float, E: float, nu: float) -> float:
    """Timoshenko 悬臂梁端部挠度（含剪切变形）[mm]。

    .. math::
        \\delta = \\frac{FL^3}{3EI} + \\frac{FL}{kGA},
        \\quad G = \\frac{E}{2(1+\\nu)},\\quad k = 5/6
    """
    I = W * H**3 / 12.0
    A = W * H
    G = E / (2.0 * (1.0 + nu))
    return F * L**3 / (3.0 * E * I) + F * L / (SHEAR_CORRECTION_K * G * A)


def cantilever_first_bending_hz(L: float, W: float, H: float, E: float, density_g_mm3: float) -> float:
    """悬臂梁一阶弯曲固有频率 [Hz]（固支-自由）。

    .. math::
        f_1 = \\frac{1.875^2}{2\\pi}\\sqrt{\\frac{EI}{\\rho A L^4}}
    """
    I = W * H**3 / 12.0
    A = W * H
    return (1.875**2) / (2.0 * math.pi) * math.sqrt(E * I / (density_g_mm3 * A * L**4))


# ==============================================================================
# 静力音叉
# ==============================================================================
def bench_cantilever_static(
    env: PrintEnv,
    workdir: str | Path,
    mesh_size_mm: float = 1.0,
    L: float = 100.0,
    W: float = 10.0,
    H: float = 10.0,
    F: float = 10.0,
    verbose: bool = False,
) -> tuple[BenchCase, MeshResult]:
    """悬臂梁静力音叉：建几何 → 网格 → 求解 → 与解析解对比。

    几何直接用 gmsh 的 OCC 建（不经过 build123d），这样音叉只检验
    ``mesh`` 与 ``fem`` 两层——符合"分层验证"的原则。
    """
    import gmsh

    from .fem import solve_static
    from .mesh import MeshError

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    step_like = workdir / "bench_cantilever.step"
    msh_path = workdir / "bench_cantilever.msh"

    # ---- 1. 几何（gmsh OCC）------------------------------------------------
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("bench_cantilever")
        gmsh.model.occ.addBox(0, 0, 0, L, W, H)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_like))
    finally:
        gmsh.finalize()

    # ---- 2. 网格 -----------------------------------------------------------
    mres = build_mesh(
        step_like,
        msh_path,
        mesh_size_mm=mesh_size_mm,
        boundaries=(
            BoundarySpec("fixed", "x", 0.0, tol=1e-6),
            BoundarySpec("load", "x", L, tol=1e-6),
        ),
        verbose=verbose,
    )

    # ---- 3. 求解 -----------------------------------------------------------
    mesh, _cell_tags, facet_tags = read_into_dolfinx(msh_path)
    res = solve_static(
        mesh,
        facet_tags,
        env,
        fixed_boundary="fixed",
        load_boundary="load",
        load_vector_n=(0.0, 0.0, -F),
        boundary_tags=mres.boundary_tags,
    )

    # ---- 4. 对比解析解 -----------------------------------------------------
    expected = cantilever_tip_deflection_timoshenko(L, W, H, F, env.material.E_xy_mpa, env.material.poisson_xy)
    eb = cantilever_tip_deflection_eb(L, W, H, F, env.material.E_xy_mpa)

    case = BenchCase(
        name="cantilever_static_tip_deflection",
        expected=expected,
        measured=abs(res.max_disp_mm),
        tolerance=env.verification.static_tol,
        unit="mm",
        formula=r"delta = FL^3/(3EI) + FL/(kGA)",
        note=(
            f"EB={eb:.4f} mm, Timoshenko={expected:.4f} mm; "
            f"dofs={res.n_dofs}, tets={res.n_cells}, h={mesh_size_mm} mm; "
            f"力平衡偏差见 force_balance 判据"
        ),
    )
    return case, mres


# ==============================================================================
# 网格收敛检查
# ==============================================================================
def mesh_convergence_check(
    env: PrintEnv,
    workdir: str | Path,
    sizes_mm: tuple[float, ...] = (2.0, 1.0, 0.5),
    **kwargs,
) -> list[BenchCase]:
    """对同一算例做网格加密，检查结果是否收敛（相邻两档变化 < 容差）。

    这是比"单点对比解析解"更严格的健康判据：如果结果随网格加密乱跳，
    说明管线（或问题设置）有问题。
    """
    out: list[BenchCase] = []
    prev: float | None = None
    for h in sizes_mm:
        case, _ = bench_cantilever_static(env, Path(workdir) / f"h{h}", mesh_size_mm=h, **kwargs)
        measured = case.measured
        if prev is None:
            out.append(
                BenchCase(
                    name=f"convergence_h{h}",
                    expected=measured,  # 首档无参照
                    measured=measured,
                    tolerance=1.0,
                    unit="mm",
                    note="基准档（无前值可比）",
                )
            )
        else:
            out.append(
                BenchCase(
                    name=f"convergence_h{h}",
                    expected=prev,
                    measured=measured,
                    tolerance=env.verification.mesh_convergence_tol,
                    unit="mm",
                    note=f"相对上一档变化（容差 {env.verification.mesh_convergence_tol * 100:.0f}%）",
                )
            )
        prev = measured
    return out


# ==============================================================================
# 全量音叉
# ==============================================================================
def run_all(env: PrintEnv, workdir: str | Path, mesh_size_mm: float = 1.0, verbose: bool = False) -> BenchReport:
    """运行全部音叉算例。

    当前实现：悬臂梁静力（mesh + fem 两层）。
    计划补充：模态（需 SLEPc）、屈曲、热传导——每加一项就多一层保险。
    """
    rep = BenchReport()
    case, mres = bench_cantilever_static(env, workdir, mesh_size_mm=mesh_size_mm, verbose=verbose)
    rep.cases.append(case)
    rep.mesh_info = {
        "nodes": mres.n_nodes,
        "tets": mres.n_tets,
        "volume_mm3": mres.volume_mm3,
        "boundaries": mres.boundary_tags,
        "h": mres.mesh_size_mm,
    }
    return rep


if __name__ == "__main__":  # python -m cadloop.bench
    import sys
    from .config import load_env

    env = load_env()
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(env.summary())
    print()
    report = run_all(env, Path("./runs/_bench"), mesh_size_mm=h)
    print(report.summary())
    print()
    print("网格:", report.mesh_info)
    sys.exit(0 if report.all_passed else 1)
