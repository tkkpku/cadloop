"""cadloop.pipeline — 编排层：把各阶段串成一条可诊断的流水线。

设计原则（借鉴 Forge ``forge/pipeline.py`` 与 cad-cae-copilot 的 failure-recovery）
----------------------------------------------------------------------------------
1. **每阶段优雅降级**：某阶段失败不销毁前面已产出的东西。几何永远可用，
   即使求解器没配好——这让流程可以分步推进而不是"全有或全无"。
2. **失败分级**：
   * ``success`` — 全阶段通过
   * ``partial`` — 部分阶段失败，但已有可用产物（**诊断包本身即交付物**）
   * ``failed``  — 关键阶段失败，无可用产物
3. **禁止静默降级**：绝不把"估算值"当成"求解结果"上报；失败必须显式记录在
   ``errors`` 与 ``stages`` 中。
4. **版本账本**：每次运行向 ``ledger.csv`` 追加一行，长期用于把仿真预测与
   真实实验（打印/破坏测试）对齐。
"""

from __future__ import annotations

import csv
import datetime as _dt
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Check, PrintEnv
from .report import (
    CREDIBILITY_ANALYTIC,
    CREDIBILITY_ESTIMATE,
    CREDIBILITY_GEOMETRY,
    CREDIBILITY_SCREENING,
    CREDIBILITY_SOLVER,
    Report,
    value,
)

LEDGER_COLUMNS = [
    "timestamp", "env_name", "env_version", "calibrated", "part_name",
    "volume_mm3", "mass_g", "bbox_mm", "est_print_hours",
    "status", "n_checks", "n_failed", "failed_names",
    "max_disp_mm", "max_vm_mpa", "max_szz_tension_mpa",
    "orientation", "notes",
]


@dataclass
class PipelineResult:
    """一次流水线运行的结果。"""

    part_name: str
    status: str = "failed"
    stages: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metrics: Any = None          # geom.PartMetrics
    mesh_result: Any = None      # mesh.MeshResult
    static_result: Any = None    # fem.StaticResult
    dfam: Any = None             # dfam.DfamReport（来自 text-to-cad 的 dfam-check）
    checks: list[Check] = field(default_factory=list)
    robustness: Any = None       # robustness.RobustnessReport
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    report_path: str = ""
    report: Any = None      # report.Report 对象；调用方可追加判据后重渲染

    def stage_summary(self) -> str:
        icons = {"ok": "✅", "skipped": "⏭️", "failed": "❌", "pending": "…"}
        return "  ".join(f"{k}={icons.get(v, v)}" for k, v in self.stages.items())


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def run_pipeline(
    build_part: Callable[[], Any],
    env: PrintEnv,
    out_root: str | Path,
    part_name: str = "part",
    boundaries: Sequence[Any] = (),
    refine_in: Sequence[Any] = (),
    mesh_size_mm: float = 1.0,
    fixed_boundary: str = "fixed",
    load_boundary: str = "load",
    load_vector_n: Sequence[float] | None = None,
    exclude_planes: Sequence[tuple[str, float, float]] = (),
    layer_normal: str = "z",
    buckling_members: Sequence[dict] | None = None,
    dfam_json: str | Path | None = None,
    run_fea: bool = True,
    require_bench: bool | None = None,
    bench_mesh_size_mm: float = 1.0,
    bench_workdir: str | Path | None = None,
    note: str = "",
    verbose: bool = False,
) -> PipelineResult:
    """执行完整流水线。

    Parameters
    ----------
    build_part : 无参可调用对象，返回 build123d 零件（几何构建由调用方提供）
    boundaries : ``mesh.BoundarySpec`` 序列
    dfam_json  : **text-to-cad ``dfam-check`` 产出的 JSON 报告路径**。
        可打印性**不再由本仓库自行计算**（2026-09-13 收缩决策）：
        dfam-check 用射线投射实测壁厚 + 逐体拆分，比原先的参数级检查
        强一个量级。此处只把它翻译成判据并合流进报告。
        生成方式::

            python <skills>/dfam-check/scripts/dfam_tool.py measure part.stl \
                --angle-limit 45 > dfam.json

    run_fea    : 是否做有限元（False 时只做几何 + 已提供的 dfam 判据）
    require_bench : 是否强制音叉通过才继续。缺省取 ``env.verification.require_bench_pass``
    """
    from . import geom as G
    from .dfam import checks_from_report, cross_check_volume, load_report, summarize

    require_bench = env.verification.require_bench_pass if require_bench is None else require_bench

    res = PipelineResult(part_name=part_name)
    out_root = Path(out_root)
    run_dir = out_root / f"{_stamp()}_{part_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rep = Report(
        title=f"cadloop 运行报告 — {part_name}",
        env=env,
        part_name=part_name,
        status="running",
    )
    if note:
        rep.notes.append(note)

    # ==========================================================================
    # 阶段 0：音叉自检（管线健康）
    # --------------------------------------------------------------------------
    # ⚠️ 音叉**必须用自己的固定网格尺寸**，不能继承零件的 mesh_size_mm。
    #
    # 2026-09-13 踩坑（复杂度压测暴露）：原先这里传的是零件的 ``mesh_size_mm``，
    # 于是当零件用粗网格（如 3.0 mm，大件才可行）时，**音叉自身也被粗化**——
    # 实测悬臂梁网格退化到 808 节点/2776 单元，挠度误差超出 ±8% 容差，
    # 判为"管线故障"并**中止整个分析**。
    #
    # 这是把「基准尺」和「被测物」用了同一把刻度的错误：音叉的意义在于它
    # 是**与零件无关的独立基准**，其分辨率应当固定在已标定通过的值上。
    # ==========================================================================
    if require_bench:
        try:
            from .bench import run_all as run_bench

            bw = Path(bench_workdir) if bench_workdir else run_dir / "_bench"
            br = run_bench(env, bw, mesh_size_mm=bench_mesh_size_mm, verbose=verbose)
            res.stages["bench"] = "ok" if br.all_passed else "failed"
            rep.sections.append(("音叉自检（管线健康判据）", "```\n" + br.summary() + "\n```"))
            res.artifacts["bench_workdir"] = str(bw)
            if not br.all_passed:
                res.errors.append(
                    "音叉自检未通过 → 按 require_bench_pass 策略中止。"
                    "请先修管线（或确认网格分辨率），不要信任后续任何零件结论。"
                )
                res.status = "failed"
                rep.status = "failed"
                rep.checks = []
                rep.write(run_dir / "report.md")
                res.report_path = str(run_dir / "report.md")
                return res
        except Exception as e:
            res.stages["bench"] = "failed"
            res.errors.append(f"音叉执行异常: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
            if require_bench:
                res.status = "failed"
                rep.status = "failed"
                rep.write(run_dir / "report.md")
                res.report_path = str(run_dir / "report.md")
                return res
    else:
        res.stages["bench"] = "skipped"

    # ==========================================================================
    # 阶段 1：几何
    # ==========================================================================
    part = None
    try:
        part = build_part()
        metrics = G.measure(part, env, name=part_name)
        res.metrics = metrics
        res.stages["geometry"] = "ok"
        rep.values.append(value("体积", metrics.volume_mm3, "mm³", CREDIBILITY_GEOMETRY))
        rep.values.append(value("质量（估算）", metrics.mass_g, "g", CREDIBILITY_ESTIMATE,
                                "PLA 密度 1.24 g/cm³"))
        rep.values.append(value("包围盒最大值", max(metrics.bbox_mm), "mm", CREDIBILITY_GEOMETRY,
                                f"全尺寸 {tuple(round(b, 2) for b in metrics.bbox_mm)}"))
        hours = metrics.estimated_print_hours(env)
        rep.values.append(value("打印时长（估算）", hours, "h", CREDIBILITY_ESTIMATE,
                                "系数待本机标定"))
        res.checks.append(metrics.fits_in_printer(env))
        res.checks.append(
            Check(
                name="needs_split",
                ok=not metrics.needs_split(env),
                value=max(metrics.bbox_mm),
                limit=env.rules.split_threshold_mm,
                unit="mm",
                note="是否需要分件（超此值需拆除）",
            )
        )
        ph = env.printer.check_print_hours(hours)
        if ph is not None:
            res.checks.append(ph)
    except Exception as e:
        res.stages["geometry"] = "failed"
        res.errors.append(f"几何阶段失败: {type(e).__name__}: {e}")
        res.status = "failed"
        rep.status = "failed"
        rep.write(run_dir / "report.md")
        res.report_path = str(run_dir / "report.md")
        return res

    # ---- 导出 ---------------------------------------------------------------
    try:
        art = G.export(part, run_dir, stem=part_name)
        res.artifacts.update(art)
    except Exception as e:
        res.stages["export"] = "failed"
        res.errors.append(f"导出失败: {type(e).__name__}: {e}")
        res.stages.setdefault("export", "failed")

    # ==========================================================================
    # 阶段 2：可制造性（消费 text-to-cad 的 dfam-check 报告）
    # --------------------------------------------------------------------------
    # 收缩决策（2026-09-13）：本仓库不再自行实现可打印性检查。
    # dfam-check 用射线投射实测壁厚、报 p05 分位数、支持逐体拆分与六向摆放，
    # 信息量远高于原先的参数级检查；重复实现只会维护两份会漂移的判据。
    # 此处职责变成「翻译 + 合流」。
    # ==========================================================================
    if dfam_json:
        try:
            drep = load_report(dfam_json)
            res.dfam = drep
            checks, notes = checks_from_report(drep, env)
            res.checks.extend(checks)
            res.checks.append(cross_check_volume(drep, metrics.volume_mm3))
            res.stages["dfam"] = "ok"
            body = [summarize(drep)] + [f"- {n}" for n in notes]
            rep.sections.append(("可制造性（dfam-check 实测）", "\n".join(body)))
            rep.values.append(
                value("壁厚 p05（dfam 实测）", drep.wall_p05_mm, "mm", CREDIBILITY_GEOMETRY,
                      f"min={drep.wall_min_mm}, median={drep.wall_median_mm}, 样本 {drep.wall_samples}")
            )
            rep.values.append(
                value("支撑体积占比", drep.support_ratio_pct, "%", CREDIBILITY_ESTIMATE,
                      "棱柱上界估算，成本信号而非硬失败")
            )
        except Exception as e:
            res.stages["dfam"] = "failed"
            res.errors.append(f"dfam 报告处理失败: {type(e).__name__}: {e}")
    else:
        res.stages["dfam"] = "skipped"
        res.notes.append(
            "未提供 dfam 报告 → 本次**未做**可制造性判定。"
            "请用 dfam-check 技能生成: "
            "python <skills>/dfam-check/scripts/dfam_tool.py measure part.stl --angle-limit 45"
        )

    # ==========================================================================
    # 阶段 3：网格 + 求解
    # ==========================================================================
    if run_fea and load_vector_n is not None and boundaries:
        try:
            from .fem import solve_static
            from .mesh import build_mesh, read_into_dolfinx

            step = res.artifacts.get("step")
            if not step:
                raise RuntimeError("缺少 STEP 产物，无法划分网格")
            mres = build_mesh(step, run_dir / f"{part_name}.msh", mesh_size_mm=mesh_size_mm,
                              boundaries=boundaries, refine_in=refine_in, verbose=verbose)
            res.mesh_result = mres
            res.stages["mesh"] = "ok"
            rep.values.append(value("网格单元数", mres.n_tets, "-", CREDIBILITY_GEOMETRY,
                                    f"h={mres.mesh_size_mm} mm"))
            rep.values.append(value("网格体积", mres.volume_mm3, "mm³", CREDIBILITY_GEOMETRY,
                                    "与几何体积对照可查网格化误差"))
            res.artifacts["msh"] = mres.msh_path

            mesh, _ct, ft = read_into_dolfinx(mres.msh_path)
            sres = solve_static(
                mesh, ft, env,
                fixed_boundary=fixed_boundary,
                load_boundary=load_boundary,
                load_vector_n=load_vector_n,
                boundary_tags=mres.boundary_tags,
                exclude_planes=exclude_planes,
            )
            res.static_result = sres
            res.stages["fea"] = "ok"
            rep.values.append(value("最大位移", sres.max_disp_mm, "mm", CREDIBILITY_SOLVER,
                                    f"位置 {tuple(round(c, 2) for c in sres.max_disp_point)}"))
            rep.values.append(value("最大 von Mises（全场）", sres.max_von_mises_mpa, "MPa",
                                    CREDIBILITY_SOLVER))
            rep.values.append(value("最大 von Mises（排除边界层）", sres.max_von_mises_bulk_mpa,
                                    "MPa", CREDIBILITY_SOLVER))
            rep.values.append(value("最大层间拉应力",
                                    sres.sigma_tension_mpa.get(layer_normal, float("nan")),
                                    "MPa", CREDIBILITY_SCREENING,
                                    f"层间方向 → 零件 {layer_normal.upper()} 轴；"
                                    f"各轴最大拉应力 "
                                    + ", ".join(f"{k.upper()}={v:.2f}" for k, v in sres.sigma_tension_mpa.items())))
            if sres.exclusion_report:
                rep.sections.append(
                    ("边界层排除说明",
                     f"{sres.exclusion_report}\n\n"
                     "全约束固定端会阻止泊松横向收缩，在固定面附近产生数值边界层，"
                     "该处应力被显著高估（约束方式引入，网格加密也不会消除）。"
                     "故强度判据取**排除边界层之后**的最大值。")
                )
            res.checks.extend(sres.checks(env, layer_normal=layer_normal))
        except Exception as e:
            res.stages["mesh" if res.mesh_result is None else "fea"] = "failed"
            res.errors.append(f"FEA 阶段失败: {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
    else:
        res.stages["mesh"] = "skipped"
        res.stages["fea"] = "skipped"

    # ==========================================================================
    # 阶段 4：稳健性评估（材料参数不确定性会不会翻转结论？）
    # ==========================================================================
    try:
        from .robustness import RobustnessReport, assess, check_buckling

        rb = RobustnessReport()
        s = res.static_result
        if s is not None:
            m, sf = env.material, env.physics.safety_factor
            # 层内（von Mises，排除边界层）
            vm = s.max_von_mises_bulk_mpa if s.max_von_mises_bulk_mpa == s.max_von_mises_bulk_mpa else s.max_von_mises_mpa
            rb.verdicts.append(assess("von_mises_in_layer", vm, m.allowable_stress("in_layer", sf), "MPa"))
            # 层间（沿打印层间方向的拉应力）
            s_il = s.sigma_tension_mpa.get(layer_normal, float("nan"))
            if s_il == s_il:
                rb.verdicts.append(
                    assess(f"interlayer_tension_{layer_normal}", s_il,
                           m.allowable_stress("interlayer", sf), "MPa")
                )

        # 屈曲检查（若调用方提供了受压杆件清单）
        for spec in (buckling_members or ()):
            chk, info = check_buckling(env, **spec)
            rb.buckling.append(chk)
            rb.buckling_info.append(info)

        res.robustness = rb
        res.checks.extend(rb.checks())
        rep.sections.append(("稳健性评估", "```\n" + rb.summary() + "\n```"))
        if rb.needs_any_measurement:
            rep.notes.append(
                "⚠️ 存在敏感项（S < 2.5 或屈曲不足）：建议实测该方向材料强度，或加大截面。"
                "其余零件落在稳健区，保守兜底参数的不确定性不影响结论。"
            )
        res.stages["robustness"] = "ok"
    except Exception as e:
        res.stages["robustness"] = "failed"
        res.errors.append(f"稳健性评估失败: {type(e).__name__}: {e}")

    # ==========================================================================
    # 收尾：状态判定 + 报告
    # ==========================================================================
    failed_stages = [k for k, v in res.stages.items() if v == "failed"]
    if not failed_stages:
        res.status = "success"
    elif res.artifacts.get("step"):
        res.status = "partial"
    else:
        res.status = "failed"

    rep.status = res.status
    rep.checks = res.checks
    rep.artifacts = dict(res.artifacts)
    rep.values.append(
        value(
            "判据通过率",
            100.0 * sum(1 for c in res.checks if c.ok) / max(len(res.checks), 1),
            "%",
            CREDIBILITY_ANALYTIC,
            f"{sum(1 for c in res.checks if c.ok)}/{len(res.checks)} 项通过",
        )
    )
    rep.sections.append(("阶段状态", "```\n" + res.stage_summary() + "\n```"))
    if res.errors:
        rep.sections.append(("错误记录", "\n".join(f"- `{e.splitlines()[0]}`" for e in res.errors)))

    rp = rep.write(run_dir / "report.md")
    res.report_path = str(rp)
    res.artifacts["report"] = str(rp)
    res.report = rep          # 暴露给调用方：可在追加本件特有判据后重渲染

    append_ledger(out_root / "ledger.csv", res, env, note=note)
    return res


# ==============================================================================
# 版本账本
# ==============================================================================
def append_ledger(path: str | Path, res: PipelineResult, env: PrintEnv, note: str = "") -> Path:
    """向账本追加一行。

    账本的作用：把**仿真预测**与**真实实验结果**（打印、破坏测试）长期对齐。
    只有攒够这样的数据，材料模型的标定（``calibrated: true``）才有依据。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    s = res.static_result
    m = res.metrics
    row = {
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "env_name": env.meta.get("name", ""),
        "env_version": env.meta.get("version", ""),
        "calibrated": env.calibrated,
        "part_name": res.part_name,
        "volume_mm3": round(m.volume_mm3, 2) if m else "",
        "mass_g": round(m.mass_g, 3) if m else "",
        "bbox_mm": "x".join(str(round(b, 1)) for b in m.bbox_mm) if m else "",
        "est_print_hours": round(m.estimated_print_hours(env), 3) if m else "",
        "status": res.status,
        "n_checks": len(res.checks),
        "n_failed": sum(1 for c in res.checks if not c.ok),
        "failed_names": ";".join(c.name for c in res.checks if not c.ok),
        "max_disp_mm": round(s.max_disp_mm, 6) if s else "",
        "max_vm_mpa": round(s.max_von_mises_mpa, 5) if s else "",
        "max_szz_tension_mpa": round(s.max_sigma_zz_tension_mpa, 5) if s else "",
        "orientation": getattr(res.static_result, "layer_normal", "") if res.static_result else "",
        "notes": note,
    }
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return p
