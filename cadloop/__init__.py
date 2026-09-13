"""cadloop — text-to-cad 生态的**结构仿真（FEA）补位**。

定位（2026-09-13 收缩后）
-------------------------
`earthtojake/text-to-cad <https://github.com/earthtojake/text-to-cad>`_
（15k⭐ / MIT）已覆盖 CAD → 可制造性 → 切片 → 打印全链，且各项都更成熟：

============================  ==========================================
环节                          用谁
============================  ==========================================
建模（参数化 CAD / STEP）      text-to-cad 的 ``cad`` 技能（cadgen）
标准件（螺丝/轴承/电机）        text-to-cad 的 ``step-parts``
可打印性（壁厚/悬垂/支撑）      text-to-cad 的 ``dfam-check``
切片（真实切片器 CLI）          text-to-cad 的 ``gcode``
打印机直连                     text-to-cad 的 ``bambu-labs``
**结构强度 / 刚度 / 稳定性**    🎯 **本仓库**（它没有 FEA）
============================  ==========================================

它的 11 个技能中**没有任何有限元**（``sdf`` 是 SDFormat 机器人仿真）——
中间缺的正是「这零件够不够强、会不会屈曲、层间会不会裂」这一环。

分层结构
--------
::

    config.py       边界条件（打印机/耗材/公差/标准件/物理）——唯一真相源
    geom.py         几何度量与导出（measure / export STEP+STL）
    dfam.py         🆕 消费 dfam-check 的 JSON → 判据（可制造性合流）
    mesh.py         gmsh 网格 + 边界标记 + 局部细化（physical group 强制施加）
    fem.py          FEniCSx 线弹性 + 各向异性层间校核 + 姿态映射
    robustness.py   🌟 结论稳健性（材料参数不敏感？）+ 屈曲检查
    bench.py        音叉自检（解析解基准，管线健康判据）
    report.py       报告生成（带可信度分级）
    pipeline.py     编排：bench → 几何 → dfam → 网格 → 求解 → 稳健性 → 报告

    （已归档至 ../archive/：rules_print.py 可打印性、coupons.py 标定试件、
      geom_standard_parts.py 标准件库 —— 被 text-to-cad 对应技能取代）

设计哲学（四轮踩坑的沉淀）
--------------------------
1. **先问，再测**：能用零成本方式（问人/查配置）拿到已知信息，就不要穷举测绘。
   标定试件（coupons）仅在稳健性判据触发时才需要打印。
2. **判据看敏感度，不看精度**：材料参数估错会不会翻转结论？块状结构裕度
   5–30 倍，参数无所谓；薄壁/细杆裕度 <2.5 倍，才必须实测。见 ``robustness``。
3. **屈曲常早于强度失效**：100 mm 长的 5×5 立柱受压 31 N 即屈曲，而按强度
   能承数百牛。只做强度校核会给出错误的"安全"结论。
4. **打印姿态决定失效模式**：同一份应力场，按姿态选对应的正应力分量即可，
   **无需对每种摆放重算**（见 ``fem.StaticResult.checks`` 的 ``layer_normal``）。
5. **网格尺寸相对最小特征定**：``size ≈ 最小特征 / 3~4``；薄特征处必须局部
   细化，否则 P1 四面体会人工刚化（实测可致弯曲刚度虚高 3.5 倍）。
6. **不要重复造轮子**：动手前先查生态里有没有更成熟的实现——本仓库砍掉
   三个模块，全是因为别人做得更好。

模块导入策略
------------
``config`` 与 ``robustness`` 只依赖 PyYAML / 标准库，因此本包在**未安装
CAD/FEA 内核的环境里也能导入**，便于做配置与判据的单元测试。重依赖
（build123d / gmsh / dolfinx）在各自模块内延迟导入，缺失时抛出带明确指引的异常。
"""

from __future__ import annotations

from .config import (
    Check,
    ConfigError,
    DEFAULT_CONFIG_PATH,
    DesignRules,
    Fits,
    Material,
    Physics,
    PrintEnv,
    Printer,
    Verification,
    load_env,
)
from .dfam import (
    DfamError,
    DfamReport,
    checks_from_report,
    cross_check_volume,
    load_report,
    summarize as summarize_dfam,
)
from .robustness import (
    RobustnessReport,
    RobustnessVerdict,
    assess,
    check_buckling,
    euler_buckling_load,
)

__version__ = "0.3.0"

__all__ = [
    "Check",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "DesignRules",
    "DfamError",
    "DfamReport",
    "Fits",
    "Material",
    "Physics",
    "PrintEnv",
    "Printer",
    "RobustnessReport",
    "RobustnessVerdict",
    "Verification",
    "assess",
    "check_buckling",
    "checks_from_report",
    "cross_check_volume",
    "euler_buckling_load",
    "load_env",
    "load_report",
    "summarize_dfam",
    "__version__",
]
