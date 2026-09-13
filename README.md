# cadloop

**The missing FEA link in the [text-to-cad](https://github.com/earthtojake/text-to-cad) ecosystem — structural simulation for 3D-printed parts.**

[text-to-cad](https://github.com/earthtojake/text-to-cad) (15k★, MIT) covers the whole
manufacturing chain — CAD authoring, `step-parts`, `dfam-check`, `gcode`, Bambu Lab
printing — but across its 11 agent skills there is **no finite element analysis at all**.
The question *"is this part strong enough, will it buckle, will the layer lines split?"*
is simply not answered.

`cadloop` answers it. It consumes STEP/STL + a `dfam-check` report and returns
strength / stiffness / stability verdicts — with **FDM anisotropy and print
orientation** treated as first-class design variables, in a **fully open,
pure-Python stack** (Gmsh + FEniCSx).

> 源码注释与文档为中文。

---

## 为什么是这个形态

调研 `text-to-cad` 后的定位收缩：**它已经做得比我们好的，我们不做**。

| 环节 | 用谁 |
| :--- | :--- |
| 建模（参数化 CAD / STEP） | `text-to-cad` 的 `cad` 技能（cadgen） |
| 标准件（螺丝/轴承/电机） | `text-to-cad` 的 `step-parts` |
| 可打印性（壁厚/悬垂/支撑） | `text-to-cad` 的 `dfam-check` |
| 切片（真实切片器 CLI） | `text-to-cad` 的 `gcode` |
| 打印机直连 | `text-to-cad` 的 `bambu-labs` |
| **结构强度 / 刚度 / 稳定性** | 🎯 **本仓库** |

本仓库因此砍掉了三个自研模块（可打印性检查、标定试件库、标准件库），
转做生态里唯一缺的那一环 —— 详见 [`archive/README.md`](archive/README.md)。

---

## 它做什么

```
STEP / build123d 参数化模型
        │
        ├──► gmsh 网格（physical group 强制、局部细化）
        │
        ├──► FEniCSx 线弹性求解
        │
        ├──► ★ FDM 各向异性校核：层内 vs 层间分开判据
        │      同一份应力场，按**打印姿态**选对应正应力分量即可，
        │      无需对每种摆放重算（layer_normal = x/y/z）
        │
        ├──► ★ 结论稳健性：材料参数估错会不会翻转结论？（分区 robust→critical）
        │
        ├──► ★ 欧拉屈曲：细长杆常早于强度失效
        │
        ├──► dfam-check 报告合流 + 体积交叉校验
        │
        └──► 带可信度分级的 Markdown 报告 + 版本账本
```

**三条设计纪律**（都是踩坑换来的，写在 `cadloop/__init__.py`）：

1. **先问，再测** —— 能零成本问到的信息（切片器孔补偿值）不要穷举测绘；
   材料参数只在稳健性判据触发时才需要实测。
2. **判据看敏感度，不看精度** —— 块状结构裕度 5~30 倍，参数估错无所谓；
   薄壁/细杆裕度 <2.5 倍才必须实测。`robustness.py` 把这件事变成机械判据。
3. **屈曲常早于强度** —— 100 mm 长的 5×5 立柱受压 31 N 即屈曲，而按强度能承数百牛。
   只做强度校核会给出错误的"安全"结论。

---

## 它不做什么

诚实边界（这些留给生态里的其它工具）：

- ❌ 不建模 —— 用 `text-to-cad` 的 `cad` 技能
- ❌ 不做可打印性检查 —— 用 `dfam-check`（它用射线投射**实测**壁厚，比参数级检查强一个量级）
- ❌ 不做切片 —— 用 `gcode` 技能，走真实切片器 CLI
- ❌ 不做接触/非线性/动力学 —— 只做线弹性静力 + 屈曲 + 结论稳健性
- ❌ 不做各向异性**精确**解 —— 见下方「已知限制」

---

## 快速开始

```bash
conda create -n cadloop python=3.11 -y
conda activate cadloop
pip install build123d gmsh fenics-dolfinx mpi4py pyyaml numpy scipy
# 若已有可用的 FEniCSx 环境，直接 clone 更快：
#   conda create -n cadloop --clone <你的 FEM 环境>
```

跑内置算例（**金标准**：柔性铰链，有解析解可对照）：

```bash
python -m cadloop.examples.flexure_hinge
```

跑**复杂度压测件**：

```bash
python -m cadloop.examples.planetary_gearbox   # 1201 面内齿圈，真渐开线齿形
python -m cadloop.examples.airliner            # 22 件、翼展 500.00 mm
python -m cadloop.examples.biplane             # 双翼机（半硬壳机身）
```

接入 `dfam-check`（可选但推荐）：

```bash
python <skills>/dfam-check/scripts/dfam_tool.py measure part.stl --angle-limit 45 > dfam.json
python -m cadloop.examples.flexure_hinge --dfam dfam.json --layer-normal y
```

---

## 架构

| 模块 | 职责 |
| :--- | :--- |
| `config.py` | 边界条件（打印机/耗材/公差/物理）—— 唯一真相源，全部外置在 `config/printenv.yaml` |
| `geom.py` | 几何度量与导出（`measure` / `export` STEP+STL） |
| `dfam.py` | 消费 `dfam-check` 的 JSON → 判据 + 体积交叉校验 |
| `mesh.py` | gmsh 网格 + 边界标记 + 局部细化（**physical group 强制施加**） |
| `fem.py` | FEniCSx 线弹性 + 各向异性层间校核 + 打印姿态映射 |
| `robustness.py` | 结论稳健性分区 + 欧拉屈曲 |
| `bench.py` | **音叉自检**（解析解基准，不过则拒绝输出零件结论） |
| `report.py` | 报告生成（带可信度分级） |
| `pipeline.py` | 编排：bench → 几何 → dfam → 网格 → 求解 → 稳健性 → 报告 |

**边界条件外置**是这个项目最重要的设计：打印机、耗材、工艺限值、安全系数
全在 `config/printenv.yaml`，换机器/换材料只改配置，不动代码。

---

## 已验证的能力（带实测数据）

| 能力 | 证据 |
| :--- | :--- |
| **音叉自检** | 悬臂梁挠度实测 vs Timoshenko 解析：**误差 −3.48%**（容差 ±8%） |
| **几何交叉校验** | `trimesh` 读 STL vs `build123d` 算实体：体积偏差 **0.00%** |
| **真渐开线齿形** | 分度圆齿厚 = πm/2，**精确到 1e-9**；外齿/内齿轮廓零自交 |
| **真 NACA 翼型** | NACA 0012 截面积 = 0.08221 c²，与理论值偏差 **0.000%** |
| **放样体积** | 锥形翼放样 vs 解析积分：偏差 **0.353%** |
| **支反力自检** | 残差法（非应力面积分）：偏差 **5.5e-11%**（机器精度） |
| **复杂几何规模** | 内齿圈 1201 面 / 3543 棱，构建 2–3.5 s；10 万四面体网格 49.6 s |
| **FDM 层间姿态映射** | 同一 FEA 场评估 3 种打印姿态：层间应力 27.8 → 13.4 MPa（改摆放即可） |

---

## 已知限制（诚实清单）

| # | 限制 | 现状 |
| :-: | :--- | :--- |
| 1 | **各向异性是 screening 级**：只做"各向同性场求解 + 层间拉伸判据"，不是完整横观各向同性本构 | 有明确标注，不冒充精确解 |
| 2 | **边界条件只支持平面选择** —— 无法表达圆柱面（轴承孔、轴颈）与逐区切向载荷 | **最大缺口**，面向真实零件时需手工近似 |
| 3 | **应力奇异不会自动识别**：尖角内凹处 FEA 给出网格相关的非物理值 | `robustness` 会标 `critical`，但不会说"这是奇异值" |
| 4 | **网格收敛未自动执行**：粗网格可能低估应力（实测位移只有解析值的 0.374×）→ **假通过风险** | 需手工做 2 点对比 |
| 5 | 不含接触 / 非线性 / 动力学 / 热 | 设计范围之外 |
| 6 | 材料参数默认走**保守兜底值**（注塑标准值 ×0.8，层间再 ×0.5，SF=2.0） | 报告措辞会明确标注"非实测" |

欢迎 PR —— 尤其欢迎攻 #1~#4。

---

## 示例

| 算例 | 是什么 | 演示什么 |
| :--- | :--- | :--- |
| `flexure_hinge.py` | 柔性铰链 | **金标准**：有解析解、可做网格收敛 |
| `planetary_gearbox.py` | 模数 2 单级行星齿轮箱 | 真渐开线齿形、多体、内齿悬垂 |
| `biplane.py` | 0.5 m 双翼机 | 半硬壳机身、流线支柱、扭转螺旋桨 |
| `airliner.py` | 0.5 m 双发单翼客机 | NACA 2412 后掠翼、翼吊发动机、22 件分件 |

**所有模型都是参数化的**：改一个数字，整机重新生成。模型文件（STEP/STL）不随
仓库分发 —— 跑一次代码就有了，这才是参数化建模的意义。

```bash
python -m cadloop.examples.airliner            # 生成 22 件 + 总装
```

---

## 许可

[MIT](LICENSE)

## 致谢

- [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad) —— 本项目的定位由它定义
- [build123d](https://github.com/gumyr/build123d) / [gmsh](https://gmsh.info/) / [FEniCSx](https://fenicsproject.org/)
