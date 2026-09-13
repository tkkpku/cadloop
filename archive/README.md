# archive/ — 已停用模块（2026-09-13）

本目录存放**被 text-to-cad 生态取代**的模块。它们**不在 `cadloop` 包内**，
因此不会被误 import；保留仅为历史参考与回滚。

| 文件 | 原职责 | 被谁取代 | 停用理由 |
| :--- | :--- | :--- | :--- |
| `rules_print.py` | 可打印性检查（参数级壁厚、悬垂、床接触） | **`dfam-check`** 技能 | 它用**射线投射实测**壁厚并报 min/p05/p25/median 分位数，还给六向摆放候选与逐体拆分；我们那份是"调用方自报数字"的参数级检查，信息量差一个量级 |
| `coupons.py` | 标定试件库（ASTM D638 Type V 等） | **改为按需**调用 | 标定策略改为「先问再测 + 保守兜底 + 敏感度触发」：孔补偿可**问**、材料参数用保守值 + 稳健性判据自动决定是否需实测。试件几何仍可按需生成 |
| `geom_standard_parts.py` | `geom.py` 的原始完整副本（标准件库 + 摆放分析） | **`step-parts`** + **`dfam-check`** | 标准件库被 `step-parts`（可从 step.parts 下载**厂商 STEP 模型**）取代；摆放分析被 dfam-check 取代 |

## 回滚方式

```powershell
Copy-Item archive\rules_print.py   cadloop\cadloop\
Copy-Item archive\coupons.py       cadloop\cadloop\
Copy-Item archive\geom_standard_parts.py cadloop\cadloop\geom.py
```

⚠️ 回滚 `geom.py` 会一并退回标准件库——注意 `dfam-check` / `step-parts`
路线下的新代码不再依赖它们。

## 归档时的状态

三份文件在归档时都是**可导入、语法正确**的（`rules_print` / `coupons`
带 `__deprecated__` 标记；标记位于 `from __future__` **之后**——放前面会
导致整个模块 `SyntaxError`，这个坑本轮踩过两次）。
