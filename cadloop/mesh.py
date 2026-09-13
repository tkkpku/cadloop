"""cadloop.mesh — gmsh 网格生成与边界标记。

⚠️ 本模块固化了两次实测踩坑（不写进去别人一定会再踩一遍）：

坑 1：``gmsh → dolfinx`` 必须打 physical group
    否则 ``dolfinx.io.gmshio.model_to_mesh`` 抛
    ``IndexError: index -1 is out of bounds for axis 0 with size 0``
    —— 报错症状与根因完全不相干，极难定位。
    本模块的 ``build_mesh`` 无条件为体与指定面打 physical group。

坑 2：FEniCSx 0.8 的 gmsh 接口是 ``dolfinx.io.gmshio``
    （``dolfinx.io.gmsh`` 是 0.9+ 的写法）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover
    import gmsh  # type: ignore

    _HAS_GMSH = True
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
    gmsh = None  # type: ignore
    _HAS_GMSH = False
    _IMPORT_ERROR = _e


class MeshError(RuntimeError):
    """网格生成失败。"""


def require_gmsh() -> None:
    if not _HAS_GMSH:
        raise MeshError(
            f"gmsh 不可用（{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}）。"
            "请确认已激活 cadloop 环境：conda activate cadloop"
        )


# 体始终占 tag=1；面 tag 从 10 起，避免与体冲突
VOLUME_TAG = 1
FACE_TAG_BASE = 10


@dataclass
class BoundarySpec:
    """按几何判据描述一个边界（面）。

    工程语言里常说"固定 x=0 端面""在 x=L 端面加载"。这里直接把它编码成判据，
    由 gmsh 去几何里找面，避免依赖易变的 face tag 顺序。

    Parameters
    ----------
    name : 边界名（``"fixed"`` / ``"load"`` / ...），同时作为 physical group 名
    axis : ``"x"`` / ``"y"`` / ``"z"``
    value : 该轴上的坐标值
    tol  : 判据容差 [mm]
    tag  : 指定 physical tag；``None`` 时自动分配
    """

    name: str
    axis: str
    value: float
    tol: float = 1e-3
    tag: int | None = None


@dataclass
class RefinementBox:
    """局部网格细化区域（轴对齐盒子）。

    ⚠️ 为什么必须支持局部细化
    -------------------------
    P1 四面体单元在**弯曲**问题中存在人工刚化（剪切锁死）：若厚度方向只有一层
    单元，弯曲刚度会被严重高估。本机实测（柔性铰链 t=0.8 mm）：

    ===========  ==========  ==============
    网格 h        厚度方向层数  位移 / 解析值
    ===========  ==========  ==============
    0.70 mm       ~1.1        0.28
    0.50 mm       ~1.6        （见收敛测试）
    ===========  ==========  ==============

    而全局细化到 h=0.2 mm 会让单元数暴涨到 ~10⁶（不可行）。因此正确做法是
    **只在薄特征附近细化**，其余区域保持粗网格。

    经验法则：``size_mm ≈ 最小特征尺寸 / 3~4``。
    """

    name: str
    bounds_mm: tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax
    size_mm: float
    thickness_mm: float = 0.0  # 过渡层厚度（0 = 硬边界）


@dataclass
class MeshResult:
    """网格生成结果。"""

    msh_path: str
    n_nodes: int
    n_cells: int
    n_tets: int
    volume_mm3: float
    boundary_tags: dict[str, int] = field(default_factory=dict)
    mesh_size_mm: float = 0.0
    log: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Mesh: {self.n_nodes} nodes / {self.n_tets} tets  "
            f"体积={self.volume_mm3:.2f} mm³  h={self.mesh_size_mm:.2f} mm  "
            f"边界面={self.boundary_tags}"
        )


def _find_faces_on_plane(axis: str, value: float, tol: float) -> list[int]:
    """找出所有落在给定平面上的面 tag。

    判据：面包围盒在该轴方向退化为常数（即平面垂直于该轴）且坐标值匹配。
    """
    require_gmsh()
    idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    found: list[int] = []
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)  # (xmin,ymin,zmin, xmax,ymax,zmax)
        lo, hi = bb[idx], bb[idx + 3]
        if abs(lo - value) <= tol and abs(hi - value) <= tol:
            found.append(tag)
    return found


def build_mesh(
    step_path: str | Path,
    out_msh: str | Path,
    mesh_size_mm: float,
    boundaries: Sequence[BoundarySpec] = (),
    refine_in: Sequence[RefinementBox] = (),
    optimize: bool = True,
    second_order: bool = False,
    verbose: bool = False,
) -> MeshResult:
    """从 STEP 文件生成 3D 体网格，并标记边界。

    Parameters
    ----------
    step_path     : build123d 导出的 STEP 文件
    out_msh       : 输出 ``.msh`` 路径
    mesh_size_mm  : 全局最大单元尺寸
    boundaries    : 需要标记的边界面判据
    optimize      : 是否做网格质量优化（Netgen）
    second_order  : 是否生成二阶单元（P2，精度更高、自由度更多）
    """
    require_gmsh()
    step_path, out_msh = Path(step_path), Path(out_msh)
    if not step_path.exists():
        raise MeshError(f"STEP 文件不存在: {step_path}")
    out_msh.parent.mkdir(parents=True, exist_ok=True)

    log: list[str] = []
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("cadloop_part")

        # --- 1. 导入 OCC 几何 ------------------------------------------------
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        vols = gmsh.model.getEntities(3)
        if not vols:
            raise MeshError(
                f"STEP 中未找到三维体: {step_path}。"
                "请确认导出的是实体（solid）而非曲面/壳体。"
            )
        vol_tags = [t for _, t in vols]
        log.append(f"imported {len(vol_tags)} volume(s): {vol_tags}")

        # --- 2. 体必须打 physical group（坑 1）-----------------------------
        gmsh.model.addPhysicalGroup(3, vol_tags, tag=VOLUME_TAG)
        gmsh.model.setPhysicalName(3, VOLUME_TAG, "solid")

        # --- 3. 边界标记 -----------------------------------------------------
        boundary_tags: dict[str, int] = {}
        next_tag = FACE_TAG_BASE
        for spec in boundaries:
            faces = _find_faces_on_plane(spec.axis, spec.value, spec.tol)
            if not faces:
                raise MeshError(
                    f"边界 {spec.name!r} 未匹配到任何面 "
                    f"({spec.axis}={spec.value}±{spec.tol})。"
                    "请检查坐标是否写错，或容差是否过小。"
                )
            tag = spec.tag if spec.tag is not None else next_tag
            next_tag = max(next_tag, tag) + 1
            gmsh.model.addPhysicalGroup(2, faces, tag=tag)
            gmsh.model.setPhysicalName(2, tag, spec.name)
            boundary_tags[spec.name] = tag
            log.append(f"boundary {spec.name!r}: tag={tag}, {len(faces)} face(s)")

        # --- 4. 网格控制 -----------------------------------------------------
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_mm))
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_mm) * 0.2)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)  # 避免曲率自适应把网格拉到极小
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)            # Delaunay

        # --- 4b. 局部细化（薄特征必须，见 RefinementBox 文档）----------------
        if refine_in:
            field_ids: list[int] = []
            for rb in refine_in:
                xmin, ymin, zmin, xmax, ymax, zmax = rb.bounds_mm
                fid = gmsh.model.mesh.field.add("Box")
                gmsh.model.mesh.field.setNumber(fid, "VIn", float(rb.size_mm))
                gmsh.model.mesh.field.setNumber(fid, "VOut", float(mesh_size_mm))
                gmsh.model.mesh.field.setNumber(fid, "XMin", float(xmin))
                gmsh.model.mesh.field.setNumber(fid, "XMax", float(xmax))
                gmsh.model.mesh.field.setNumber(fid, "YMin", float(ymin))
                gmsh.model.mesh.field.setNumber(fid, "YMax", float(ymax))
                gmsh.model.mesh.field.setNumber(fid, "ZMin", float(zmin))
                gmsh.model.mesh.field.setNumber(fid, "ZMax", float(zmax))
                if rb.thickness_mm > 0:
                    gmsh.model.mesh.field.setNumber(fid, "Thickness", float(rb.thickness_mm))
                field_ids.append(fid)
                log.append(f"refinement {rb.name!r}: size={rb.size_mm} mm in {rb.bounds_mm}")

            if len(field_ids) == 1:
                background = field_ids[0]
            else:
                background = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(background, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(background)

            # 使用背景网格时必须关掉其他尺寸来源，否则会互相覆盖
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        if second_order:
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

        # --- 5. 生成 ---------------------------------------------------------
        gmsh.model.mesh.generate(3)
        if second_order:
            gmsh.model.mesh.setOrder(2)
        if optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception as e:  # 优化失败不该让整个流程崩
                log.append(f"optimize skipped: {type(e).__name__}: {e}")

        # --- 6. 统计与写出 ---------------------------------------------------
        node_tags, coord, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        n_tets = 0
        for et, tags in zip(elem_types, elem_tags):
            if et in (4, 11):  # 4-node / 10-node tet
                n_tets += len(tags)
        n_cells = sum(len(t) for t in elem_tags)

        try:
            vol = sum(gmsh.model.occ.getMass(3, t) for t in vol_tags)
        except Exception:
            vol = float("nan")

        gmsh.write(str(out_msh))
        log.append(f"wrote {out_msh}")

        return MeshResult(
            msh_path=str(out_msh),
            n_nodes=n_nodes,
            n_cells=n_cells,
            n_tets=n_tets,
            volume_mm3=vol,
            boundary_tags=boundary_tags,
            mesh_size_mm=float(mesh_size_mm),
            log=log,
        )
    finally:
        gmsh.finalize()


# ==============================================================================
# 读入 dolfinx
# ==============================================================================
def read_into_dolfinx(msh_path: str | Path, gdim: int = 3):
    """把 ``.msh`` 读成 dolfinx 网格对象。

    ⚠️ FEniCSx 0.8 用 ``dolfinx.io.gmshio``；0.9+ 才是 ``dolfinx.io.gmsh``。
    不同版本返回值形态不同（0.8 返回 MeshData 对象，0.7 返回元组），此处统一处理。

    Returns
    -------
    (mesh, cell_tags, facet_tags)
    """
    from mpi4py import MPI

    try:
        from dolfinx.io import gmshio  # FEniCSx 0.8
    except ImportError as e:  # pragma: no cover
        raise MeshError(
            "无法导入 dolfinx.io.gmshio。若使用 FEniCSx 0.9+，接口已改名。"
        ) from e

    res = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, gdim=gdim)

    # 0.8: MeshData(mesh, cell_tags, facet_tags)；0.7: tuple
    mesh = getattr(res, "mesh", None)
    cell_tags = getattr(res, "cell_tags", None)
    facet_tags = getattr(res, "facet_tags", None)
    if mesh is None:
        if isinstance(res, (tuple, list)):
            mesh = res[0]
            cell_tags = res[1] if len(res) > 1 else None
            facet_tags = res[2] if len(res) > 2 else None
        else:
            raise MeshError(f"无法解析 gmshio.read_from_msh 的返回类型: {type(res)}")
    return mesh, cell_tags, facet_tags
