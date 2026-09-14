# -*- coding: utf-8 -*-
"""景区区域图与图拉普拉斯，对应论文式(12)。

论文 3.4 节：游客沿景区道路与功能动线移动，相邻区域的人数变化具有连续性，
故按景区功能区连通关系构建图 G = (A, E, W_g)，并取空间平滑项

    R_spa(N) = 1/2 * sum_t n_t^T L n_t,   L = D - W_g          (12)

本模块从区域邻接关系构建 W_g、D 与 L，并提供切比雪夫卷积所需的缩放形式
L~ = 2L/lambda_max - I。

在线部署时，邻接矩阵由景区 GIS 图层给出：两个子区域只要存在可直接通行的
道路连接（含台阶、栈道、内部通道），即置边；边权取通道宽度或通行能力。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def adjacency_from_edges(n_nodes: int,
                         edges: Iterable[Tuple[int, int]],
                         weights: Sequence[float] = None,
                         symmetric: bool = True) -> torch.Tensor:
    """由边表构建邻接矩阵 W_g。

    Args:
        n_nodes: 区域数 N。
        edges: (i, j) 序列。
        weights: 每条边的权重；None 时全取 1。
        symmetric: 是否对称化（景区道路双向通行时为 True）。

    Returns:
        (N, N) 稠密对称邻接矩阵。
    """
    a = torch.zeros(n_nodes, n_nodes)
    edges = list(edges)
    if weights is None:
        weights = [1.0] * len(edges)
    for (i, j), w in zip(edges, weights):
        a[i, j] = w
        if symmetric:
            a[j, i] = w
    a.fill_diagonal_(0.0)
    return a


def adjacency_from_geometry(region_polygons: List, share_threshold: float = 0.0
                            ) -> torch.Tensor:
    """由区域多边形构建邻接矩阵：共享边界长度大于阈值的两区域视为相邻。

    需要 shapely。若环境中没有 shapely，请改用 adjacency_from_edges 手工给出边表。

    Args:
        region_polygons: shapely Polygon 列表，长度 N。
        share_threshold: 共享边界长度阈值（米），默认 0 表示只要相接就算相邻。
    """
    try:
        from shapely.ops import shared_paths  # noqa: F401
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            'adjacency_from_geometry 需要 shapely；'
            '也可改用 adjacency_from_edges 直接给边表') from exc

    n = len(region_polygons)
    a = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            inter = region_polygons[i].boundary.intersection(
                region_polygons[j].boundary)
            length = getattr(inter, 'length', 0.0)
            if length > share_threshold:
                a[i, j] = a[j, i] = float(length)
    return a


def graph_laplacian(adjacency: torch.Tensor, normalized: bool = True
                    ) -> Dict[str, torch.Tensor]:
    """构建度矩阵 D 与拉普拉斯 L。

    Args:
        adjacency: (N, N) 非负邻接矩阵 W_g。
        normalized: True 返回对称归一化拉普拉斯 I - D^{-1/2} A D^{-1/2}；
                    False 返回组合拉普拉斯 D - A（即论文式(12) 的 L）。

    Returns:
        {'degree': (N,), 'laplacian': (N, N)}
    """
    w = adjacency.float()
    w = (w + w.t()) / 2.0                                       # 对称化
    degree = w.sum(dim=1)
    d_inv_sqrt = torch.where(degree > 0, degree.rsqrt(),
                             torch.zeros_like(degree))
    if normalized:
        lap = torch.eye(w.shape[0]) - (d_inv_sqrt[:, None] * w
                                       * d_inv_sqrt[None, :])
    else:
        lap = torch.diag(degree) - w
    return {'degree': degree, 'laplacian': lap}


def chebyshev_scaled(lap: torch.Tensor) -> torch.Tensor:
    """缩放拉普拉斯：L~ = 2L/lambda_max - I，使特征值落在 [-1, 1]。

    切比雪夫图卷积要求输入矩阵的特征值在 [-1, 1] 内，否则高阶项会发散。
    lambda_max 用幂迭代估计，避免对 N x N 矩阵做完整特征分解。

    Args:
        lap: (N, N) 对称拉普拉斯矩阵。

    Returns:
        (N, N) 缩放后的 L~。
    """
    n = lap.shape[0]
    v = torch.randn(n, dtype=lap.dtype, device=lap.device)
    v = v / v.norm().clamp(min=1e-9)
    lam = 0.0
    for _ in range(100):
        w = lap @ v
        norm = w.norm()
        if norm < 1e-12:
            break
        v = w / norm
        lam = float(v @ (lap @ v))
    lam = max(lam, 1e-3)                                        # 防止除零
    return 2.0 * lap / lam - torch.eye(n, dtype=lap.dtype, device=lap.device)


def build_laplacian(adjacency: torch.Tensor,
                    normalized: bool = True) -> torch.Tensor:
    """一步到位：邻接矩阵 -> 切比雪夫卷积可直接使用的 L~。"""
    lap = graph_laplacian(adjacency, normalized=normalized)['laplacian']
    return chebyshev_scaled(lap)


def region_areas_from_polygons(region_polygons: List) -> torch.Tensor:
    """由多边形计算各区域有效面积 s_k（式(1) 的分母）。"""
    return torch.tensor([float(p.area) for p in region_polygons])
