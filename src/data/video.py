# -*- coding: utf-8 -*-
"""监控视频的处理，对应论文式(6)、式(7) 与 3.3.1 节。

论文 3.3.1 节把视频观测写成带掩码的局部人数观测：

    y^vis_{i,t} = m^vis_{i,t} * sum_k H_ik n_{k,t} + eps^vis_{i,t}       (6)

    sigma^{2,vis}_{i,t} = sigma_0^2 + sigma_1^2 y^vis_{i,t} + sigma_2^2 o_{i,t}  (7)

其中 H_ik 为摄像头 i 对区域 a_k 的有效覆盖比例，由部署位置、朝向与视场角
投影得到；m^vis 为有效观测权重（在线状态、图像质量、目标是否在有效视野内）；
o_{i,t} 为遮挡比例或图像质量指标。式(7) 的异方差性使高人数、高遮挡的观测
在反演中自动获得较低权重。

本模块提供：
    coverage_from_fov()           由摄像机位姿与视场角投影得到 H
    coverage_from_overlap_areas() 由重叠面积直接给出 H（无 GIS 时使用）
    sample_frames()               每 60 s 一帧的抽帧索引
    quality_weights()             由亮度与清晰度得到 m^vis 与 o
    occlusion_from_density()      由密度估计遮挡比例
    simulate()                    按式(6)(7) 生成视觉观测
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


# ------------------------------------------------------------------ 覆盖矩阵
@dataclass
class CameraPose:
    """摄像头部署参数。平面坐标单位与区域多边形一致（建议米）。"""

    x: float
    y: float
    yaw_deg: float                  # 朝向，0 为正东，逆时针为正
    fov_deg: float = 60.0           # 水平视场角
    range_m: float = 80.0           # 有效识别距离


def coverage_from_fov(cameras: Sequence[CameraPose],
                      region_polygons: List,
                      n_samples: int = 40) -> torch.Tensor:
    """由摄像机位姿与视场角投影计算覆盖矩阵 H，对应式(6) 的 H_ik。

    做法：在每个区域外接矩形内均匀撒点，统计落在“摄像机有效视锥”内的
    采样点比例，作为该摄像头对该区域的有效覆盖比例。对完全没有采样点落入
    视锥的区域，H_ik = 0，这正是论文命题 1 所刻画的缺口区域。

    需要 shapely。

    Args:
        cameras: 摄像头参数列表，长度 M。
        region_polygons: shapely Polygon 列表，长度 N。
        n_samples: 每区域每方向的采样点数。

    Returns:
        (M, N) 覆盖比例矩阵，取值 [0, 1]。
    """
    try:
        from shapely.geometry import Point
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError('coverage_from_fov 需要 shapely') from exc

    m, n = len(cameras), len(region_polygons)
    h = torch.zeros(m, n)
    for i, cam in enumerate(cameras):
        yaw = math.radians(cam.yaw_deg)
        half = math.radians(cam.fov_deg) / 2.0
        for k, poly in enumerate(region_polygons):
            minx, miny, maxx, maxy = poly.bounds
            hits = 0
            total = 0
            for a in range(n_samples):
                for b in range(n_samples):
                    px = minx + (maxx - minx) * (a + 0.5) / n_samples
                    py = miny + (maxy - miny) * (b + 0.5) / n_samples
                    if not poly.contains(Point(px, py)):
                        continue
                    total += 1
                    dx, dy = px - cam.x, py - cam.y
                    dist = math.hypot(dx, dy)
                    if dist > cam.range_m:
                        continue
                    ang = math.atan2(dy, dx)
                    # 归一化到 [-pi, pi] 后与朝向比较
                    delta = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(delta) <= half:
                        hits += 1
            if total > 0:
                h[i, k] = hits / total
    return h.clamp(0.0, 1.0)


def coverage_from_overlap_areas(overlap_area: torch.Tensor,
                                region_area: torch.Tensor) -> torch.Tensor:
    """由重叠面积直接给出 H，便于在没有 GIS 数据的环境中使用。

    Args:
        overlap_area: (M, N) 摄像头视野与该区域的重叠面积。
        region_area:  (N,) 区域总面积。

    Returns:
        (M, N) 覆盖比例，已截断到 [0, 1]。
    """
    h = overlap_area.float() / region_area.view(1, -1).clamp(min=1e-9)
    return h.clamp(0.0, 1.0)


def gap_regions(coverage: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """返回缺口区域的布尔掩码：论文中 H_{:,k} = 0 的区域（命题 1）。"""
    return coverage.max(dim=0).values <= tol


# ------------------------------------------------------------------ 抽帧与质量
def sample_frames(step_minutes: int = 15, interval_seconds: int = 60,
                  n_substeps: int = 30) -> torch.Tensor:
    """每 60 s 采一帧的抽帧方案。

    论文 5.1 节：视频每 60 秒采样一帧，累计约 104 万帧（12 路摄像头 x 60 天
    x 1440 帧/天）；5.4 节：每个基本步内抽取 Ns = 30 个视频子步。

    Args:
        step_minutes: 基本时间步长度（分钟）。
        interval_seconds: 抽帧间隔（秒）。
        n_substeps: 每个基本步内的视频子步数 Ns。

    Returns:
        (n_substeps,) 各子步在基本步内的起始秒数。
    """
    step_seconds = step_minutes * 60
    sub_len = step_seconds / n_substeps
    # 子步起始时刻，并按抽帧间隔对齐到最近的可用帧
    starts = torch.arange(n_substeps, dtype=torch.float32) * sub_len
    aligned = (starts / interval_seconds).floor() * interval_seconds
    return aligned


def frames_per_camera(days: int, cameras: int = 12,
                      interval_seconds: int = 60) -> int:
    """按抽帧方案估算帧总数，用于核对论文给出的“约 104 万帧”。"""
    return int(days * 24 * 3600 // interval_seconds * cameras)


def quality_weights(brightness: torch.Tensor, sharpness: torch.Tensor,
                    online: Optional[torch.Tensor] = None,
                    b_lo: float = 40.0, b_hi: float = 90.0,
                    s_lo: float = 20.0, s_hi: float = 120.0
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """由亮度与清晰度得到有效观测权重 m^vis 与质量指标 o。

    论文把 m^vis 解释为“摄像头是否在线、图像质量是否合格以及目标是否处于
    有效视野内”。此处用亮度与清晰度两个可测指标合成：

        q = clip((brightness - b_lo) / (b_hi - b_lo), 0, 1)
          * clip((sharpness  - s_lo) / (s_hi - s_lo), 0, 1)
        m^vis = q * online
        o     = 1 - q                  （式(7) 中的遮挡/质量指标）

    Args:
        brightness: (..., M) 平均亮度，建议 0-255。
        sharpness:  (..., M) 清晰度（如拉普拉斯方差）。
        online:     (..., M) 在线状态，1 在线 0 离线；None 时视为全在线。

    Returns:
        (m_vis, o)，形状同输入。
    """
    def _clip01(x):
        return x.clamp(0.0, 1.0)

    qb = _clip01((brightness - b_lo) / (b_hi - b_lo))
    qs = _clip01((sharpness - s_lo) / (s_hi - s_lo))
    q = qb * qs
    if online is not None:
        q = q * online.float()
    return q, 1.0 - q


def occlusion_from_density(n_true: torch.Tensor, coverage: torch.Tensor,
                           k_occ: float = 0.12) -> torch.Tensor:
    """由局部人数估计遮挡比例：人越多，相互遮挡越严重。

    o_{i,t} = 1 - exp(-k_occ * sum_k H_ik n_{k,t})

    该式满足式(7) 的单调性要求：遮挡比例随视野内人数增加而增大，
    因而高人数观测的方差更大、在反演中的权重更低。

    Args:
        n_true:  (T, N) 真实人数。
        coverage:(M, N) 覆盖矩阵 H。
        k_occ:   遮挡系数。

    Returns:
        (T, M) 遮挡比例，取值 [0, 1)。
    """
    local = n_true @ coverage.t()                               # (T, M)
    return 1.0 - torch.exp(-k_occ * local)


# ------------------------------------------------------------------ 观测仿真
@dataclass
class VisualNoise:
    """式(7) 的噪声系数。论文给出形式但未给出取值，此处为可调默认。"""

    sigma0: float = 1.0
    sigma1: float = 0.05
    sigma2: float = 4.0
    seed: int = 0


def simulate(n_true: torch.Tensor, coverage: torch.Tensor,
             m_vis: Optional[torch.Tensor] = None,
             o_vis: Optional[torch.Tensor] = None,
             noise: Optional[VisualNoise] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """按式(6) 与式(7) 生成视觉观测及其方差。

    Args:
        n_true:  (T, N) 真实人数。
        coverage:(M, N) 覆盖矩阵 H。
        m_vis:   (T, M) 有效观测权重；None 时全为 1。
        o_vis:   (T, M) 遮挡/质量指标；None 时由 occlusion_from_density 给出。
        noise:   噪声系数。

    Returns:
        (y_vis (T, M), var_vis (T, M))
    """
    t = n_true.shape[0]
    m = coverage.shape[0]
    if m_vis is None:
        m_vis = torch.ones(t, m)
    if o_vis is None:
        o_vis = occlusion_from_density(n_true, coverage)

    mean = m_vis * (n_true @ coverage.t())                      # 式(6) 的确定性部分
    if noise is None:                                           # 只取确定性部分
        return mean, torch.zeros_like(mean)

    var = (noise.sigma0 ** 2 + noise.sigma1 ** 2 * mean.clamp(min=0)
           + noise.sigma2 ** 2 * o_vis)                          # 式(7)
    g = torch.Generator().manual_seed(noise.seed)
    eps = torch.randn(mean.shape, generator=g) * var.sqrt()
    return (mean + eps).clamp(min=0.0), var
