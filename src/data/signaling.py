# -*- coding: utf-8 -*-
"""手机信令数据的处理，对应论文式(9)、式(17) 与 3.3.2 节。

论文 3.3.2 节指出信令观测的三个事实，本模块逐一落实：

    1. 空间上是多区域的混合观测      -> sector_region_mixing_matrix() 得到 B
    2. 记录的是设备数而非人数        -> 由渗透率 pi 换算，见 simulate() 的说明
    3. 时间上是对多个状态的聚合      -> time_aggregation_matrix() 得到 W

隐私处理在 aggregate_device_counts() 中完成：原始记录进入本模块时已完成
运营商侧的聚合与脱敏，只保留 (时间窗口, 基站扇区) 粒度上的设备计数，
不保留用户标识、终端号码与位置轨迹。这与论文伦理声明中的表述一致。

关于时间分辨率的说明（论文内部存在一处不一致）：
    3.1 节写“连续时间离散为 T 个分钟级时间步”
    5.1 节写“基本时间步为 15 分钟，序列长度 T=96”，96 x 15 min = 24 h
本模块按后者取默认值（step_minutes=15），此时信令窗口与基本步等长，
W 退化为单位矩阵；若要在代码中恢复式(9) 与式(17) 的非退化形式，
把 step_minutes 设为 1、signal_window_minutes 保持 15 即可。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ------------------------------------------------------------------ 时间聚合
def time_aggregation_matrix(n_steps: int, step_minutes: int = 15,
                            window_minutes: int = 15,
                            soft: bool = False) -> torch.Tensor:
    """构建式(9) 的时间聚合矩阵 W。

    W[j, t] 表示分钟状态 t 对第 j 个信令窗口的贡献。

    按分钟折算成“基本步”后，第 t 个基本步覆盖
    [t*step_minutes, (t+1)*step_minutes) 分钟区间。

    Args:
        n_steps: 基本时间步数 T。
        step_minutes: 基本时间步长度（分钟）。
        window_minutes: 信令窗口长度（分钟），须能被 step_minutes 整除或反之。
        soft: True 时按时间重叠比例分配（窗口与基本步不对齐时的软聚合）；
              False 时按窗口包含关系硬分配。

    Returns:
        (J, T) 非负矩阵，每行非负、行和为窗口覆盖的基本步数（硬分配时）。
    """
    if window_minutes <= 0 or step_minutes <= 0:
        raise ValueError('时间步长必须为正')
    total_minutes = n_steps * step_minutes
    n_windows = total_minutes // window_minutes
    if n_windows == 0:
        raise ValueError('窗口长于总时长，无法聚合')

    w = torch.zeros(n_windows, n_steps)
    win_len = window_minutes / step_minutes                    # 每窗覆盖的基本步数
    for j in range(n_windows):
        lo = j * win_len
        hi = (j + 1) * win_len
        for t in range(n_steps):
            if soft:
                overlap = max(0.0, min(hi, t + 1) - max(lo, t))
                if overlap > 0:
                    w[j, t] = overlap / win_len
            else:
                if lo <= t < hi:
                    w[j, t] = 1.0
    return w


def window_index(n_steps: int, step_minutes: int = 15,
                 window_minutes: int = 15) -> torch.Tensor:
    """返回每个基本步所属的信令窗口下标，长度 T。"""
    assert step_minutes == window_minutes, (
        '仅在窗口与基本步等长时窗口下标是单值映射；'
        '其他情形请直接用 time_aggregation_matrix 得到的 W')
    return torch.arange(n_steps)


# ------------------------------------------------------------------ 空间混合
def sector_region_mixing_matrix(sector_polygons: List,
                                region_polygons: List,
                                normalize: str = 'column'
                                ) -> torch.Tensor:
    """构建式(9) 的空间混合矩阵 B。

    B[l, k] 表示区域 a_k 的信令设备落到信令单元（基站扇区）l 的比例。
    Eq(9) 中 B 把区域设备数映射为扇区设备数，故默认按列归一化，即
    每个区域的全部设备按覆盖面积比例分摊到覆盖它的各扇区上，
    sum_l B[l, k] = 1。若改为 normalize='row'，则按扇区归一化，适用于
    把扇区观测反摊回区域的场景。

    需要 shapely。多边形为经纬度或投影坐标均可，只需同一坐标系。

    Args:
        sector_polygons: shapely Polygon 列表，长度 L。
        region_polygons: shapely Polygon 列表，长度 N。
        normalize: 'column' | 'row' | 'none'。

    Returns:
        (L, N) 非负矩阵。
    """
    try:
        from shapely.geometry import shape  # noqa: F401
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError('sector_region_mixing_matrix 需要 shapely') from exc

    l, n = len(sector_polygons), len(region_polygons)
    b = torch.zeros(l, n)
    for i in range(l):
        sec = sector_polygons[i]
        for k in range(n):
            reg = region_polygons[k]
            inter = sec.intersection(reg)
            if inter.is_empty:
                continue
            # 用区域落在扇区内的面积占该区域面积的比例作贡献
            b[i, k] = float(inter.area) / max(float(reg.area), 1e-9)

    if normalize == 'column':
        col = b.sum(dim=0, keepdim=True)
        b = torch.where(col > 0, b / col.clamp(min=1e-9), b)
    elif normalize == 'row':
        row = b.sum(dim=1, keepdim=True)
        b = torch.where(row > 0, b / row.clamp(min=1e-9), b)
    elif normalize != 'none':
        raise ValueError("normalize 只能取 'column' / 'row' / 'none'")
    return b


def mixing_from_overlap_areas(overlap_area: torch.Tensor,
                              region_area: torch.Tensor,
                              normalize: str = 'column') -> torch.Tensor:
    """直接由面积重叠表构建 B，便于在没有 shapely 的环境中使用。

    Args:
        overlap_area: (L, N) 第 l 个扇区与第 k 个区域的重叠面积。
        region_area: (N,) 各区域总面积。
        normalize: 同 sector_region_mixing_matrix。
    """
    b = overlap_area.float() / region_area.view(1, -1).clamp(min=1e-9)
    b = b.clamp(0.0, 1.0)
    if normalize == 'column':
        b = b / b.sum(dim=0, keepdim=True).clamp(min=1e-9)
    elif normalize == 'row':
        b = b / b.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return b


# ------------------------------------------------------------------ 脱敏与聚合
def aggregate_device_counts(records: Sequence[Tuple[float, int]],
                            n_windows: int, n_sectors: int,
                            window_seconds: int = 900,
                            t0: float = 0.0) -> torch.Tensor:
    """把脱敏后的原始信令记录聚合成 (窗口, 扇区) 粒度的设备计数。

    这是运营商侧完成的一步，对应论文 5.1 节的“信令来自 7 个基站扇区，
    按 15 分钟聚合”，以及伦理声明中的“仅保留按 15 分钟时间窗口与基站扇区
    聚合的设备计数”。

    输入 records 只允许包含两项：时间戳与该时刻所在的扇区编号。
    **不接受任何用户标识字段**，函数会在入口处检查并拒绝多余元素，
    以免上游误把标识符传进来。

    Args:
        records: [(timestamp_seconds, sector_id), ...]。
        n_windows: 窗口数 J。
        n_sectors: 扇区数 L。
        window_seconds: 窗口长度，默认 900 s（15 min）。
        t0: 起始时间戳。

    Returns:
        (J, L) 设备计数。
    """
    counts = torch.zeros(n_windows, n_sectors)
    for rec in records:
        if len(rec) != 2:
            raise ValueError(
                '信令记录只允许 (timestamp, sector_id) 两项，'
                '不接受用户标识等额外字段；当前长度 %d' % len(rec))
        ts, sec = rec
        if not (0 <= int(sec) < n_sectors):
            continue
        w = int((ts - t0) // window_seconds)
        if 0 <= w < n_windows:
            counts[w, int(sec)] += 1.0
    return counts


# ------------------------------------------------------------------ 观测仿真
@dataclass
class SignalNoise:
    """信令观测的噪声设定。论文未给出信令噪声模型，此处按乘性 + 加性组合。"""

    relative: float = 0.03          # 乘性噪声标准差（占观测值比例）
    absolute: float = 1.0           # 加性噪声标准差（设备数）
    seed: int = 0


def simulate(n_true: torch.Tensor, pi: torch.Tensor,
             mixing: torch.Tensor, time_agg: torch.Tensor,
             noise: Optional[SignalNoise] = None) -> torch.Tensor:
    """按式(9) 由真实人数与渗透率生成信令观测。

        y^sig_j = sum_t W_jt B diag(pi_t) n_t + eps_j

    Args:
        n_true:  (T, N) 真实区域人数。
        pi:      (T, N) 真实渗透率。
        mixing:  (L, N) 空间混合矩阵 B。
        time_agg:(J, T) 时间聚合矩阵 W。
        noise:   噪声设定；None 时不加噪。

    Returns:
        (J, L) 信令设备数观测。
    """
    mixed = pi * n_true                                          # (T, N) 设备数
    win = torch.einsum('jt,tn->jn', time_agg, mixed)             # (J, N)
    obs = win @ mixing.t()                                       # (J, L)
    if noise is not None:
        g = torch.Generator(device='cpu').manual_seed(noise.seed)
        shape = tuple(obs.shape)
        n1 = torch.randn(shape, generator=g, dtype=obs.dtype).to(obs.device)
        n2 = torch.randn(shape, generator=g, dtype=obs.dtype).to(obs.device)
        eps = n1 * noise.absolute + n2 * noise.relative * obs.abs()
        obs = (obs + eps).clamp(min=0.0)
    return obs


def penetration_bounds_check(pi: torch.Tensor, pi_min: float,
                             pi_max: float) -> Dict[str, float]:
    """检查渗透率是否落在物理区间内，供数据质检使用。"""
    return {'min': float(pi.min()), 'max': float(pi.max()),
            'below': int((pi < pi_min).sum()), 'above': int((pi > pi_max).sum())}


def data_availability_statement() -> str:
    """与论文「数据可用性声明」一致的说明文本，供 README 与发布页复用。"""
    return (
        '本文使用的 ShanghaiTech 与 Mall 为公开人群计数数据集，可分别从原始'
        '发布来源获取。自建景区多模态数据集（Scenic-MM）涉及景区运营数据与'
        '运营商信令数据的使用授权限制，不予公开。')
