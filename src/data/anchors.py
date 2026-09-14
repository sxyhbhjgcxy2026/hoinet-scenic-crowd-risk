# -*- coding: utf-8 -*-
"""锚点观测与渗透率标定，对应论文式(11) 与 3.3.3 节。

论文 3.3.3 节：仅有信令观测时，人数与渗透率之间存在尺度耦合（命题 2），
无法同时确定。锚点直接观测真实人数，为渗透率提供绝对尺度：

    y^anc_t = P n_t + eps^anc_t                                    (11)

论文表 1 给出的锚点部署为“60 天 x 3 出入口 x 2 次”，共 360 次，即每个
闸机点位每日给出两次计数；并设置 100% / 50% / 25% / 0% 四档锚点稀疏化
实验检验缺失影响。

说明：论文关于锚点的时间分布另有两处表述，即“每 30 min 可用一次而实际
仅 12 个时段有记录”与“占全部时间步的 0.7%”，二者与表 1 的算式互不相容
（按表 1 算得 360 / 5760 = 6.25%）。本模块以表 1 的算式为准。

本模块提供：
    gate_anchor_times()      闸机锚点（每日两次）
    manual_sampling_times()  人工抽样锚点（每日两次）
    build_anchor_matrix()    由锚点记录构建选择矩阵 P
    anchor_mask()            生成 (T, A) 可用性掩码
    sparsify()               按档位稀释锚点
    estimate_penetration()   由锚点与信令联合估计渗透率的初值
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math

import torch


@dataclass
class AnchorLayout:
    """锚点点位布局。

    Attributes:
        gate_regions: 安装闸机的出入口所对应的区域下标，论文为 3 个。
        manual_regions: 人工抽样点位对应的区域下标。
        gate_times_per_day: 每日闸机计数次数，论文表 1 的算式为 2。
        manual_times_per_day: 每日人工抽样次数，论文为 2。
        steps_per_day: 每日基本时间步数，T=96 时为 96（15 min 步长）。
    """

    gate_regions: Sequence[int] = (0, 6, 12)
    manual_regions: Sequence[int] = (3, 9)
    gate_times_per_day: int = 2
    manual_times_per_day: int = 2
    steps_per_day: int = 96


def build_anchor_matrix(layout: AnchorLayout, n_regions: int) -> torch.Tensor:
    """构建式(11) 的锚点选择矩阵 P，形状 (A, N)。

    每行对应一个锚点点位，该行在对应区域位置为 1，其余为 0。闸机点位直接
    观测所在出入口区域的人数；人工抽样点位同理。
    """
    regions = list(layout.gate_regions) + list(layout.manual_regions)
    a = len(regions)
    p = torch.zeros(a, n_regions)
    for i, k in enumerate(regions):
        if not (0 <= k < n_regions):
            raise ValueError('锚点区域下标 %d 超出范围 N=%d' % (k, n_regions))
        p[i, k] = 1.0
    return p


def gate_anchor_times(steps_per_day: int = 96, times_per_day: int = 2,
                      open_hour: int = 8, close_hour: int = 18
                      ) -> torch.Tensor:
    """闸机锚点在一天内的时间步：开放时段内均匀取 times_per_day 个。

    论文表 1 给出的锚点数量算式为“60 天 x 3 出入口 x 2 次”，即每个闸机
    点位每日给出两次计数，故这里在开放时段内均匀取 times_per_day 个时间步。

    注意论文另有一处表述称锚点“每 30 min 可用一次而实际仅 12 个时段有记录”
    以及“占全部时间步的 0.7%”，这三个表述互不相容。本实现以表 1 的算式为准，
    即每日两次、60 天共 360 次。
    """
    per_hour = steps_per_day / 24.0
    lo, hi = int(open_hour * per_hour), int(close_hour * per_hour)
    span = hi - lo
    return torch.tensor([lo + span * (i + 1) // (times_per_day + 1)
                         for i in range(times_per_day)])


def manual_sampling_times(steps_per_day: int = 96,
                          times_per_day: int = 2) -> torch.Tensor:
    """人工抽样锚点的时间步：每日两次，均匀分布在开放时段内。"""
    per_hour = steps_per_day / 24.0
    lo, hi = int(8 * per_hour), int(18 * per_hour)
    span = hi - lo
    return torch.tensor([lo + span * (i + 1) // (times_per_day + 1)
                         for i in range(times_per_day)])


def anchor_mask(layout: AnchorLayout, n_gates: int, n_manual: int,
                t_steps: int, steps_per_day: Optional[int] = None
                ) -> torch.Tensor:
    """生成 (T, A) 的锚点可用性掩码。

    列的前 n_gates 列对应闸机点位，其后 n_manual 列对应人工抽样点位。
    每日的采样模式在整条时间轴上**逐日重复**：论文表 1 的锚点算式为
    “60 天 x 3 出入口 x 2 次”，即每个点位每日采样两次，故 T 长于一天时
    必须把一天的采样时刻平移到每一天，而不能把一天的采样时刻拉伸到
    整条时间轴上。

    早先的实现把一天的时间步按下标比例拉伸到 T 上，结果 60 天的窗口里
    闸机只触发 40 次（每个点位每天不足一次），与表 1 的算式相差近一个
    数量级。

    Args:
        t_steps: 序列长度 T。
        steps_per_day: 每日基本步数；None 时取 layout.steps_per_day。
    """
    spd = int(steps_per_day or layout.steps_per_day)
    if spd <= 0:
        raise ValueError('steps_per_day 必须为正')
    mask = torch.zeros(t_steps, n_gates + n_manual)
    gate_t = gate_anchor_times(spd, layout.gate_times_per_day)
    man_t = manual_sampling_times(spd, layout.manual_times_per_day)

    n_days = int(math.ceil(t_steps / float(spd)))
    for d in range(n_days):
        off = d * spd
        for times, lo, hi in ((gate_t, 0, n_gates),
                              (man_t, n_gates, n_gates + n_manual)):
            idx = (times + off)
            idx = idx[(idx >= 0) & (idx < t_steps)].unique()
            if len(idx):
                mask[idx, lo:hi] = 1.0
    return mask


def sparsify(mask: torch.Tensor, level: float, seed: int = 0,
             n_gates: Optional[int] = None) -> torch.Tensor:
    """按档位稀释锚点，对应论文 100% / 50% / 25% / 0% 的稀疏化实验。

    Args:
        mask:  (T, A) 原始掩码，列的顺序与 build_anchor_matrix 一致，即
               前 n_gates 列是闸机锚点，其后是人工抽样锚点。
        level: 保留比例，取 1.0 / 0.5 / 0.25 / 0.0。
        seed:  随机种子，保证实验可重复。
        n_gates: 前多少列属于常态化部署的闸机锚点。闸机不参与稀释，稀疏化
               实验针对的是临时增加的人工抽样密度，故这些列保持不动。
               默认 None 表示全部列都参与稀释；场景中应传
               ScenarioConfig.n_anchors_gate。

     早先的实现把 n_gates 取成 mask.shape[1]，即总列数，于是待稀释的列区间
    为空、任何档位都返回原掩码，100% 与 50% 两档给出完全相同的锚点。

    稀释按列独立进行，每列保留 round(该列锚点数 * level) 个且至少保留 1 个。
    因此在锚点总数很少时低档位会退化：某一列只有 2 个锚点时，50% 与 25% 都
    保留 1 个。论文的 60 天设定下测试区间有 36 个人工抽样锚点，各档位可区分
    （100% / 50% / 25% 分别为 90 / 72 / 62 个可用锚点，其中闸机列恒为 54 个）。

    Returns:
        稀释后的 (T, A) 掩码。
    """
    if not 0.0 <= level <= 1.0:
        raise ValueError('稀疏化档位应在 [0, 1] 内')
    if level >= 1.0:
        return mask.clone()
    if level <= 0.0:
        return torch.zeros_like(mask)
    if n_gates is None:
        first = 0
    else:
        first = int(n_gates)
        if not 0 <= first <= mask.shape[1]:
            raise ValueError('n_gates=%d 超出列数 %d' % (first, mask.shape[1]))
    g = torch.Generator().manual_seed(seed)
    out = mask.clone()
    for col in range(first, mask.shape[1]):
        idx = mask[:, col].nonzero().flatten()
        if idx.numel() == 0:
            continue
        keep = torch.randperm(idx.numel(), generator=g)[
            :max(1, int(round(idx.numel() * level)))]
        out[:, col] = 0.0
        out[idx[keep], col] = 1.0
    return out


# ------------------------------------------------------------------ 渗透率标定
def estimate_penetration(anchor_counts: torch.Tensor,
                         signal_counts: torch.Tensor,
                         mixing: torch.Tensor,
                         time_agg: torch.Tensor,
                         region_area: Optional[torch.Tensor] = None,
                         tau_smooth: float = 0.5,
                         n_iter: int = 50) -> torch.Tensor:
    """由锚点与信令联合估计渗透率的初值，供式(10) 的 eta^(0) 使用。

    论文 5.2 节只说明渗透率初值“由训练集锚点与信令联合估计”，未给出具体
    公式，本函数给出一种不用人数真值的实现。

    估计依据是空间混合矩阵的列归一化性质。B 的列和为 1（每个区域的设备
    按面积份额分摊到覆盖它的各扇区），因此对所有扇区求和时

        sum_l y^sig_{l,t}
            = sum_l sum_k B_{lk} pi_{kt} n_{kt}
            = sum_k pi_{kt} n_{kt} (sum_l B_{lk})
            = sum_k pi_{kt} n_{kt}

    即扇区设备总数恒等于各区域的设备总数。若渗透率在空间上近似均匀，记
    pi_{kt} = pi_t，则

        pi_t = (sum_l y^sig_{l,t}) / (sum_k n_{kt})

    其中总人数 sum_k n_{kt} 由锚点时刻的观测人数按面积份额外推：锚点只覆盖
    少数区域，故用锚点区域面积占全域面积的比例把观测人数放大到全域。

    做法说明：这里不采用“把信令按 B 的伪逆反摊回区域、再逐区域作比值”的
    思路。B 的形状为 (L, N) 且 L < N，其伪逆与 B 的乘积是秩不超过 L 的投影
    而非单位阵，反摊结果会把全域设备数摊到所有区域上；实测该路径给出的
    渗透率约为真值的 15 倍，不可用。按上式先对扇区求和再作比值则只依赖
    “B 的列和为 1”这一精确性质，实测在锚点时刻的相对误差约 20%。

    估计只在锚点时刻有值，其余时刻按时间扩散插值填补。

    Args:
        anchor_counts: (T, N) 锚点时刻该区域的人数观测，非锚点时刻为 0。
        signal_counts: (J, L) 信令设备数观测（窗口粒度）。
        mixing:        (L, N) 空间混合矩阵 B，列和为 1。
        time_agg:      (J, T) 时间聚合矩阵 W，用于把窗口观测摊到基本步。
        region_area:   (N,) 各区域面积，用于把锚点人数外推为全域总人数；
                       None 时退化为按区域个数等比外推。
        tau_smooth:    时间平滑强度，0 表示不平滑。
        n_iter:        扩散插值的迭代次数。

    Returns:
        (T, N) 渗透率初值，同一时刻的各区域取相同值。
    """
    t_steps, n_regions = anchor_counts.shape
    if mixing.shape[1] != n_regions:
        raise ValueError('mixing 的列数应为 %d，实际 %d'
                         % (n_regions, mixing.shape[1]))

    # 把窗口粒度的信令观测摊到基本步：第 t 步取覆盖它的各窗口的加权平均
    w_sum = time_agg.sum(dim=0).clamp(min=1e-9)                     # (T,)
    step_counts = (time_agg.t() @ signal_counts) / w_sum.unsqueeze(1)   # (T, L)
    sector_total = step_counts.sum(dim=1, keepdim=True)             # (T, 1)

    # 锚点人数 -> 全域总人数的外推系数
    valid = anchor_counts > 0                                      # (T, N)
    if region_area is None:
        share = (valid.sum(dim=1).clamp(min=1).float()
                 / float(n_regions))                                # (T,)
    else:
        area = region_area.to(anchor_counts.dtype).view(1, -1)
        share = ((area * valid.float()).sum(dim=1)
                 / area.sum().clamp(min=1e-9))                      # (T,)
    share = share.view(-1, 1)                                       # (T, 1)
    total_pop = anchor_counts.sum(dim=1, keepdim=True) / share.clamp(min=1e-6)

    pi_scalar = torch.full((t_steps, 1), float('nan'))
    ok = valid.any(dim=1, keepdim=True) & (total_pop > 0)
    pi_scalar = torch.where(ok, sector_total / total_pop.clamp(min=1e-6),
                            pi_scalar)

    # 时间维扩散插值：缺失时刻用相邻已观测时刻的均值填补。
    # 注意缺失位置必须直接取邻域均值，不能与自身做加权混合——缺失位置的值
    # 是 NaN，torch.nan_to_num 会把它变成 0，混合相当于每轮把估计值减半，
    # 远离锚点的时刻会因此衰减到 0。
    for _ in range(n_iter):
        missing = torch.isnan(pi_scalar)
        if not missing.any():
            break
        prev = torch.roll(pi_scalar, 1, dims=0)
        nxt = torch.roll(pi_scalar, -1, dims=0)
        prev[0], nxt[-1] = pi_scalar[0], pi_scalar[-1]
        cnt = (~torch.isnan(prev)).float() + (~torch.isnan(nxt)).float()
        if not bool((cnt > 0).any()):
            break
        acc = (torch.nan_to_num(prev, nan=0.0)
               + torch.nan_to_num(nxt, nan=0.0))
        avg = acc / cnt.clamp(min=1.0)
        # 缺失位置取邻域均值；已观测位置按 tau_smooth 做一次松弛平滑，
        # tau_smooth = 0 时完全保留观测值
        updated = torch.where(missing, avg,
                              tau_smooth * avg
                              + (1.0 - tau_smooth)
                              * torch.nan_to_num(pi_scalar, nan=0.0))
        pi_scalar = torch.where(cnt > 0, updated, pi_scalar)

    # 仍然缺失的位置用已估计时刻的中位数兜底；全部缺失时返回 NaN 之外的默认值
    known = pi_scalar[~torch.isnan(pi_scalar)]
    fallback = float(known.median()) if known.numel() else 0.3
    pi_scalar = torch.nan_to_num(pi_scalar, nan=fallback)
    return pi_scalar.expand(t_steps, n_regions).contiguous()
