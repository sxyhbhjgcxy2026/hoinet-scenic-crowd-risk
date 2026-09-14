# -*- coding: utf-8 -*-
"""风险标签与阈值，对应论文式(3) 与 3.1 节关于阈值的说明。

论文 3.1 节：
  - 阈值 tau_{k,1} < tau_{k,2} < tau_{k,3} 按区域分别设定，依据是该区域的
    最大承载量、有效面积、疏散通道宽度与历史安全事件记录，由景区管理部门
    核定，属外部固定输入而非从数据学习，因此不存在标签泄漏。
  - 拥堵持续状态 q_{k,t} = lambda_q q_{k,t-1} + (1 - lambda_q) ReLU(rho - tau_{k,1})   (3)
  - 风险等级 r_{k,t} 不由密度唯一确定，还受流速与方向、局部拥堵程度、
    出入口通行能力、有效疏散宽度以及天气与活动等外部条件影响。

本模块的 risk_labels() 按上述规则合成标签，其中难以量化的人为修正成分用
可控的噪声项表示，以便在合成数据上检验模型对标注噪声的稳健性。
真实部署时该函数应替换为管理部门的实际标注。

注意：阈值只用于生成标签与式(3) 的 q，不参与模型参数学习。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass
class RegionSpec:
    """区域属性，用于逐区域设定阈值。

    Attributes:
        area: 有效面积 s_k（m^2）。
        capacity: 最大承载量（人），由管理部门核定。
        exit_width: 有效疏散宽度（m）。
    """

    area: float
    capacity: float
    exit_width: float = 4.0


def thresholds_from_capacity(specs: Sequence[RegionSpec],
                             ratios: Sequence[float] = (0.35, 0.60, 0.85),
                             exit_penalty: float = 0.85
                             ) -> torch.Tensor:
    """由承载量与疏散宽度得到逐区域的三个密度阈值，对应式(3) 的 tau_{k,1..3}。

    密度阈值取承载量密度（capacity / area）的固定比例：

        tau_{k,c} = ratios[c-1] * capacity_k / area_k * f(exit_width_k)

    其中 f 为疏散宽度修正：疏散通道越窄，同样密度下的风险越高，故阈值相应
    下调。论文指出"同样是 1 人/m^2，观景平台的疏散难度与开阔广场并不相同"，
    正是这一修正的含义。

    Args:
        specs: 区域属性列表，长度 N。
        ratios: 三个阈值相对承载量密度的比例，须严格递增。
        exit_penalty: 疏散宽度修正系数，宽度低于 4 m 时按比例下调阈值。

    Returns:
        (N, 3) 阈值矩阵，每行严格递增。
    """
    if len(ratios) != 3 or not (ratios[0] < ratios[1] < ratios[2]):
        raise ValueError('ratios 必须是三个严格递增的比例')
    rows = []
    for s in specs:
        cap_density = s.capacity / max(s.area, 1e-9)
        # 宽度修正：以 4 m 为基准，窄于 4 m 时下调阈值
        f = min(1.0, (s.exit_width / 4.0) ** exit_penalty)
        rows.append([cap_density * r * f for r in ratios])
    return _strictly_increasing(torch.tensor(rows))


def thresholds_from_quantiles(rho: torch.Tensor,
                              target_ratios: Sequence[float] = (0.45, 0.30, 0.18, 0.07),
                              exit_correction: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
    """**仅供合成数据使用**：按密度分布的分位数反推三个阈值。

    真实部署时阈值必须由景区管理部门按承载量、疏散宽度与历史事件核定，
    对应论文 3.1 节的说明与 thresholds_from_capacity()。本函数存在的唯一
    理由是：合成数据的密度分布由生成器人为设定，若沿用按承载量折算的固定
    比例，得到的类别分布会严重偏斜（中段两类样本过少），无法用于检验模型的
    风险分级能力。因此合成数据改用分位数定阈值，把类别分布控制在目标比例上。

    使用时必须显式知晓：**这条路径不能用于真实数据**，否则阈值将由标签本身
    决定，构成标签泄漏。

    Args:
        rho: (T, N) 或任意形状的密度张量。
        target_ratios: 四个等级的期望占比，默认取论文 5.1 节的 45/30/18/7。
        exit_correction: (N,) 逐区域的疏散宽度修正因子；给出时对分位数阈值做
                         同样的相对修正，以保留“窄通道阈值更低”的性质。

    Returns:
        (N, 3) 阈值矩阵，列间严格递增。
    """
    if len(target_ratios) != 4 or abs(sum(target_ratios) - 1.0) > 1e-6:
        raise ValueError('target_ratios 必须是四个和为 1 的比例')
    n = rho.shape[-1]
    # 前三类的累积占比即三个分位点
    qs = torch.tensor([target_ratios[0],
                       target_ratios[0] + target_ratios[1],
                       target_ratios[0] + target_ratios[1] + target_ratios[2]])
    # 逐区域的相对尺度：以该区域中位数相对全局中位数的偏移为准，
    # 保留“同样是 1 人/m^2，观景平台与开阔广场阈值不同”的逐区域性质
    reg_med = rho.median(dim=0).values                          # (N,)
    scale = (reg_med / reg_med.median().clamp(min=1e-6)).clamp(0.5, 2.0)
    if exit_correction is not None:
        scale = scale * exit_correction

    # 关键一步：分位点必须在**去掉逐区域尺度之后**的量上取。
    # 若直接对原始 score 取分位再乘 scale，高密度区域的阈值被同比例抬高，
    # 会把该区域的大量样本压回低等级，实际类别分布显著偏离目标比例
    # （实测最高风险类会从 7% 掉到 2% 左右）。把尺度除掉后取分位，再乘回
    # scale，则 P(score_k >= tau_k) = P(score_k / scale_k >= q) 恒等于目标占比，
    # 两个性质同时成立。
    flat_scaled = (rho / scale.view(1, -1)).flatten()
    thr = torch.quantile(flat_scaled, qs)                       # (3,)
    tau = thr.view(1, -1) * scale.view(-1, 1)                   # (N, 3)
    # 保证同一区域内严格递增，避免出现并列导致的等级歧义
    return _strictly_increasing(tau)


def _strictly_increasing(tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐行强制严格递增：后一列不小于前一列加上 eps。"""
    out = tau.clone()
    for c in range(1, out.shape[1]):
        out[:, c] = torch.maximum(out[:, c], out[:, c - 1] + eps)
    return out


def cum_dwell_state(rho: torch.Tensor, tau_low: torch.Tensor,
                    lambda_q: float = 0.8) -> torch.Tensor:
    """式(3) 的拥堵持续状态，与模型内部的实现一致，供标签侧生成使用。

    Args:
        rho:     (T, N) 密度。
        tau_low: (N,) 低风险阈值 tau_{k,1}。
        lambda_q: 遗忘系数。

    Returns:
        (T, N) 拥堵持续状态。
    """
    excess = torch.relu(rho - tau_low.view(1, -1))
    q = torch.zeros(rho.shape[1])
    seq = []
    for t in range(rho.shape[0]):
        q = lambda_q * q + (1.0 - lambda_q) * excess[t]
        seq.append(q)
    return torch.stack(seq, dim=0)


def flow_state(x: torch.Tensor, lambda_v: float = 0.6) -> torch.Tensor:
    """状态的一阶平滑差分，用于“仍在聚集”判据。

    输入可以是人数序列，也可以是密度序列；判据要求与密度阈值可比，
    故 risk_score() 传入的是密度序列，得到的量纲为“人/m^2/步”。
    瞬时差分会放大检测噪声，一阶平滑抑制这一放大，使 v 反映的是持续的
    聚集或疏散趋势而非单步抖动。

    Args:
        x: (T, N) 某一状态量（人数或密度）。
        lambda_v: 平滑系数，越接近 1 越平滑。

    Returns:
        (T, N) 平滑后的一阶差分，首步为 0。
    """
    dn = torch.zeros_like(x)
    dn[1:] = x[1:] - x[:-1]
    v = torch.zeros_like(x)
    prev = torch.zeros(x.shape[1])
    for t in range(x.shape[0]):
        prev = lambda_v * prev + (1 - lambda_v) * dn[t]
        v[t] = prev
    return v


def risk_score(rho: torch.Tensor, ext: torch.Tensor, tau_low: torch.Tensor,
               n_true: Optional[torch.Tensor] = None,
               areas: Optional[torch.Tensor] = None,
               lambda_q: float = 0.8, beta_ext: float = 0.06,
               w_q_ratio: float = 0.25, w_v_ratio: float = 0.10
               ) -> torch.Tensor:
    """按论文 3.1 节的判据计算风险评分，阈值化之前的部分。

    判据（密度为主，拥堵持续与“仍在聚集”上调，外部条件修正）：

        score = rho
              + w_q * q                                    # 持续时间越长风险越高
              + w_v * max(v, 0)                            # 仍在聚集时上调
              + beta_ext * (ext 的线性组合)                # 天气/节假日/活动/出入口

    论文明确指出风险等级不由密度唯一确定，还受流速与方向、局部拥堵程度、
    出入口通行能力、有效疏散宽度以及天气与活动等外部条件影响，上式即为
    这些因素的可操作化。

    关于量纲：四项全部以**密度**为单位（人/m^2）。q 由式(3) 直接对密度
    计算，本就是密度量纲；净流入速率 v 若取人数差分则是“人/步”，与密度
    不可比，故这里对密度序列求差分，得到“人/m^2/步”。这一点必须保持，
    否则人数差分（量级可达数百）会压倒密度项（量级约 1），使评分退化为
    对人数变化率的单调函数，阈值也就失去逐区域的意义。

    Args:
        rho:     (T, N) 密度，式(1)。
        ext:     (T, E) 外部变量，本函数只使用前两列（天气恶劣度与活动强度）。
        tau_low: (N,) 低风险阈值 tau_{k,1}，用作拥堵持续与聚集趋势的量纲尺度。
        n_true:  (T, N) 人数；给出时按 n/areas 现算密度，用于替代 rho。
        areas:   (N,) 区域有效面积；与 n_true 配合使用。
        lambda_q: 式(3) 的遗忘系数。
        beta_ext: 外部变量的影响强度。
        w_q_ratio: 拥堵持续项的权重，以 tau_{k,1} 为单位。
        w_v_ratio: 聚集趋势项的权重，以 tau_{k,1} 为单位。

    Returns:
        (T, N) 风险评分，尚未离散化为等级。
    """
    if rho is None and n_true is not None and areas is not None:
        rho = n_true / areas.view(1, -1)

    q = cum_dwell_state(rho, tau_low, lambda_q)
    v = flow_state(rho)                                          # 密度的一阶差分

    # 拥堵持续与聚集趋势的权重：均以阈值为尺度，保证量纲一致
    w_q = w_q_ratio * tau_low.view(1, -1)
    w_v = w_v_ratio * tau_low.view(1, -1)

    score = rho + w_q * q + w_v * torch.relu(v)
    if ext.shape[-1] >= 2:
        weather = ext[:, 0].view(-1, 1)
        event = ext[:, 1].view(-1, 1)
        score = score + beta_ext * (weather + event) * tau_low.view(1, -1)
    return score


def risk_labels(score: torch.Tensor, tau: torch.Tensor,
                noise_std: float = 0.08, seed: int = 0) -> torch.Tensor:
    """把风险评分按逐区域阈值离散为 1..K 的等级。

        r = 1 + #{c : score >= tau_c}                      # 等级 1..4

    最后叠加一个零均值高斯项表示标注中的边界模糊与人为修正。论文明确指出
    风险等级"标注过程本身也存在边界模糊与人为修正"，故这里如实建模，
    并使用固定随机种子保证可重复。

    Args:
        score: (T, N) 风险评分，由 risk_score() 给出。
        tau:   (N, K-1) 逐区域阈值，严格递增。
        noise_std: 标注噪声标准差，以 tau_{k,1} 为尺度。
        seed:  随机种子。

    Returns:
        (T, N) 整数风险等级，取值 1..K。
    """
    g = torch.Generator().manual_seed(seed)
    scale = tau[:, 0].view(1, -1)
    noisy = score + torch.randn(score.shape, generator=g) * noise_std * scale

    labels = torch.ones_like(noisy, dtype=torch.long)
    for c in range(tau.shape[1]):
        labels = labels + (noisy >= tau[:, c].view(1, -1)).long()
    return labels


def external_features(n_steps: int, n_features: int = 8,
                      steps_per_day: int = 96, seed: int = 0) -> torch.Tensor:
    """生成 8 维外部变量，对应论文 5.1 节的"外部特征 8 维"。

    维度含义（论文未逐一列出，此处按 3.1 节提到的外部条件设定）：
        0 天气恶劣度      1 活动强度        2 节假日标志
        3 出入口开放比例  4 气温归一化      5 降水归一化
        6 时段正弦        7 时段余弦

    前两维是 risk_labels() 使用的维度。真实部署时由气象接口、活动排期表与
    出入口闸机状态直接给出。

    Args:
        n_steps: 时间步数 T（可跨多天）。
        n_features: 特征维数，论文为 8。
        steps_per_day: 每日步数，用于构造日周期项。
        seed: 随机种子。

    Returns:
        (T, n_features)
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n_steps, dtype=torch.float32)
    phase = 2 * torch.pi * (t % steps_per_day) / steps_per_day

    weather = torch.rand(n_steps, generator=g) * 0.5
    event = (torch.rand(n_steps, generator=g) < 0.15).float()
    holiday = (torch.rand(n_steps, generator=g) < 0.10).float()
    gate_open = 0.6 + 0.4 * (torch.cos(phase) * 0.5 + 0.5)
    temp = 0.5 + 0.4 * torch.sin(phase - torch.pi / 2)
    rain = torch.rand(n_steps, generator=g) * 0.3
    cols = [weather, event, holiday, gate_open, temp, rain,
            torch.sin(phase), torch.cos(phase)]
    x = torch.stack(cols[:n_features], dim=-1)
    return x.float()


def class_distribution(labels: torch.Tensor, n_classes: int = 4,
                       one_based: bool = True) -> torch.Tensor:
    """统计各类占比，用于核对论文 5.1 节的 45% / 30% / 18% / 7%。

    Args:
        labels: 风险等级标签。
        n_classes: 类别数 K。
        one_based: True 表示标签按论文正文的 1..K 记（低/中/高/极高）；
                   False 表示按交叉熵要求的 0..K-1 记。

    Returns:
        (K,) 各类占比，顺序为低风险到极高风险。

    Note:
        默认为 1 基是刻意的：若直接用 bincount 处理 1..K 的标签，会得到
        长度 K+1 的计数且首元素恒为 0，看起来像“多出来一类”。
    """
    lab = labels.flatten().long()
    if one_based:
        lab = lab - 1
    if int(lab.min()) < 0 or int(lab.max()) >= n_classes:
        raise ValueError('标签取值范围 [%d, %d] 与 n_classes=%d 不符'
                         % (int(lab.min()), int(lab.max()), n_classes))
    c = torch.bincount(lab, minlength=n_classes).float()
    return c / c.sum().clamp(min=1.0)
