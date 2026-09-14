# -*- coding: utf-8 -*-
"""合成一个与 Scenic-MM 结构一致的景区多模态场景，对应论文 5.1 节。

本模块按论文 5.1 节描述的数据结构生成一份可运行的场景：区域与承载量、
摄像头覆盖与缺口、信令空间混合与时间聚合、锚点部署、外部协变量，以及
由承载量核定的风险阈值与由此得到的风险标签。生成的数据在张量形状、
量纲与观测模型上与论文一致，可用于端到端地运行、检查与调试本仓库的
全部代码路径。

生成的数据在结构上严格对齐论文 5.1 节的 Scenic-MM 描述：
    18 个子区域，12 路摄像头覆盖其中 12 个，另 6 个为缺口区
    7 个基站扇区，15 min 聚合
    序列长度 96（15 min 步长，对应 24 h），连续 60 天，共 5760 个时间步
    视频每 60 s 一帧，12 路 x 60 天 x 1440 帧/天 = 103.68 万帧
    外部特征 8 维
    锚点 3 个出入口闸机 + 人工抽样每日两次，共 360 次
    风险标签四类，低/中/高/极高
    训练 42 天 / 验证 9 天 / 测试 9 天，区间不重叠

人数序列用“日周期 + 活动脉冲 + 区域间流动”的组合生成，不是简单随机数，
以便让风险分级成为一个有结构的任务而不是噪声拟合。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from . import anchors as anc
from . import graph as gph
from . import labels as lab
from . import signaling as sig
from . import video as vid


# 论文 5.1 节列出的 18 个功能子区域，后 6 个为缺口区
REGION_NAMES: List[str] = [
    '入口广场', '主游览道北段', '主游览道南段', '观景平台', '中心服务区',
    '文创街区', '东侧停车场', '西侧停车场', '生态湿地区', '亲水栈道',
    '山门集散区', '游客中心', '北门集散区', '索道上下站', '演艺广场',
    '后山步道', '临时展区', '应急疏散通道',
]
GAP_REGION_NAMES: List[str] = [
    '后山步道', '东侧停车场', '生态湿地区', '文创街区', '北门集散区', '临时展区',
]


@dataclass
class ScenarioConfig:
    """场景规模，默认值全部取自论文 5.1 节的 Scenic-MM 描述。"""

    n_regions: int = 18
    n_cameras: int = 12
    n_sectors: int = 7
    n_anchors_gate: int = 3
    n_anchors_manual: int = 2
    days: int = 60
    steps_per_day: int = 96                 # 15 min 步长 -> 96 步/天
    n_ext: int = 8
    n_classes: int = 4
    base_occupancy: float = 0.55            # 平均承载率
    night_floor: float = 0.30               # 夜间人数相对日间峰值的比例
    target_ratios: tuple = (0.45, 0.30, 0.18, 0.07)
    """风险四类的目标占比，取自论文 5.1 节的 45% / 30% / 18% / 7%。"""
    seed: int = 20260913
    """随机种子。固定种子保证数据可重复生成。"""


@dataclass
class Scenario:
    """合成场景的全部张量与结构矩阵。"""

    cfg: ScenarioConfig
    n_true: torch.Tensor            # (T_total, N) 真实人数
    rho_true: torch.Tensor          # (T_total, N) 真实密度
    pi_true: torch.Tensor           # (T_total, N) 真实渗透率
    risk: torch.Tensor              # (T_total, N) 风险等级 1..4
    ext: torch.Tensor               # (T_total, E) 外部变量
    y_vis: torch.Tensor             # (T_total, M) 视觉观测
    m_vis: torch.Tensor             # (T_total, M) 有效观测权重
    o_vis: torch.Tensor             # (T_total, M) 遮挡/质量指标
    y_sig: torch.Tensor             # (T_total, J) 信令观测（按基本步窗口）
    y_anc: torch.Tensor             # (T_total, A) 锚点人数观测
    anc_mask: torch.Tensor          # (T_total, A) 锚点可用性掩码
    coverage: torch.Tensor          # (M, N) 覆盖矩阵 H
    mixing: torch.Tensor            # (L, N) 空间混合矩阵 B
    time_agg: torch.Tensor          # (J, T) 时间聚合矩阵 W
    anchor_sel: torch.Tensor        # (A, N) 锚点选择矩阵 P
    areas: torch.Tensor             # (N,) 有效面积
    adjacency: torch.Tensor         # (N, N) 功能区邻接矩阵
    laplacian: torch.Tensor         # (N, N) 已缩放的 L~
    tau: torch.Tensor               # (N, 3) 逐区域阈值（生成标签所用）
    tau_capacity: torch.Tensor      # (N, 3) 由承载量核定的阈值（真实部署口径）
    covered_mask: torch.Tensor      # (N,) 覆盖区掩码
    gap_mask: torch.Tensor          # (N,) 缺口区掩码
    region_names: List[str] = field(default_factory=list)


def _region_specs(cfg: ScenarioConfig, g: torch.Generator) -> List[lab.RegionSpec]:
    """按区域类型给出面积、承载量与疏散宽度。

    览道、步道类区域面积小、疏散宽度窄；广场、停车场类区域面积大、通道宽。
    """
    wide = {'入口广场', '主游览道北段', '主游览道南段', '中心服务区',
            '东侧停车场', '西侧停车场', '北门集散区', '山门集散区',
            '演艺广场', '临时展区', '应急疏散通道'}
    specs = []
    for name in REGION_NAMES[:cfg.n_regions]:
        area = float(600 + torch.rand((), generator=g).item() * 2400)
        if name in wide:
            area *= 2.2
        density_cap = 2.5 + torch.rand((), generator=g).item() * 1.5   # 人/m^2
        width = 2.0 + torch.rand((), generator=g).item() * 4.0
        specs.append(lab.RegionSpec(area=area,
                                    capacity=density_cap * area,
                                    exit_width=width))
    return specs


def _adjacency(cfg: ScenarioConfig, n_regions: int) -> torch.Tensor:
    """构造功能区邻接关系：按区域编号构成一条主环线，再补若干横向连接。

    真实部署时这里替换为景区 GIS 的连通关系，见 data/graph.py 的说明。
    """
    edges = []
    for i in range(n_regions):
        edges.append((i, (i + 1) % n_regions))                  # 主环线
    for i in range(0, n_regions - 3, 3):
        edges.append((i, i + 3))                                # 横向连接
    weights = [1.0] * len(edges)
    return gph.adjacency_from_edges(n_regions, edges, weights)


def _coverage(cfg: ScenarioConfig, gap_mask: torch.Tensor,
              n_regions: int, g: torch.Generator) -> torch.Tensor:
    """构造覆盖矩阵 H：12 路摄像头覆盖 12 个区域，缺口区覆盖为 0。

    论文 5.1 节：部署 12 路摄像头覆盖其中 12 个，另 6 个为缺口区，
    摄像头有效覆盖面积占比均值 41.6%。本函数让覆盖区的覆盖比例在
    [0.25, 0.95] 内随机，缺口区严格为 0，以同时满足命题 1 的前提。
    """
    h = torch.zeros(cfg.n_cameras, n_regions)
    covered = (~gap_mask).nonzero().flatten()
    for i in range(cfg.n_cameras):
        k = int(covered[i % len(covered)])
        # 单路摄像头只覆盖所在区域的一部分，可能在多个区域有部分覆盖
        h[i, k] = 0.25 + torch.rand((), generator=g).item() * 0.70
        # 相邻区域存在部分视野重叠
        j = (k + 1) % n_regions
        if not gap_mask[j]:
            h[i, j] = torch.rand((), generator=g).item() * 0.25
    return h.clamp(0.0, 1.0)


def _mixing(cfg: ScenarioConfig, n_regions: int, g: torch.Generator
            ) -> torch.Tensor:
    """构造空间混合矩阵 B：每个区域的设备按面积比例分摊到各扇区。

    论文 5.1 节：信令来自 7 个基站扇区。一个扇区可能覆盖多个子区域，
    故 B 在列上归一化（每个区域的设备分布到覆盖它的扇区上）。
    """
    b = torch.rand(cfg.n_sectors, n_regions, generator=g) * 0.4
    # 让每个区域主要落在 1-2 个扇区上，模拟扇区覆盖不跨越太多区域
    for k in range(n_regions):
        primary = k % cfg.n_sectors
        b[primary, k] = 1.0
    b = b / b.sum(dim=0, keepdim=True).clamp(min=1e-9)
    return b


def _population(cfg: ScenarioConfig, specs: List[lab.RegionSpec],
                g: torch.Generator) -> tuple:
    """生成真实人数序列 n_{k,t}。

    组合三个成分：
        日周期    双峰（上午与下午各一个高峰）
        活动脉冲  节假日/演艺活动触发的短时聚集
        区域间流动 由主环线相邻区域的一阶滞后项引入
    """
    t_total = cfg.days * cfg.steps_per_day
    n = cfg.n_regions
    t = torch.arange(t_total, dtype=torch.float32)
    phase = 2 * torch.pi * (t % cfg.steps_per_day) / cfg.steps_per_day

    # 日周期双峰：上午 10 点与下午 15 点附近
    daily = (0.45 * torch.exp(-((phase - 2 * math.pi * 10 / 24) ** 2) / 0.35)
             + 0.55 * torch.exp(-((phase - 2 * math.pi * 15 / 24) ** 2) / 0.5))
    daily = daily / daily.max().clamp(min=1e-6)
    # 夜间底噪：景区闭园后仍有值守、住宿与过境人员，人数不会降到零。
    # 若不加底噪，密度在夜间归零，风险等级会在最低与最高之间两极分化，
    # 中段两级几乎没有样本，类别分布退化为 U 形。
    daily = cfg.night_floor + (1.0 - cfg.night_floor) * daily

    # 周内节律：周末与节假日更高
    day_idx = (t // cfg.steps_per_day).long()
    weekend = ((day_idx % 7) >= 5).float()
    holiday = (torch.rand(cfg.days, generator=g) < 0.12).float()[day_idx]

    # 活动脉冲：少量时段出现短时聚集。脉冲中心需避开首尾各一天的边界，
    # 故时间跨度至少要有两天；不足两天时无法安放脉冲，明确报错而不是
    # 让 torch.randint 抛出下界等于上界的含义不明的异常。
    if t_total < 2 * cfg.steps_per_day:
        raise ValueError(
            '场景时间跨度 %d 步不足两天（steps_per_day=%d），无法生成活动脉冲；'
            '请把 days 设为 2 以上' % (t_total, cfg.steps_per_day))
    pulse = torch.zeros(t_total)
    n_pulses = max(4, t_total // (cfg.steps_per_day * 3))
    for _ in range(n_pulses):
        center = int(torch.randint(cfg.steps_per_day, t_total - cfg.steps_per_day,
                                   (1,), generator=g).item())
        width = int(torch.randint(2, 8, (1,), generator=g).item())
        amp = 0.2 + torch.rand((), generator=g).item() * 0.5
        lo, hi = max(0, center - width), min(t_total, center + width)
        pulse[lo:hi] += amp * torch.exp(
            -((torch.arange(lo, hi, dtype=torch.float32) - center) ** 2) / (2 * width))

    base = torch.tensor([s.capacity for s in specs], dtype=torch.float32)
    # 各区域的相对客流权重：广场与主道高，步道与展区低
    weight = torch.ones(n)
    for k, name in enumerate(REGION_NAMES[:n]):
        if name in ('入口广场', '主游览道北段', '中心服务区', '演艺广场'):
            weight[k] = 1.4
        elif name in ('后山步道', '应急疏散通道', '生态湿地区'):
            weight[k] = 0.55

    drive = (daily * (1.0 + 0.22 * weekend + 0.35 * holiday) + pulse)
    n_true = (base.view(1, -1) * weight.view(1, -1)
              * cfg.base_occupancy * drive.view(-1, 1))

    # 区域间流动：相邻区域的人流以一定滞后渗入
    adj = _adjacency(cfg, n)
    adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-9)
    flow = torch.zeros_like(n_true)
    for step in range(1, t_total):
        flow[step] = 0.82 * flow[step - 1] + 0.18 * (adj @ n_true[step - 1])
    n_true = 0.86 * n_true + 0.14 * flow

    # 观测噪声与个体扰动
    n_true = n_true * (1.0 + 0.05 * torch.randn(n_true.shape, generator=g))
    return n_true.clamp(min=0.0), adj


def _penetration(cfg: ScenarioConfig, g: torch.Generator) -> torch.Tensor:
    """真实渗透率：低秩时变 + 区域基础值 + 扰动，与式(10) 的形式一致。"""
    t_total = cfg.days * cfg.steps_per_day
    rank = 4
    u = torch.randn(cfg.n_regions, rank, generator=g) * 0.15
    v = torch.randn(t_total, rank, generator=g) * 0.15
    b = torch.randn(cfg.n_regions, generator=g) * 0.10
    delta = torch.randn(t_total, cfg.n_regions, generator=g) * 0.05
    eta = v @ u.t() + b.view(1, -1) + delta
    pi = 0.05 + 0.80 * torch.sigmoid(eta)
    return pi.clamp(0.05, 0.85)


def build_scenario(cfg: Optional[ScenarioConfig] = None,
                   anchor_sparsity: float = 1.0) -> Scenario:
    """生成完整的合成场景。

    Args:
        cfg: 场景配置；None 时用论文 5.1 节的默认规模。
        anchor_sparsity: 锚点稀疏化档位，取 1.0 / 0.5 / 0.25 / 0.0，
                         对应论文的 100% / 50% / 25% / 0% 四档实验。

    Returns:
        Scenario。
    """
    cfg = cfg or ScenarioConfig()
    g = torch.Generator().manual_seed(cfg.seed)
    t_total = cfg.days * cfg.steps_per_day
    n = cfg.n_regions

    # ---- 结构与标签侧 ----
    specs = _region_specs(cfg, g)
    areas = torch.tensor([s.area for s in specs])
    adj = _adjacency(cfg, n)
    laplacian = gph.build_laplacian(adj, normalized=True)

    n_true, _ = _population(cfg, specs, g)
    rho_true = n_true / areas.view(1, -1)
    pi_true = _penetration(cfg, g)

    # 阈值的取法见 labels.thresholds_from_quantiles 的说明：合成数据用分位数
    # 定阈值以控制类别分布，真实数据必须用 thresholds_from_capacity 由管理
    # 部门核定。此处把两种口径都算出来，前者用于生成标签，后者一并随数据保存，
    # 便于核对两者的差异（真实部署时以 capacity 口径为准）。
    tau_capacity = lab.thresholds_from_capacity(specs)

    gap_names = set(GAP_REGION_NAMES)
    gap_mask = torch.tensor([name in gap_names
                             for name in REGION_NAMES[:n]], dtype=torch.bool)
    covered_mask = ~gap_mask

    coverage = _coverage(cfg, gap_mask, n, g)
    # 覆盖矩阵与缺口区划分互为校验：缺口区必须无任何摄像头覆盖
    assert not coverage[:, gap_mask].gt(0).any(), (
        '缺口区的覆盖比例必须为 0，否则命题 1 的前提不成立')

    ext = lab.external_features(t_total, cfg.n_ext, cfg.steps_per_day, cfg.seed)
    # 阈值必须定在**评分**的分布上而非密度的分布上：等级判据比较的是
    # score = rho + 拥堵持续项 + 聚集趋势项 + 外部条件项，若按 rho 的分位数
    # 取阈值，附加项会把大量样本推过最高阈值，最高风险类会显著超出目标占比。
    score = lab.risk_score(rho_true, ext, tau_capacity[:, 0], lambda_q=0.8)
    tau = lab.thresholds_from_quantiles(score, cfg.target_ratios)
    risk = lab.risk_labels(score, tau, seed=cfg.seed)

    # ---- 观测侧 ----
    mixing = _mixing(cfg, n, g)
    time_agg = sig.time_aggregation_matrix(t_total, step_minutes=15,
                                           window_minutes=15)
    n_windows = time_agg.shape[0]

    # 夜间视频信噪比下降：用日周期调制亮度与清晰度，进而得到 m^vis 与 o
    phase = 2 * torch.pi * (torch.arange(t_total) % cfg.steps_per_day
                            ) / cfg.steps_per_day
    night = (torch.cos(phase) + 1) / 2                          # 0=白天 1=夜间
    quality = (0.65 - 0.35 * night).unsqueeze(1).expand(-1, cfg.n_cameras)
    m_vis, o_vis = vid.quality_weights(quality * 255.0, quality * 200.0)
    y_vis, _ = vid.simulate(n_true, coverage, m_vis=m_vis, o_vis=o_vis,
                            noise=vid.VisualNoise(seed=cfg.seed))

    y_sig = sig.simulate(n_true, pi_true, mixing, time_agg,
                         noise=sig.SignalNoise(seed=cfg.seed))

    # ---- 锚点 ----
    layout = anc.AnchorLayout(gate_regions=(0, 11, 5),
                              manual_regions=(3, 8),
                              manual_times_per_day=2,
                              steps_per_day=cfg.steps_per_day)
    anchor_sel = anc.build_anchor_matrix(layout, n)
    n_gate, n_manual = len(layout.gate_regions), len(layout.manual_regions)
    mask = anc.anchor_mask(layout, n_gate, n_manual, t_total)
    # 闸机锚点是常态化部署，不参与稀释；稀疏化针对的是人工抽样锚点的密度
    mask = anc.sparsify(mask, anchor_sparsity, seed=cfg.seed, n_gates=n_gate)
    # 锚点观测值 = 该点位对应区域的真实人数 + 计量噪声
    picking = (anchor_sel @ n_true.t()).t()                     # (T, A)
    y_anc = (picking * mask
             + torch.randn_like(picking) * 1.0 * mask).clamp(min=0.0)

    return Scenario(
        cfg=cfg, n_true=n_true, rho_true=rho_true, pi_true=pi_true,
        risk=risk, ext=ext, y_vis=y_vis, m_vis=m_vis, o_vis=o_vis,
        y_sig=y_sig, y_anc=y_anc, anc_mask=mask,
        coverage=coverage, mixing=mixing, time_agg=time_agg,
        anchor_sel=anchor_sel, areas=areas, adjacency=adj,
        laplacian=laplacian, tau=tau, tau_capacity=tau_capacity,
        covered_mask=covered_mask, gap_mask=gap_mask,
        region_names=REGION_NAMES[:n])


def describe(sc: Scenario) -> Dict[str, object]:
    """返回一批描述性统计，用于核对合成数据是否符合论文 5.1 节的描述。"""
    cfg = sc.cfg
    t_total = sc.y_vis.shape[0]
    dist = lab.class_distribution(sc.risk, cfg.n_classes)
    return {
        '总时间步': int(t_total),
        '天数': cfg.days,
        '区域数': cfg.n_regions,
        '摄像头数': cfg.n_cameras,
        '缺口区数': int(sc.gap_mask.sum()),
        '信令扇区数': cfg.n_sectors,
        '信令窗口数': int(sc.time_agg.shape[0]),
        '视频帧数_估算': vid.frames_per_camera(cfg.days, cfg.n_cameras),
        '锚点可用次数': int(sc.anc_mask.sum()),
        '锚点占比': float(sc.anc_mask.sum() / (t_total * sc.anc_mask.shape[1])),
        # 论文 5.1 节的“摄像头有效覆盖面积占比均值 41.6%”对应的是：
        # 先按区域把多路摄像头的覆盖比例合并（同一像素被多路拍到不重复计），
        # 再对全部 N 个区域取均值。若直接对 (M, N) 矩阵取均值，会把大量
        # “该摄像头拍不到该区域”的零一并平均进去，得到偏小一个量级的数。
        '有效覆盖面积占比均值': float(sc.coverage.sum(dim=0).clamp(max=1.0).mean()),
        '覆盖区覆盖率均值': float(
            sc.coverage.sum(dim=0).clamp(max=1.0)[sc.covered_mask].mean()),
        '缺口区覆盖比例最大值': float(sc.coverage[:, sc.gap_mask].max())
        if bool(sc.gap_mask.any()) else 0.0,
        '风险类分布': [round(float(v), 4) for v in dist],
        '阈值_中位数_分位数口径': [round(float(v), 3) for v in
                                   sc.tau.median(dim=0).values],
        '阈值_中位数_承载量口径': [round(float(v), 3) for v in
                                   sc.tau_capacity.median(dim=0).values],
        '人数均值': float(sc.n_true.mean()),
        '密度均值': float(sc.rho_true.mean()),
        '渗透率均值': float(sc.pi_true.mean()),
    }
