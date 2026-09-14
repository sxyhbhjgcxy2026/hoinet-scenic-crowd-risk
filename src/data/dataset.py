# -*- coding: utf-8 -*-
"""把场景张量切成模型可用的序列样本，对应论文 5.4 节的训练设置。

论文 5.2 节的数据划分：
    数据按时间顺序划分为训练集 42 天、验证集 9 天、测试集 9 天，
    三个区间不重叠，以避免同一时段的信息泄漏到评估阶段。

本模块据此把连续 60 天的场景张量切成三段，再在每段内滑窗成长度为
T=96（一天）的序列。窗口在同一划分区间内滑动，不跨越区间边界。

张量约定
--------
所有张量在窗口维之后的第一维是时间，第二维是空间：
    y_vis   (B, T, M)    视觉观测
    m_vis   (B, T, M)    视觉有效观测权重
    o_vis   (B, T, M)    遮挡/质量指标
    y_sig   (B, T, J)    信令观测（已按基本步窗口摊平）
    y_anc   (B, T, A)    锚点观测
    a_mask  (B, T, A)    锚点可用性
    ext     (B, T, E)    外部特征
    n_true  (B, T, N)    真值人数，仅用于评估
    risk    (B, T, N)    真值风险等级，仅用于评估
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .synthesize import Scenario


# 论文 5.2 节的划分比例（天）
TRAIN_DAYS = 42
VAL_DAYS = 9
TEST_DAYS = 9


@dataclass
class SplitIndex:
    """一个划分区间的天数范围。"""

    name: str
    start_day: int
    end_day: int

    def steps(self, steps_per_day: int) -> Tuple[int, int]:
        return self.start_day * steps_per_day, self.end_day * steps_per_day


def default_splits(days: int = 60, train: int = TRAIN_DAYS,
                   val: int = VAL_DAYS, test: int = TEST_DAYS):
    """给出三个不重叠的时间划分区间，按论文 5.2 节的 70% / 15% / 15% 分配。

    论文 5.2 节的 42 / 9 / 9 天正是在 days=60 时该比例的结果，因此这里按
    比例推算而非硬编码天数，便于用较短的天数快速跑通流程（例如 days=6 时
    得到 4 / 1 / 1 天）。三个区间首尾相接、互不重叠。

    Args:
        days: 总天数。
        train / val / test: 显式指定天数的覆盖值；三者之和必须等于 days。
            默认值 42/9/9 只在 days=60 时生效，其余情形按比例推算。
    """
    if train + val + test == days:
        return [SplitIndex('train', 0, train),
                SplitIndex('val', train, train + val),
                SplitIndex('test', train + val, days)]

    if days < 3:
        raise ValueError('总天数 %d 不足以划分出训练/验证/测试三段' % days)
    n_train = max(1, int(round(days * 0.70)))
    n_val = max(1, int(round(days * 0.15)))
    n_test = days - n_train - n_val
    if n_test < 1:                                     # 天数过少时保证三段非空
        n_train, n_val, n_test = days - 2, 1, 1
    return [SplitIndex('train', 0, n_train),
            SplitIndex('val', n_train, n_train + n_val),
            SplitIndex('test', n_train + n_val, days)]


class ScenicWindowDataset(Dataset):
    """把某个划分区间内的场景切成 (T, ...) 序列样本。

    Args:
        sc: 合成或真实场景。
        split: 划分区间。
        seq_len: 序列长度 T，论文为 96。
        stride: 滑窗步长，默认为 1（逐基本步滑窗）。
        with_target: True 时同时返回人数与风险真值，评估阶段需要。
    """

    def __init__(self, sc: Scenario, split: SplitIndex,
                 seq_len: Optional[int] = None, stride: int = 1,
                 with_target: bool = True):
        self.sc = sc
        self.split = split
        self.seq_len = int(seq_len or sc.cfg.steps_per_day)
        self.stride = int(stride)
        self.with_target = with_target

        lo, hi = split.steps(sc.cfg.steps_per_day)
        span = hi - lo
        if span < self.seq_len:
            raise ValueError('区间 %s 只有 %d 步，短于序列长度 %d'
                             % (split.name, span, self.seq_len))
        self.start = lo
        self.n_samples = (span - self.seq_len) // self.stride + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not 0 <= idx < self.n_samples:
            raise IndexError('样本下标 %d 越界，共 %d 个样本'
                             % (idx, self.n_samples))
        s = self.start + idx * self.stride
        e = s + self.seq_len
        sc = self.sc

        # 信令观测在 (J, L) 粒度上，先按时间聚合矩阵回摊到基本步上。
        # 无论 W 是否退化为单位阵，这一步都给出定义良好的 (T, L) 序列。
        sig = torch.einsum('jt,jl->tl', sc.time_agg, sc.y_sig)[s:e]

        item = {
            'y_vis': sc.y_vis[s:e],
            'm_vis': sc.m_vis[s:e],
            'o_vis': sc.o_vis[s:e],
            'y_sig': sig,
            'y_anc': sc.y_anc[s:e],
            'a_mask': sc.anc_mask[s:e],
            'ext': sc.ext[s:e],
        }
        if self.with_target:
            item['n_true'] = sc.n_true[s:e]
            # 数据集与论文正文按 1..K 记风险等级（低/中/高/极高），
            # 而交叉熵要求类别下标从 0 起，故在此统一减去 1。
            # 评估阶段用回 1..K 时再加回来（见 src/evals/metrics.py 的说明）。
            item['risk'] = sc.risk[s:e] - 1
            item['rho_true'] = sc.rho_true[s:e]
            # 真实渗透率，供 6.6 节锚点稀疏化实验报告式(10) 的估计相对误差
            item['pi_true'] = sc.pi_true[s:e]
        return item


class ScenarioTensors:
    """把整个场景一次性搬到设备上，避免每个 batch 重复搬运静态结构矩阵。

    结构矩阵（H、B、W、P、L）在整个场景内共享，属于模型的一部分而非样本的
    一部分，因此由本类持有并在训练设备上常驻。

    Args:
        sc: 场景。
        device: 目标设备。
    """

    _STRUCT_KEYS = ('coverage', 'mixing', 'time_agg', 'anchor_sel', 'laplacian')

    def __init__(self, sc: Scenario, device: torch.device):
        self.sc = sc
        self.device = device
        for k in self._STRUCT_KEYS:
            setattr(self, k, getattr(sc, k).to(device).float())
        self.areas = sc.areas.to(device).float()
        self.adjacency = sc.adjacency.to(device).float()
        self.gap_mask = sc.gap_mask.to(device)
        self.covered_mask = sc.covered_mask.to(device)
        self.tau = sc.tau.to(device).float()
        self.n_cameras = sc.coverage.shape[0]
        self.n_regions = sc.coverage.shape[1]

    def structure(self) -> Dict[str, torch.Tensor]:
        """返回模型 forward 需要的结构参数。"""
        return {
            'H': self.coverage,
            'B': self.mixing,
            'W': self.time_agg,
            'P': self.anchor_sel,
            'L': self.laplacian,
            'areas': self.areas,
        }


def collate(samples) -> Dict[str, torch.Tensor]:
    """把若干样本叠成 batch，结构与默认 collate 相同，这里显式写出以便阅读。"""
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}


def make_loaders(sc: Scenario, device: torch.device, seq_len: int = 96,
                 batch_size: int = 8, stride: int = 1, num_workers: int = 0,
                 splits: Optional[list] = None):
    """构建训练/验证/测试三个 DataLoader。

    Args:
        sc: 场景。
        device: 设备（仅用于返回的 ScenarioTensors）。
        seq_len: 序列长度 T。
        batch_size: 批大小，论文为 8。
        stride: 滑窗步长。
        num_workers: DataLoader 工作进程数。
        splits: 自定义划分；None 时按论文 5.2 节的 42/9/9 天划分。

    Returns:
        (loaders, tensors, splits)，其中 loaders 是 {'train','val','test'}
        到 DataLoader 的字典。
    """
    from torch.utils.data import DataLoader

    splits = splits or default_splits(sc.cfg.days)
    tensors = ScenarioTensors(sc, device)
    loaders = {}
    for sp in splits:
        ds = ScenicWindowDataset(sc, sp, seq_len=seq_len, stride=stride)
        loaders[sp.name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(sp.name == 'train'),
            num_workers=num_workers, collate_fn=collate, drop_last=False)
    return loaders, tensors, splits
