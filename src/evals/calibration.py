# -*- coding: utf-8 -*-
"""概率校准评价，对应论文表 4 与 5.5 节。

    Brier 分数   多分类形式，sum_c (p_c - y_c)^2 的均值
    ECE          期望校准误差，等宽分箱，论文取 15 箱

论文表 4 同时报告“最大置信度偏差”，即各分箱 |acc(bin) - conf(bin)| 的最大值。
数学上最大值恒不小于按样本数加权的平均值，因此 MCE >= ECE 必须成立；本模块
在返回前会校验该关系，作为一次数值自检。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def brier_score(probs: torch.Tensor, labels: torch.Tensor,
                n_classes: int = None) -> torch.Tensor:
    """多分类 Brier 分数。

    Args:
        probs:  (..., K) 概率分布。
        labels: (...) 整数标签。
    """
    k = n_classes or probs.shape[-1]
    onehot = F.one_hot(labels.long(), k).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(dim=-1).mean()


def ece(probs: torch.Tensor, labels: torch.Tensor,
        n_bins: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
    """期望校准误差与最大置信度偏差。

    按预测置信度等宽分箱。对多分类问题，置信度取 max_c p_c，正确性取
    预测是否命中真实类别，这与论文表 4 的口径一致（表中“ECE”与“最大置信度
    偏差”列同时给出）。

    Args:
        probs:  (..., K)
        labels: (...)
        n_bins: 论文取 15。

    Returns:
        (ece_value, mce_value)。mce 为各非空分箱 |acc - conf| 的最大值，
        因而恒满足 mce >= ece。
    """
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels.long()).to(probs.dtype)
    conf = conf.flatten()
    correct = correct.flatten()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    total = conf.numel()
    if total == 0:
        return probs.new_zeros(()), probs.new_zeros(())

    ece_val = probs.new_zeros(())
    mce_val = probs.new_zeros(())
    max_dev = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # 最后一箱右闭，避免置信度恰好为 1 的样本被漏掉
        sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        avg_conf = conf[sel].mean()
        avg_acc = correct[sel].mean()
        dev = (avg_acc - avg_conf).abs()
        ece_val = ece_val + dev * sel.sum() / total
        max_dev = max(max_dev, float(dev))
    mce_val = probs.new_tensor(max_dev)

    # 数值自检：最大偏差恒不小于加权平均值
    assert float(mce_val) + 1e-9 >= float(ece_val), (
        'MCE < ECE 在数学上不可能，说明分箱逻辑有误')
    return ece_val, mce_val


def temperature_scale(logits: torch.Tensor, labels: torch.Tensor,
                      max_iter: int = 50) -> torch.Tensor:
    """用温度缩放做后处理校准，作为读出头本身校准效果的对照。

    论文的做法是把单调性约束直接施加在读出层上（3.2 节），温度缩放是常见的
    事后校准基线，此处提供以支持对照实验。

    Returns:
        标量温度 T，使得 softmax(logits / T) 的负对数似然最小。
    """
    logits = logits.detach()
    labels = labels.long()
    log_t = torch.zeros((), device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_t).clamp(min=1e-3).detach()


def reliability_curve(probs: torch.Tensor, labels: torch.Tensor,
                      n_bins: int = 15):
    """返回可靠性图所需的 (bin_center, avg_confidence, avg_accuracy, count)。"""
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels.long()).to(probs.dtype)
    conf, correct = conf.flatten(), correct.flatten()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        centers.append(float((lo + hi) / 2))
        accs.append(float(correct[sel].mean()))
        confs.append(float(conf[sel].mean()))
        counts.append(int(sel.sum()))
    return centers, confs, accs, counts


def calibration_report(probs: torch.Tensor, labels: torch.Tensor,
                       n_bins: int = 15) -> Dict[str, float]:
    """论文表 4 的一行：Brier、ECE、最大置信度偏差。

    键 'max_confidence_deviation' 与论文表 4 的列名一致；同时给出等价的
    短名 'mce'，便于脚本按统一字段读取。
    """
    e, m = ece(probs, labels, n_bins)
    return {'brier': float(brier_score(probs, labels)),
            'ece': float(e),
            'max_confidence_deviation': float(m),
            'mce': float(m)}
