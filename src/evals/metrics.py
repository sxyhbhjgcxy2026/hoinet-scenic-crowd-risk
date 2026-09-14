# -*- coding: utf-8 -*-
"""评价指标，对应论文 5.5 节。

三个层面：
    状态反演  MAE / RMSE / SMAPE，并分覆盖区与缺口区报告
    风险读出  准确率 / 宏平均 F1 / 加权平均 F1 / 极高风险召回率
    概率校准  Brier 分数 / 期望校准误差 ECE

关于 SMAPE：论文用 SMAPE 替代 MAPE，以避免清晨时段真实人数接近零时发散。
本模块的 SMAPE 采用对称形式 2|y-yhat| / (|y| + |yhat| + eps)，取值域 [0, 2]。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def _flat(x: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is None:
        return x.flatten(), None
    m = mask.flatten().bool()
    return x.flatten()[m], m


# ------------------------------------------------------------------ 状态反演
def mae(pred: torch.Tensor, true: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    p, _ = _flat(pred, mask)
    t, _ = _flat(true, mask)
    if p.numel() == 0:
        return pred.new_tensor(float('nan'))
    return (p - t).abs().mean()


def rmse(pred: torch.Tensor, true: torch.Tensor,
         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    p, _ = _flat(pred, mask)
    t, _ = _flat(true, mask)
    if p.numel() == 0:
        return pred.new_tensor(float('nan'))
    return ((p - t) ** 2).mean().sqrt()


def smape(pred: torch.Tensor, true: torch.Tensor,
          mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """对称平均绝对百分比误差，返回小数（0.086 表示 8.6%）。"""
    p, _ = _flat(pred, mask)
    t, _ = _flat(true, mask)
    if p.numel() == 0:
        return pred.new_tensor(float('nan'))
    denom = (p.abs() + t.abs()).clamp(min=eps)
    return (2.0 * (p - t).abs() / denom).mean()


def _group(n_hat: torch.Tensor, n_true: torch.Tensor,
           mask: Optional[torch.Tensor]) -> Dict[str, float]:
    """一个分组（整体/覆盖区/缺口区）上的四项误差。"""
    if mask is None:
        return {'mae': float(mae(n_hat, n_true)),
                'rmse': float(rmse(n_hat, n_true)),
                'smape': float(smape(n_hat, n_true)),
                'count': int(n_hat.numel())}
    m = _broadcast(mask, n_hat)
    p, flat = _flat(n_hat, m)
    return {'mae': float(mae(n_hat, n_true, m)),
            'rmse': float(rmse(n_hat, n_true, m)),
            'smape': float(smape(n_hat, n_true, m)),
            'count': int(flat.sum())}


def state_metrics(n_hat: torch.Tensor, n_true: torch.Tensor,
                  rho_hat: Optional[torch.Tensor] = None,
                  rho_true: Optional[torch.Tensor] = None,
                  gap: Optional[torch.Tensor] = None,
                  covered: Optional[torch.Tensor] = None,
                  areas: Optional[torch.Tensor] = None) -> Dict[str, object]:
    """论文表 2 的口径：整体、覆盖区、缺口区分别报告人数与密度误差。

    Args:
        n_hat / n_true: (B, T, N) 反演与真实人数。
        rho_hat / rho_true: (B, T, N) 反演与真实密度；给出时额外报告密度 MAE。
        gap / covered: (N,) 区域分组掩码。未显式给出 covered 时由 gap 取反。
        areas: (N,) 区域有效面积；给出且 rho 未给出时，由 n / areas 现算密度。

    Returns:
        嵌套字典，形如

            {'overall':  {'mae':..., 'rmse':..., 'smape':..., 'count':...},
             'covered':  {...},
             'gap':      {...},
             'gap_over_covered': 1.35,
             'density_mae': 0.042}

        其中 gap 与 covered 仅在给出对应掩码时出现。缺口区误差与覆盖区误差
        之比 gap_over_covered 是论文 5.5 节重点讨论的量。
    """
    out: Dict[str, object] = {'overall': _group(n_hat, n_true, None)}

    if covered is None and gap is not None:
        covered = ~gap.bool() if gap.dim() == 1 else ~gap.bool()
    if covered is not None:
        out['covered'] = _group(n_hat, n_true, covered)
    if gap is not None:
        out['gap'] = _group(n_hat, n_true, gap)

    if 'covered' in out and 'gap' in out:
        c = out['covered']['mae']
        out['gap_over_covered'] = float(out['gap']['mae'] / c) if c > 0 else float('nan')

    # 密度误差：优先用显式给出的 rho，否则由人数除以面积现算
    if rho_hat is None and rho_true is None and areas is not None:
        s = areas.view(1, 1, -1)
        rho_hat, rho_true = n_hat / s, n_true / s
    if rho_hat is not None and rho_true is not None:
        out['density_mae'] = float(mae(rho_hat, rho_true))
        out['density_rmse'] = float(rmse(rho_hat, rho_true))
    return out


def _broadcast(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """把 (N,) 的掩码广播到 (B, T, N)。"""
    if mask.dim() == 1:
        return mask.view(1, 1, -1).expand_as(like)
    return mask


# ------------------------------------------------------------------ 风险读出
def confusion_matrix(pred: torch.Tensor, true: torch.Tensor,
                     n_classes: int, mask: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
    """返回 (K, K) 混淆矩阵，行是真值、列是预测。"""
    p, _ = _flat(pred, mask)
    t, _ = _flat(true, mask)
    idx = t.long() * n_classes + p.long()
    cm = torch.bincount(idx, minlength=n_classes ** 2)
    return cm.view(n_classes, n_classes).float()


def risk_metrics(probs: torch.Tensor, labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """论文 5.5 节的风险读出口径。

    Args:
        probs:  (B, T, N, K) 风险概率分布。
        labels: (B, T, N) 真实风险等级。

    Note:
        论文指出加权平均召回率恒等于准确率，故此处仍如实给出该恒等式对应的
        两个数值，便于复核；主要指标为宏平均 F1 与极高风险召回率。
    """
    k = probs.shape[-1]
    pred = probs.argmax(dim=-1)
    cm = confusion_matrix(pred, labels, k, mask)               # (K, K)
    total = cm.sum().clamp(min=1.0)
    acc = cm.diag().sum() / total

    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    precision = tp / (tp + fp).clamp(min=1e-9)
    recall = tp / (tp + fn).clamp(min=1e-9)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)

    support = cm.sum(dim=1)
    weight = support / total
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * weight).sum()

    # 极高风险（最高等级）召回率
    extreme_recall = recall[-1]
    # 加权平均召回率：按类别占比加权，恒等于准确率，此处如实计算以便复核
    weighted_recall = (recall * weight).sum()

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'extreme_recall': float(extreme_recall),
        'weighted_recall': float(weighted_recall),
        'per_class_recall': [float(v) for v in recall],
        'per_class_f1': [float(v) for v in f1],
        'support': [int(v) for v in support],
    }


def coverage_at_risk(probs: torch.Tensor, labels: torch.Tensor,
                     target_level: int = 3) -> float:
    """目标等级被正确识别的比例，用于预警漏报分析。"""
    pred = probs.argmax(dim=-1)
    sel = labels == target_level
    if sel.sum() == 0:
        return float('nan')
    return float((pred[sel] == target_level).float().mean())
