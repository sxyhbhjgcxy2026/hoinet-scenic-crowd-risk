# -*- coding: utf-8 -*-
"""训练目标，对应论文式(23)—式(26)。

    L = L_state + lambda_r * L_risk + lambda_o * L_obs + lambda_c * L_cons   (23)

    L_state  (24)  对人数与密度同时施加绝对误差，末尾加相对误差项
    L_risk   (25)  类别加权交叉熵 + 序数损失
    L_obs        式(14)—式(16) 三项观测一致性损失之和
    L_cons   (26)  重叠区域上视频与信令两条通道反演密度的加权一致性
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """式(23)—式(26) 的权重。

    论文 3.7 节的式(23) 把总损失写作
    $\\mathcal{L}_{state} + \\lambda_r \\mathcal{L}_{risk}
    + \\lambda_o \\mathcal{L}_{obs} + \\lambda_c \\mathcal{L}_{cons}$，
    5.4 节报告其中三个系数为 $\\mu_1 = 0.1$、$\\mu_2 = 0.05$、
    $\\mu_3 = 10^{-4}$。两套记号按数值对应：$\\mu_1$ 即 $\\lambda_r$，
    $\\mu_2$ 即 $\\lambda_o$，$\\mu_3$ 即 $\\lambda_c$。式(24) 的状态项
    本身不带权重，故不存在与 $\\mathcal{L}_{state}$ 对应的 $\\mu$。

    式(24) 的 beta、lambda_rel 与式(25) 的 lambda_ord 在论文 5.4 节未列出
    取值，此处为实现的补充设定，取值为 1.0 / 0.1 / 0.1。

    另需注意 5.4 节还提到一个“正则系数 $\\lambda$”（在 {0.01, 0.1, 1} 内
    搜索），它作用于式(13) 的 $\\mathcal{R}_{spa}$、$\\mathcal{R}_{tem}$、
    $\\mathcal{R}_{jump}$、$\\mathcal{R}_{\\pi}$ 四项正则，不出现在本类中：
    本网络是式(13) 的展开实现，这四项正则的**近端步**由式(21) 的图卷积与
    时间卷积算子承担，因而不作为损失项参与训练。
    """

    lambda_risk: float = 0.1        # lambda_r，风险分类损失权重
    lambda_obs: float = 0.05        # lambda_o，观测一致性损失权重
    lambda_cons: float = 1e-4       # lambda_c，跨模态一致性损失权重
    beta: float = 1.0               # 式(24) 密度项系数
    lambda_rel: float = 0.1         # 式(24) 相对误差项权重
    lambda_ord: float = 0.1         # 式(25) 序数损失权重
    eps: float = 1e-3               # 式(24) 相对误差分母的稳定项


def state_loss(n_hat: torch.Tensor, rho_hat: torch.Tensor,
               n_true: torch.Tensor, rho_true: torch.Tensor,
               w: LossWeights,
               valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """式(24) 状态重建损失。

    Args:
        n_hat:   (B, T, N) 反演人数。
        rho_hat: (B, T, N) 反演密度。
        n_true:  (B, T, N) 人数标签。
        rho_true:(B, T, N) 密度标签。
        valid:   (B, T, N) 掩码，1 表示该位置有人数标注（论文中标注非逐点完备）。
    """
    if valid is None:
        valid = torch.ones_like(n_hat)
    cnt = valid.sum().clamp(min=1.0)
    abs_n = (n_hat - n_true).abs()
    abs_rho = (rho_hat - rho_true).abs()
    # 第一项：人数与密度的绝对误差
    main = ((abs_n + w.beta * abs_rho) * valid).sum() / cnt
    # 第二项：相对误差，避免高密度区域主导训练
    rel = (abs_n / (n_true + w.eps) * valid).sum() / cnt
    return main + w.lambda_rel * rel


def risk_loss(logits: torch.Tensor, labels: torch.Tensor,
              w: LossWeights,
              class_weights: Optional[torch.Tensor] = None,
              valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """式(25) 风险分类损失：类别加权交叉熵 + 序数损失。

    Args:
        logits:  (B, T, N, K) 风险读出分数。
        labels:  (B, T, N) 风险等级，取值 0..K-1。
        class_weights: (K,) 类别权重 omega，用于缓解低风险样本占比过高。
        valid:   (B, T, N) 掩码。
    """
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    else:
        valid = valid.bool()
    if valid.sum() == 0:
        return logits.new_zeros(())

    flat_logits = logits[valid]                                # (P, K)
    flat_labels = labels[valid]                                # (P,)
    ce = F.cross_entropy(flat_logits, flat_labels,
                         weight=class_weights, reduction='mean')

    # 序数项：期望等级与真实等级的绝对差，使“高误判为低”的代价更大
    k = logits.shape[-1]
    levels = torch.arange(k, device=logits.device, dtype=flat_logits.dtype)
    expected = (F.softmax(flat_logits, dim=-1) * levels).sum(dim=-1)
    ordinal = (expected - flat_labels.to(flat_logits.dtype)).abs().mean()
    return ce + w.lambda_ord * ordinal


def observation_loss(aux: Dict) -> torch.Tensor:
    """式(14)—式(16) 三项观测数据项之和（观测层一致性）。"""
    return aux['loss_vis'] + aux['loss_sig'] + aux['loss_anc']


def cross_modal_consistency(n_hat: torch.Tensor, rho_hat: torch.Tensor,
                            g_vis: torch.Tensor, g_sig: torch.Tensor,
                            step: torch.Tensor, areas: torch.Tensor,
                            overlap: Optional[torch.Tensor] = None,
                            quality: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
    """式(26) 跨模态一致性损失。

        L_cons = sum_{(k,t) in Omega_ov} w_{k,t} |rho_hat^vis - rho_hat^sig|

    论文要求“两条通道反演出的密度在置信度加权后保持一致”。本实现把单通道
    反演密度取为：以当前状态为起点、只用该通道的反投影走一步梯度后的密度

        n^c = ReLU(n_hat - eta * g^c),  rho^c = n^c / s_k

    该取法直接复用式(18) 的分项 g^vis 与 g^sig（论文 3.6 节明确说明这两项
    就是各通道把残差送回区域的反投影），无需额外的前向过程。

    Args:
        n_hat:   (B, T, N) 最终状态。
        rho_hat: (B, T, N) 最终密度。
        g_vis:   (B, T, N) 视频通道反投影。
        g_sig:   (B, T, N) 信令通道反投影。
        step:    标量张量，式(20) 的末阶段步长 eta。
        areas:   (N,) 区域有效面积 s_k。
        overlap: (N,) 或 (B, T, N) 重叠区掩码 Omega_ov，1 表示该区域视频与
                 信令同时覆盖。None 时由 coverage 与 mixing 自动推出。
        quality: (B, T, N) 置信度权重 w_{k,t}，None 时取 1。

    Note:
        该损失只在重叠区生效。缺口区域信令是唯一信息来源，若在缺口区施加
        一致性约束会退化为对自身输出的平凡约束（论文 3.7 节末）。
    """
    s = areas.view(1, 1, -1)
    n_vis = F.relu(n_hat - step * g_vis)
    n_sig = F.relu(n_hat - step * g_sig)
    diff = (n_vis - n_sig).abs() / s

    if overlap is None:
        mask = torch.ones_like(diff)
    else:
        mask = overlap.to(diff.dtype)
    wt = torch.ones_like(diff) if quality is None else quality.to(diff.dtype)
    denom = (mask * wt).sum().clamp(min=1.0)
    return (diff * mask * wt).sum() / denom


def total_loss(preds: Dict, batch: Dict, w: LossWeights,
               class_weights: Optional[torch.Tensor] = None,
               overlap: Optional[torch.Tensor] = None,
               areas: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
    """式(23) 的总损失。

    Args:
        preds: HOINet.forward 的返回值。
        batch: 含 n_true、rho_true、risk、valid_state、valid_risk、quality 的字典。

    Returns:
        含 'total' 与各分项的字典，便于训练日志按项监控。
    """
    l_state = state_loss(preds['n_hat'], preds['rho_hat'],
                         batch['n_true'], batch['rho_true'], w,
                         batch.get('valid_state'))
    l_risk = risk_loss(preds['logits'], batch['risk'], w, class_weights,
                       batch.get('valid_risk'))
    l_obs = observation_loss(preds['aux'])
    l_cons = cross_modal_consistency(
        preds['n_hat'], preds['rho_hat'], preds['aux']['g_vis'],
        preds['aux']['g_sig'], preds['aux']['step'],
        areas if areas is not None else torch.ones(
            1, device=preds['n_hat'].device),
        overlap, batch.get('quality'))

    total = (l_state + w.lambda_risk * l_risk
             + w.lambda_obs * l_obs + w.lambda_cons * l_cons)
    return {'total': total, 'state': l_state, 'risk': l_risk,
            'obs': l_obs, 'cons': l_cons}


class ClassWeightEstimator(nn.Module):
    """按类别频次的倒数估计式(25) 的类别权重 omega。

    论文 5.1 节给出四类占比 45% / 30% / 18% / 7%，直接取倒数为权重会让
    极高风险类权重过大（约 14 倍），故按 w_c = (1/f_c)^p 折算，p 默认 0.5。
    """

    def __init__(self, n_classes: int = 4, power: float = 0.5,
                 normalize: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.power = power
        self.normalize = normalize
        self.register_buffer('weights', torch.ones(n_classes))

    @torch.no_grad()
    def fit(self, labels: torch.Tensor) -> torch.Tensor:
        """由训练集标签统计各类频次并写入权重。"""
        counts = torch.bincount(labels.flatten().long(),
                                minlength=self.n_classes).float().clamp(min=1.0)
        freq = counts / counts.sum()
        w = (1.0 / freq) ** self.power
        if self.normalize:
            w = w / w.mean()
        self.weights.copy_(w)
        return self.weights
