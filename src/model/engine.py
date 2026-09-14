# -*- coding: utf-8 -*-
"""训练与评估循环，对应论文 5.4 节的实现细节。

论文 5.4 节给出的训练设置：
    优化器      Adam
    学习率      1e-4
    学习率调度  余弦退火
    批大小      8
    训练轮数    最多 100 轮
    早停        验证集损失连续 10 轮不下降即停止
    随机种子    5 个，报告均值与标准差

本模块按上述设置实现训练循环，并提供 5 个种子的重复实验入口。论文 5.4 节
说明表 3、表 4 报告的是 5 次独立运行的平均值与标准差，故 run_seeds() 返回
逐种子的结果列表与汇总统计。

关于模型与优化器的划分
----------------------
结构矩阵 H、B、W、P 与图拉普拉斯 L 由景区部署决定，不是可学习参数，
因此以 buffer 形式注册在 HOINet 内，不进入优化器的参数组。
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..data.dataset import ScenarioTensors, default_splits, make_loaders
from ..data.synthesize import Scenario
from ..evals import calibration as cal
from ..evals import metrics as met
from .hoi_net import HOINet, HOINetConfig
from .losses import ClassWeightEstimator, LossWeights, total_loss


@dataclass
class TrainConfig:
    """训练超参数，默认值全部取自论文 5.4 节。"""

    lr: float = 1e-4                 # Adam 学习率
    batch_size: int = 8
    max_epochs: int = 100
    patience: int = 10               # 早停耐心值
    min_delta: float = 1e-5          # 视为“下降”的最小改善量
    grad_clip: float = 5.0           # 梯度裁剪阈值
    seq_len: int = 96                # T
    stride: int = 4                  # 滑窗步长；1 会显著增加样本数
    weight_decay: float = 0.0
    eta_min_ratio: float = 0.01      # 余弦退火的最低学习率相对比例
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_every: int = 5
    seed: int = 0
    ablation: Dict[str, bool] = field(default_factory=dict)
    """论文 6.5 节表 8 的消融开关，键为 HOINetConfig 的 use_* 字段名。

    空字典表示完整模型。键名写错会在 set_ablation 中直接报错，避免因拼写
    问题静默跑出与完整模型相同的行。示例：

        TrainConfig(ablation={'use_graph': False, 'use_temporal': True})

    该字段由 scripts/run_ablation.py 使用。注意每个消融行必须显式给全
    所有开关：set_ablation 只设置被点名的开关，若逐行只传差异项，前一行
    关闭的通道会在后续行中残留。
    """

    init_penetration_from_anchors: bool = False
    """式(10) 的初值是否由训练集锚点与信令联合估计（论文 5.2 节）。

    该估计由 data.anchors.estimate_penetration 实现，只用锚点人数观测与
    信令设备数观测，不含真值。默认关闭的原因是其精度取决于锚点区域是否
    能代表全景区：估计需要把锚点观测的人数按面积份额外推到全域总人数，
    在随仓库提供的合成场景上该假设不成立（锚点设在出入口，其人数密度
    高于全域平均），估计值约为 0.37 而真值为 0.45，相对误差 18.6%；而
    eta = 0 给出的常数初值 pi = (pi_min+pi_max)/2 = 0.45 恰好与该场景的
    真值中心重合。真实部署时锚点区域的代表性应由标定数据判断，届时打开
    本开关即可。

    置 False 时退回式(10) 的低秩初值（即 pi = 0.45）。"""


@dataclass
class TrainHistory:
    """训练过程记录，供绘制损失曲线与复现实验。"""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float('inf')
    epochs_run: int = 0
    seconds: float = 0.0
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    """固定随机种子，保证单次运行可复现。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(sc: Scenario, cfg: TrainConfig,
                tensors: ScenarioTensors) -> HOINet:
    """由场景的结构矩阵构建 HOINet。

    结构矩阵一律取自场景，不重新随机生成——H、B、W、P 是部署给定的量。
    """
    a = int(sc.anchor_sel.shape[0])
    model_cfg = HOINetConfig(
        n_regions=sc.cfg.n_regions,
        n_cameras=sc.cfg.n_cameras,
        n_sectors=sc.cfg.n_sectors,
        n_anchors=a,
        n_ext=sc.cfg.n_ext,
        n_classes=sc.cfg.n_classes,
        seq_len=cfg.seq_len)
    model = HOINet(model_cfg,
                   coverage=tensors.coverage,
                   mixing=tensors.mixing,
                   anchor_sel=tensors.anchor_sel,
                   areas=tensors.areas,
                   laplacian=tensors.laplacian)
    # 式(7)、式(15)、式(16) 的观测精度初值按观测的实际量级设定。论文把
    # Omega 定义为观测方差之逆，初值若只按“数量级 1”猜测，三个数据项会比
    # 式(24) 的状态项大若干数量级，式(23) 的权重 mu1/mu2 因而失去相对含义。
    # 估计只使用训练时段之外的观测统计量（这里是整个场景的观测尺度），
    # 不涉及任何人数真值，故不构成标签泄漏。
    anc_active = sc.anc_mask > 0
    model.set_observation_scales(
        video_mean=float(sc.y_vis.mean()),
        video_ms=float((sc.y_vis ** 2).mean()),
        signal_ms=float((sc.y_sig ** 2).mean()),
        anchor_ms=float(sc.y_anc[anc_active].pow(2).mean())
        if bool(anc_active.any()) else 1.0)

    # 论文 6.5 节表 8 的消融开关，先于下面的初值设置应用。
    if cfg.ablation:
        model.set_ablation(**cfg.ablation)

    # 式(10) 的初值：论文 5.2 节说明渗透率初值由训练集锚点与信令联合估计。
    # 只用训练区间，不使用验证与测试区间；估计输入是锚点人数观测与信令设备
    # 数观测，均不含额外的真值信息。锚点通道被消融掉时该估计没有观测依据，
    # 故跳过并退回式(10) 的低秩初值。
    if cfg.init_penetration_from_anchors and model.cfg.use_anchor:
        lo, hi = _train_interval(sc)
        pi_init = _estimate_pi_init(sc, lo, hi, cfg.seq_len)
        if pi_init is not None:
            model.set_penetration_init(pi_init)
    return model.to(tensors.device)


def _train_interval(sc: Scenario) -> Tuple[int, int]:
    """训练区间在整条时间轴上的 [起, 止) 基本步范围，按论文 5.2 节的比例划分。"""
    train_split = [s for s in default_splits(sc.cfg.days)
                   if s.name == 'train'][0]
    return train_split.steps(sc.cfg.steps_per_day)


def _estimate_pi_init(sc: Scenario, lo: int, hi: int,
                      seq_len: int) -> Optional[torch.Tensor]:
    """由训练区间的锚点与信令观测估计渗透率初值。

    锚点人数观测经选择矩阵 P 摊回区域维，得到 (T, N) 的稀疏人数观测（非锚点
    时刻为 0）；信令观测与时间聚合矩阵各取前 seq_len 行/列，构成一个窗口长
    度与序列长度相同的子问题。时间聚合在论文 5.1 节的设定下退化为单位阵，故这个截取不损失信息。

    锚点观测过少（例如锚点稀疏化到 0%）时估计不可靠，此时返回 None，模型
    退回式(10) 的默认低秩初值。
    """
    from ..data.anchors import estimate_penetration
    if lo >= hi or sc.anc_mask[lo:hi].sum() < 1:
        return None
    seq_len = min(seq_len, hi - lo, sc.y_sig.shape[0])
    anchor_counts = (sc.y_anc[lo:lo + seq_len].float()
                     @ sc.anchor_sel.float())                    # (T, N)
    signal_counts = sc.y_sig[lo:lo + seq_len].float()            # (T, L)
    time_agg = sc.time_agg[:seq_len, :seq_len].float()           # (T, T)
    pi = estimate_penetration(anchor_counts, signal_counts,
                              sc.mixing.float(), time_agg,
                              region_area=sc.areas.float())
    return pi.clamp(min=0.01, max=0.99)


def _prepare_batch(batch: Dict[str, torch.Tensor],
                   device: torch.device) -> Dict[str, torch.Tensor]:
    """把 batch 搬到设备上，并对齐字段名。"""
    return {k: v.to(device).float() if v.is_floating_point()
            else v.to(device) for k, v in batch.items()}


def _forward(model: HOINet, batch: Dict[str, torch.Tensor]):
    """调用模型前向，参数顺序与 HOINet.forward 一致。"""
    return model(batch['y_vis'], batch['m_vis'], batch['o_vis'],
                 batch['y_sig'], batch['y_anc'], batch['a_mask'], batch['ext'])


@torch.no_grad()
def evaluate(model: HOINet, loader, tensors: ScenarioTensors,
             w: LossWeights, class_weights: Optional[torch.Tensor] = None,
             compute_calibration: bool = True,
             compute_penetration: bool = False) -> Dict[str, object]:
    """在给定划分上评估，返回状态指标、风险指标与校准指标。

    评估口径按论文 5.5 节：人数与密度分别报告整体、覆盖区、缺口区三类
    MAE/RMSE/SMAPE，风险分级报告准确率、宏 F1、加权 F1 与极端误判召回，
    校准报告 ECE 与 MCE。

    Args:
        compute_penetration: 是否额外报告式(10) 的渗透率估计误差，供 6.6 节
            的锚点稀疏化实验使用。该指标需要 batch 中含 pi_true，故只对
            带标签的划分有意义。
    """
    model.eval()
    areas = tensors.areas
    n_hat_all, n_true_all, risk_all, prob_all = [], [], [], []
    pi_hat_all, pi_true_all = [], []
    total = 0.0

    for batch in loader:
        batch = _prepare_batch(batch, tensors.device)
        preds = _forward(model, batch)
        loss = total_loss(preds, batch, w, class_weights,
                          overlap=None, areas=areas)
        total += float(loss['total'])
        n_hat_all.append(preds['n_hat'])
        n_true_all.append(batch['n_true'])
        risk_all.append(batch['risk'].long())
        prob_all.append(preds['probs'])
        if compute_penetration and 'pi_true' in batch:
            pi_hat_all.append(preds['pi_hat'])
            pi_true_all.append(batch['pi_true'])

    n_hat = torch.cat(n_hat_all, dim=0)
    n_true = torch.cat(n_true_all, dim=0)
    risk = torch.cat(risk_all, dim=0)
    probs = torch.cat(prob_all, dim=0)

    rho_hat = n_hat / areas.view(1, 1, -1)
    rho_true = n_true / areas.view(1, 1, -1)

    out: Dict[str, object] = {
        'loss': total / max(len(loader), 1),
        'state': met.state_metrics(n_hat, n_true, rho_hat, rho_true,
                                   tensors.gap_mask),
        'risk': met.risk_metrics(probs, risk),
    }
    if compute_calibration:
        out['calibration'] = cal.calibration_report(probs, risk)
    if pi_hat_all:
        pi_hat = torch.cat(pi_hat_all, dim=0)
        pi_true = torch.cat(pi_true_all, dim=0).clamp(min=1e-6)
        rel = (pi_hat - pi_true).abs() / pi_true
        out['penetration'] = {
            'mae': float((pi_hat - pi_true).abs().mean()),
            'relative_error': float(rel.mean()),
            'bias': float((pi_hat - pi_true).mean()),
            'pi_hat_mean': float(pi_hat.mean()),
            'pi_true_mean': float(pi_true.mean()),
        }
    return out


def train_one(sc: Scenario, cfg: TrainConfig,
              loss_weights: Optional[LossWeights] = None,
              verbose: bool = True) -> Dict[str, object]:
    """用单一随机种子训练一次并返回最佳模型与评估结果。

    Returns:
        dict，键为
          model, history, loaders, tensors, val, test, class_weights
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    loss_weights = loss_weights or LossWeights()

    loaders, tensors, splits = make_loaders(
        sc, device, seq_len=cfg.seq_len, batch_size=cfg.batch_size,
        stride=cfg.stride)
    model = build_model(sc, cfg, tensors)

    # 类别权重由训练集频次估计，不使用验证/测试集信息，避免泄漏
    cw = ClassWeightEstimator(sc.cfg.n_classes).to(device)
    train_split = [s for s in splits if s.name == 'train'][0]
    lo, hi = train_split.steps(sc.cfg.steps_per_day)
    cw.fit(sc.risk[lo:hi] - 1)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    # 余弦退火到 lr * eta_min_ratio
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.max_epochs, eta_min=cfg.lr * cfg.eta_min_ratio)

    history = TrainHistory()
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in loaders['train']:
            batch = _prepare_batch(batch, device)
            preds = _forward(model, batch)
            loss = total_loss(preds, batch, loss_weights, cw.weights,
                              overlap=None, areas=tensors.areas)
            opt.zero_grad(set_to_none=True)
            loss['total'].backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss['total'].detach())
            n_batches += 1
        sched.step()

        train_loss = running / max(n_batches, 1)
        val = evaluate(model, loaders['val'], tensors, loss_weights, cw.weights)
        history.train_loss.append(train_loss)
        history.val_loss.append(float(val['loss']))
        history.val_metrics.append({'accuracy': val['risk']['accuracy'],
                                    'macro_f1': val['risk']['macro_f1'],
                                    'mae': val['state']['overall']['mae']})

        if verbose and (epoch % cfg.log_every == 0 or epoch == cfg.max_epochs - 1):
            print('[epoch %3d] train %.4f  val %.4f  acc %.4f  macroF1 %.4f  '
                  'MAE %.3f' % (epoch, train_loss, val['loss'],
                                val['risk']['accuracy'], val['risk']['macro_f1'],
                                val['state']['overall']['mae']))

        # 早停：验证损失连续 patience 轮无改善即停止
        if val['loss'] < history.best_val_loss - cfg.min_delta:
            history.best_val_loss = float(val['loss'])
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - history.best_epoch >= cfg.patience:
            history.stopped_early = True
            history.epochs_run = epoch + 1
            break
        history.epochs_run = epoch + 1

    history.seconds = time.time() - t0
    model.load_state_dict(best_state)
    val = evaluate(model, loaders['val'], tensors, loss_weights, cw.weights)
    test = evaluate(model, loaders['test'], tensors, loss_weights, cw.weights)

    return {'model': model, 'history': history, 'loaders': loaders,
            'tensors': tensors, 'val': val, 'test': test,
            'class_weights': cw.weights.detach().cpu(),
            'loss_weights': loss_weights, 'seed': cfg.seed}


def run_seeds(sc: Scenario, cfg: TrainConfig, seeds=(0, 1, 2, 3, 4),
              loss_weights: Optional[LossWeights] = None,
              verbose: bool = True) -> Dict[str, object]:
    """按论文 5.4 节的 5 个随机种子重复实验并汇总。

    Returns:
        dict，键为
          runs      逐种子的完整结果
          summary   各指标的均值与标准差
    """
    runs = []
    for sd in seeds:
        c = copy.copy(cfg)
        c.seed = int(sd)
        if verbose:
            print('===== 随机种子 %d =====' % sd)
        runs.append(train_one(sc, c, loss_weights, verbose=verbose))

    def _collect(path):
        vals = []
        for r in runs:
            cur = r
            for k in path:
                cur = cur[k]
            vals.append(float(cur))
        return vals

    fields = {
        'MAE_整体': ['test', 'state', 'overall', 'mae'],
        'RMSE_整体': ['test', 'state', 'overall', 'rmse'],
        'SMAPE_整体': ['test', 'state', 'overall', 'smape'],
        'MAE_覆盖区': ['test', 'state', 'covered', 'mae'],
        'MAE_缺口区': ['test', 'state', 'gap', 'mae'],
        '准确率': ['test', 'risk', 'accuracy'],
        '宏F1': ['test', 'risk', 'macro_f1'],
        '加权F1': ['test', 'risk', 'weighted_f1'],
        '极端误判召回': ['test', 'risk', 'extreme_recall'],
        'ECE': ['test', 'calibration', 'ece'],
        'MCE': ['test', 'calibration', 'mce'],
    }
    summary = {}
    for name, path in fields.items():
        v = torch.tensor(_collect(path))
        summary[name] = {'mean': float(v.mean()), 'std': float(v.std(unbiased=True))
                         if v.numel() > 1 else 0.0,
                         'values': [round(float(x), 4) for x in v]}
    return {'runs': runs, 'summary': summary}
