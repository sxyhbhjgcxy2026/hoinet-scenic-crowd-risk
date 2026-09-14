# -*- coding: utf-8 -*-
"""论文 6.6 节的三类观测缺陷鲁棒性实验（RQ6）。

三类缺陷分别对应表 10、表 11 与表 12：

    表 10  信令时间戳相对视频平移 Δ ∈ {0, ±1, ±2, ±4} 个基本步
    表 11  测试时段渗透率扰动 δ ∈ {0, 0.1, 0.2, 0.3}
    表 12  测试时段锚点稀疏化到 100% / 50% / 25% / 0%

三者的共同口径是：模型在**干净的**训练与验证区间上训练，只在测试区间注入
缺陷。这正是论文“测试阶段观测质量退化”的设定——若在训练期一并注入缺陷，
模型会把缺陷学到参数里，测得的鲁棒性就失去了意义。

缺陷的注入方式
--------------
时间平移：把测试区间的信令观测沿时间轴整体平移 Δ 步。数据集已按式(9) 的
    时间聚合矩阵 W 把信令摊到基本步，故这里对 y_sig 的时间维做平移。平移
    后越界的部分丢弃，新进入的部分补 0（表示该时刻无观测）。

渗透率扰动：把测试区间的渗透率乘以 (1 + δ) 后由式(9) 重新生成信令观测。
    真值人数 n 不变，因此这是“观测随渗透率漂移而状态不变”的情形。注意锚点
    观测不随渗透率变化：式(11) 的锚点直接观测人数，与设备渗透率无关，故这里
    不动 y_anc，否则会把“锚点观测噪声被重新采样”混入渗透率扰动的影响。表 11
    的两列分别取 use_penetration=True（模型按式(10) 补偿）与
    use_penetration=False（令 pi = 1，即不补偿）。

锚点稀疏化：把测试区间的锚点掩码按档位抽稀。0% 档位下模型只能依靠训练期
    学到的渗透率先验与信令观测，对应表 12 的最后一行。

渗透率估计相对误差的定义
------------------------
表 12 的第三列报告式(10) 的估计相对误差。本脚本按
    mean_{t,k} |pi_hat - pi_true| / pi_true
在测试区间的全部滑窗与全部区域上取平均，即 src/model/engine.py 中
evaluate(..., compute_penetration=True) 的 relative_error。该定义与
“逐时刻逐区域取相对误差再平均”一致，不随区域面积加权。

用法：
    python scripts/run_robustness.py --data data/scenic_mm_synth.pt \\
        --out runs/robustness --seeds 0 1 2 3 4
    python scripts/run_robustness.py --defects shift --shifts 0 1 2 4 --epochs 5
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import anchors as anc                       # noqa: E402
from src.data import dataset as ds                        # noqa: E402
from src.data import signaling as sig                     # noqa: E402
from src.data.synthesize import ScenarioConfig, build_scenario  # noqa: E402
from src.model.engine import (TrainConfig, build_model, evaluate,  # noqa: E402
                              set_seed)
from src.model.losses import (ClassWeightEstimator, LossWeights,  # noqa: E402
                              total_loss)


# --------------------------------------------------------------- 缺陷注入
def _test_span(sc):
    """测试区间在整条时间轴上的 [起, 止)。"""
    test = [s for s in ds.default_splits(sc.cfg.days) if s.name == 'test'][0]
    return test.steps(sc.cfg.steps_per_day)


def _shift(signal: torch.Tensor, delta: int) -> torch.Tensor:
    """沿时间维把信令观测整体平移 delta 步，越界部分补 0。"""
    if delta == 0:
        return signal.clone()
    out = torch.zeros_like(signal)
    t, j = signal.shape
    d = abs(int(delta))
    if d >= t:
        return out
    if delta > 0:                       # 观测整体滞后：新的时刻读到旧的观测
        out[d:] = signal[:t - d]
    else:                               # 观测整体超前
        out[:t - d] = signal[d:]
    return out


def corrupt_shift(sc, delta):
    """注入信令时间戳平移缺陷，只作用于测试区间。"""
    lo, hi = _test_span(sc)
    y = sc.y_sig.clone()
    y[lo:hi] = _shift(sc.y_sig[lo:hi], delta)
    return dataclasses.replace(sc, y_sig=y)


def corrupt_penetration(sc, delta, pi_min, pi_max, seed=0):
    """注入渗透率扰动，只作用于测试区间。

    信令观测由式(9) 用扰动后的渗透率重新生成，人数真值不变。锚点观测不动：
    式(11) 的锚点直接观测人数，与设备渗透率无关。观测只在测试区间被替换，
    训练与验证区间保持干净。
    """
    lo, hi = _test_span(sc)
    pi = sc.pi_true.clone()
    pi[lo:hi] = (sc.pi_true[lo:hi] * (1.0 + delta)).clamp(pi_min, pi_max)

    y_sig = sc.y_sig.clone()
    y_sig[lo:hi] = sig.simulate(sc.n_true[lo:hi], pi[lo:hi], sc.mixing,
                               sc.time_agg[:, lo:hi],
                               noise=sig.SignalNoise(seed=seed))[lo:hi]
    return dataclasses.replace(sc, y_sig=y_sig, pi_true=pi)


def corrupt_anchor_sparsity(sc, level, seed=0):
    """注入锚点稀疏化缺陷，只作用于测试区间。

    闸机锚点是常态化部署，不参与稀释（见 anchors.sparsify 的 n_gates 说明），
    故只稀释人工抽样锚点。这与论文 6.6 节“临时增加的锚点密度”的设定一致。
    """
    lo, hi = _test_span(sc)
    mask = sc.anc_mask.clone()
    sub = anc.sparsify(sc.anc_mask[lo:hi], level, seed=seed,
                       n_gates=sc.cfg.n_anchors_gate)
    mask[lo:hi] = sub
    # 被抽掉的锚点时刻，其人数观测一并清零，避免观测与掩码不一致
    y_anc = sc.y_anc.clone()
    y_anc[lo:hi] = sc.y_anc[lo:hi] * sub
    return dataclasses.replace(sc, y_anc=y_anc, anc_mask=mask)


# --------------------------------------------------------------- 训练与评估
def train_clean(sc, cfg, loss_weights, ablation=None):
    """在干净场景上训练一个模型并返回它。"""
    c = copy.copy(cfg)
    c.ablation = dict(ablation or {})
    set_seed(c.seed)
    device = torch.device(c.device)
    loaders, tensors, splits = ds.make_loaders(
        sc, device, seq_len=c.seq_len, batch_size=c.batch_size, stride=c.stride)
    model = build_model(sc, c, tensors)

    cw = ClassWeightEstimator(sc.cfg.n_classes).to(device)
    train_split = [s for s in splits if s.name == 'train'][0]
    lo, hi = train_split.steps(sc.cfg.steps_per_day)
    cw.fit(sc.risk[lo:hi] - 1)

    opt = torch.optim.Adam(model.parameters(), lr=c.lr,
                           weight_decay=c.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=c.max_epochs, eta_min=c.lr * c.eta_min_ratio)

    best_loss, best_state, best_epoch = float('inf'), None, -1
    for epoch in range(c.max_epochs):
        model.train()
        for batch in loaders['train']:
            batch = {k: (v.to(device).float() if v.is_floating_point()
                         else v.to(device)) for k, v in batch.items()}
            preds = model(batch['y_vis'], batch['m_vis'], batch['o_vis'],
                          batch['y_sig'], batch['y_anc'], batch['a_mask'],
                          batch['ext'])
            loss = total_loss(preds, batch, loss_weights, cw.weights,
                              overlap=None, areas=tensors.areas)
            opt.zero_grad(set_to_none=True)
            loss['total'].backward()
            if c.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            opt.step()
        sched.step()

        val = evaluate(model, loaders['val'], tensors, loss_weights, cw.weights)
        if val['loss'] < best_loss - c.min_delta:
            best_loss, best_epoch = float(val['loss']), epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= c.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tensors, cw.weights, c


def evaluate_corrupted(sc_corrupt, model, cfg, class_weights):
    """在注入缺陷的场景上评估。

    注意 loaders 与 tensors 都随缺陷场景重建：结构矩阵 H、B、W、P、L 不受
    缺陷影响，但观测与掩码被替换，故 tensors 中的 struct 部分保持一致。
    """
    device = torch.device(cfg.device)
    loaders, tensors, _ = ds.make_loaders(
        sc_corrupt, device, seq_len=cfg.seq_len, batch_size=cfg.batch_size,
        stride=cfg.stride)
    out = evaluate(model, loaders['test'], tensors, LossWeights(),
                   class_weights, compute_penetration=True)
    return out


def run_one(sc, cfg, loss_weights, ablation, defects):
    """训练一次，然后对每个缺陷档位评估。"""
    model, _, cw, c = train_clean(sc, cfg, loss_weights, ablation)
    rows = []
    for defect, value, corrupt in defects:
        sc_bad = corrupt(sc)
        out = evaluate_corrupted(sc_bad, model, c, cw)
        rows.append({'defect': defect, 'value': value,
                     'accuracy': out['risk']['accuracy'],
                     'macro_f1': out['risk']['macro_f1'],
                     'mae': out['state']['overall']['mae'],
                     'pi_relative_error':
                         out['penetration']['relative_error']})
    return rows


def parse_args():
    p = argparse.ArgumentParser(description='论文 6.6 节的观测缺陷鲁棒性实验')
    p.add_argument('--data', default=None,
                   help='由 make_synth_data.py 生成的 .pt；不给则现生成')
    p.add_argument('--out', default='runs/robustness', help='输出目录')
    p.add_argument('--defects', nargs='+',
                   default=['shift', 'penetration', 'anchor'],
                   choices=['shift', 'penetration', 'anchor'],
                   help='要跑的缺陷类型')
    p.add_argument('--shifts', type=int, nargs='+', default=[0, 1, 2, 4],
                   help='表 10 的平移幅度；正负号各测一次')
    p.add_argument('--deltas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3],
                   help='表 11 的渗透率扰动幅度')
    p.add_argument('--levels', type=float, nargs='+', default=[1.0, 0.5, 0.25, 0.0],
                   help='表 12 的锚点保留比例')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=96)
    p.add_argument('--days', type=int, default=60)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--device', default=None)
    return p.parse_args()


def load_scenario(path, days):
    if path and os.path.exists(path):
        print('读取 %s' % path)
        from src.data.synthesize import Scenario
        blob = torch.load(path, map_location='cpu', weights_only=False)
        return Scenario(cfg=ScenarioConfig(**blob['config']),
                        **blob['state_dict'])
    print('未提供数据文件，现场生成合成场景')
    return build_scenario(ScenarioConfig(days=days))


def build_defects(sc, args):
    """按要跑的缺陷类型组装 (名称, 档位, 注入函数) 列表。"""
    out = []
    if 'shift' in args.defects:
        for d in args.shifts:
            if d == 0:
                out.append(('shift', 0, lambda s: corrupt_shift(s, 0)))
            else:
                for sign in (+1, -1):
                    out.append(('shift', sign * d,
                                lambda s, dd=sign * d: corrupt_shift(s, dd)))
    if 'penetration' in args.defects:
        for delta in args.deltas:
            out.append(('penetration', delta,
                        lambda s, dd=delta: corrupt_penetration(
                            s, dd, pi_min=0.05, pi_max=0.85)))
    if 'anchor' in args.defects:
        for lv in args.levels:
            out.append(('anchor', lv,
                        lambda s, ll=lv: corrupt_anchor_sparsity(s, ll)))
    return out


def main():
    args = parse_args()
    sc = load_scenario(args.data, args.days)
    os.makedirs(args.out, exist_ok=True)

    base = TrainConfig(lr=args.lr, batch_size=args.batch_size,
                       max_epochs=args.epochs, patience=args.patience,
                       seq_len=args.seq_len, stride=args.stride)
    if args.device:
        base.device = args.device
    w = LossWeights()
    defects = build_defects(sc, args)

    # 表 11 需要两列：按式(10) 补偿渗透率，以及令 pi = 1 不补偿
    variants = [('HOINet', None)]
    if 'penetration' in args.defects:
        variants.append(('无渗透率补偿', {'use_penetration': False}))

    results = {}
    for name, ablation in variants:
        per_seed = {sd: [] for sd in args.seeds}
        for sd in args.seeds:
            c = TrainConfig(**vars(base))
            c.seed = sd
            print('\n===== %s / 种子 %d =====' % (name, sd))
            per_seed[sd] = run_one(sc, c, w, ablation, defects)
            for r in per_seed[sd]:
                print('  %-12s %-6g accuracy %.4f  macroF1 %.4f'
                      % (r['defect'], r['value'], r['accuracy'], r['macro_f1']))

        # 逐档位汇总均值与标准差
        summary = []
        for i, (defect, value, _) in enumerate(defects):
            acc = torch.tensor([per_seed[sd][i]['accuracy'] for sd in args.seeds])
            f1 = torch.tensor([per_seed[sd][i]['macro_f1'] for sd in args.seeds])
            pi = torch.tensor([per_seed[sd][i]['pi_relative_error']
                               for sd in args.seeds])
            _std = (lambda v: float(v.std(unbiased=True))
                    if v.numel() > 1 else 0.0)
            summary.append({
                'defect': defect, 'value': value,
                'accuracy_mean': float(acc.mean()), 'accuracy_std': _std(acc),
                'macro_f1_mean': float(f1.mean()), 'macro_f1_std': _std(f1),
                'pi_relative_error_mean': float(pi.mean()),
                'pi_relative_error_std': _std(pi)})
        results[name] = {'per_seed': per_seed, 'summary': summary}

    path = os.path.join(args.out, 'robustness.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'scenario_days': sc.cfg.days, 'seeds': list(args.seeds),
                   'results': results}, f, ensure_ascii=False, indent=2)

    for name, blob in results.items():
        print('\n%s' % name)
        print('%-12s %-8s %-16s %-12s %s'
              % ('缺陷', '档位', '准确率/%', '宏平均 F1', '渗透率相对误差/%'))
        for e in blob['summary']:
            print('%-12s %-8g %6.2f ± %-5.2f  %7.4f    %6.2f'
                  % (e['defect'], e['value'], 100 * e['accuracy_mean'],
                     100 * e['accuracy_std'], e['macro_f1_mean'],
                     100 * e['pi_relative_error_mean']))
    print('\n结果已写入 %s' % path)


if __name__ == '__main__':
    main()
