# -*- coding: utf-8 -*-
"""评估已训练的 HOINet，并按论文 5.5 节的口径输出三张表。

用法：
    python scripts/evaluate.py --ckpt runs/hoinet/model_seed0.pt
    python scripts/evaluate.py --ckpt runs/hoinet/model_seed0.pt --anchor-sparsity 0.25

输出的三张表对应论文的：
    状态反演   整体 / 覆盖区 / 缺口区的 MAE、RMSE、SMAPE，以及密度 MAE
    风险读出   准确率、宏 F1、加权 F1、极高风险召回
    概率校准   Brier、ECE、最大置信度偏差

--anchor-sparsity 可选 1.0 / 0.5 / 0.25 / 0.0，用于复现论文关于锚点密度的
消融实验；该参数只影响评估时锚点的可用性，不改变已训练权重。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ScenarioTensors, make_loaders  # noqa: E402
from src.data.synthesize import (ScenarioConfig, build_scenario)  # noqa: E402
from src.model.engine import TrainConfig, build_model, evaluate  # noqa: E402
from src.model.losses import LossWeights  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='评估 HOINet')
    p.add_argument('--ckpt', required=True, help='train.py 保存的 .pt')
    p.add_argument('--data', default=None, help='数据集 .pt；不给则现生成')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--anchor-sparsity', type=float, default=1.0,
                   help='锚点档位：1.0 / 0.5 / 0.25 / 0.0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--out', default=None, help='把结果写成 JSON 的路径')
    p.add_argument('--device', default=None)
    return p.parse_args()


def load_scenario(path):
    if path and os.path.exists(path):
        from src.data.synthesize import Scenario
        blob = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ScenarioConfig(**blob['config'])
        return Scenario(cfg=cfg, **blob['state_dict'])
    return build_scenario(ScenarioConfig())


def fmt_table(title, rows, header):
    width = max(len(str(r[0])) for r in rows) + 2
    lines = ['', title, '-' * (width + 12 * (len(header) - 1))]
    lines.append('%-*s' % (width, '') +
                 ''.join('%12s' % h for h in header))
    for r in rows:
        lines.append('%-*s' % (width, r[0]) +
                     ''.join('%12.4f' % v for v in r[1:]))
    return '\n'.join(lines)


def main():
    args = parse_args()
    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    sc = load_scenario(args.data)
    # 锚点档位只改评估时的可用性：重算掩码，其余结构不变
    if args.anchor_sparsity < 1.0:
        from src.data import anchors as anc
        from src.data.synthesize import GAP_REGION_NAMES
        layout = anc.AnchorLayout(gate_regions=(0, 11, 5),
                                  manual_regions=(3, 8),
                                  manual_times_per_day=2,
                                  steps_per_day=sc.cfg.steps_per_day)
        n_gate = len(layout.gate_regions)
        n_manual = len(layout.manual_regions)
        mask = anc.anchor_mask(layout, n_gate, n_manual, sc.n_true.shape[0])
        # 闸机锚点不参与稀释（见 anchors.sparsify 的 n_gates 说明）。
        # 掩码为 0 的位置模型不会读取锚点观测（见 HOINet._anchor_grad 中的
        # resid 与 anc_mask 相乘），故 y_anc 无需一并清零。
        sc.anc_mask = anc.sparsify(mask, args.anchor_sparsity,
                                   seed=sc.cfg.seed, n_gates=n_gate)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    tcfg = TrainConfig(**ckpt['config']) if 'config' in ckpt else TrainConfig()
    tcfg.device = str(device)
    tcfg.batch_size = args.batch_size
    tcfg.stride = args.stride

    loaders, tensors, _ = make_loaders(sc, device, seq_len=tcfg.seq_len,
                                       batch_size=tcfg.batch_size,
                                       stride=tcfg.stride)
    model = build_model(sc, tcfg, tensors)
    model.load_state_dict(ckpt['model'])
    model.eval()

    res = evaluate(model, loaders[args.split], tensors, LossWeights())

    st, rk, cl = res['state'], res['risk'], res['calibration']
    print('评估划分：%s    锚点档位：%.0f%%'
          % (args.split, args.anchor_sparsity * 100))

    print(fmt_table('表A 状态反演误差（论文表 2 口径）',
                    [['整体', st['overall']['mae'], st['overall']['rmse'],
                      st['overall']['smape'], st.get('density_mae', float('nan'))],
                     ['覆盖区', st['covered']['mae'], st['covered']['rmse'],
                      st['covered']['smape'], float('nan')],
                     ['缺口区', st['gap']['mae'], st['gap']['rmse'],
                      st['gap']['smape'], float('nan')]],
                    ['MAE', 'RMSE', 'SMAPE', '密度MAE']))
    print('  缺口区/覆盖区 MAE 之比：%.4f' % st['gap_over_covered'])

    print(fmt_table('表B 风险读出（论文表 3 口径）',
                    [['全部', rk['accuracy'], rk['macro_f1'],
                      rk['weighted_f1'], rk['extreme_recall']]],
                    ['准确率', '宏F1', '加权F1', '极高风险召回']))
    print('  各类召回率：' + '  '.join('%.3f' % v for v in rk['per_class_recall']))

    print(fmt_table('表C 概率校准（论文表 4 口径）',
                    [['全部', cl['brier'], cl['ece'],
                      cl['max_confidence_deviation']]],
                    ['Brier', 'ECE', '最大置信度偏差']))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'split': args.split,
                       'anchor_sparsity': args.anchor_sparsity,
                       'state': st, 'risk': rk, 'calibration': cl},
                      f, ensure_ascii=False, indent=2)
        print('\n结果已写入 %s' % args.out)


if __name__ == '__main__':
    main()
