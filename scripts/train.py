# -*- coding: utf-8 -*-
"""训练 HOINet。

用法：
    python scripts/train.py --data data/scenic_mm_synth.pt --epochs 100
    python scripts/train.py --seeds 0 1 2 3 4 --out runs/hoinet

默认超参数全部对应论文 5.4 节：Adam，学习率 1e-4，余弦退火，批大小 8，
最多 100 轮，验证损失连续 10 轮不降即早停。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthesize import (ScenarioConfig, build_scenario)  # noqa: E402
from src.model.engine import TrainConfig, run_seeds, train_one  # noqa: E402
from src.model.losses import LossWeights  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='训练 HOINet')
    p.add_argument('--data', default=None,
                   help='由 make_synth_data.py 生成的 .pt；不给则现生成')
    p.add_argument('--out', default='runs/hoinet', help='输出目录')
    p.add_argument('--epochs', type=int, default=100, help='最大轮数，论文为 100')
    p.add_argument('--patience', type=int, default=10, help='早停耐心值，论文为 10')
    p.add_argument('--lr', type=float, default=1e-4, help='学习率，论文为 1e-4')
    p.add_argument('--batch-size', type=int, default=8, help='批大小，论文为 8')
    p.add_argument('--stride', type=int, default=4,
                   help='滑窗步长；1 会显著增加样本数与训练时间')
    p.add_argument('--seq-len', type=int, default=96, help='序列长度 T，论文为 96')
    p.add_argument('--days', type=int, default=60, help='合成数据天数')
    p.add_argument('--seeds', type=int, nargs='+', default=[0],
                   help='随机种子；论文报告 5 个种子，故用 --seeds 0 1 2 3 4')
    p.add_argument('--device', default=None)
    p.add_argument('--mu1', type=float, default=0.1, help='状态损失权重')
    p.add_argument('--mu2', type=float, default=0.05, help='观测一致性权重')
    p.add_argument('--mu3', type=float, default=1e-4, help='一致性损失权重')
    p.add_argument('--lam', type=float, default=0.1, help='风险损失权重')
    return p.parse_args()


def load_scenario(path, days):
    if path and os.path.exists(path):
        print('读取 %s' % path)
        blob = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ScenarioConfig(**blob['config'])
        # 结构矩阵与真值都在 blob 里，重建 Scenario 对象
        from src.data.synthesize import Scenario
        return Scenario(cfg=cfg, **blob['state_dict'])
    print('未提供数据文件，现场生成合成场景')
    return build_scenario(ScenarioConfig(days=days))


def main():
    args = parse_args()
    sc = load_scenario(args.data, args.days)

    cfg = TrainConfig(lr=args.lr, batch_size=args.batch_size,
                      max_epochs=args.epochs, patience=args.patience,
                      seq_len=args.seq_len, stride=args.stride)
    if args.device:
        cfg.device = args.device
    w = LossWeights(lambda_risk=args.lam, lambda_obs=args.mu2,
                    lambda_cons=args.mu3)

    os.makedirs(args.out, exist_ok=True)
    if len(args.seeds) == 1:
        result = train_one(sc, cfg, w, verbose=True)
        torch.save({'model': result['model'].state_dict(),
                    'config': vars(cfg),
                    'seed': args.seeds[0]},
                   os.path.join(args.out, 'model_seed%d.pt' % args.seeds[0]))
        report = {'seed': args.seeds[0],
                  'history': {'train_loss': result['history'].train_loss,
                              'val_loss': result['history'].val_loss,
                              'best_epoch': result['history'].best_epoch,
                              'epochs_run': result['history'].epochs_run,
                              'stopped_early': result['history'].stopped_early,
                              'seconds': result['history'].seconds},
                  'val': {'state': result['val']['state'],
                          'risk': result['val']['risk'],
                          'calibration': result['val']['calibration']},
                  'test': {'state': result['test']['state'],
                           'risk': result['test']['risk'],
                           'calibration': result['test']['calibration']}}
    else:
        result = run_seeds(sc, cfg, seeds=args.seeds, loss_weights=w)
        for r in result['runs']:
            torch.save({'model': r['model'].state_dict(),
                        'config': vars(cfg), 'seed': r['seed']},
                       os.path.join(args.out, 'model_seed%d.pt' % r['seed']))
        report = {'seeds': list(args.seeds), 'summary': result['summary']}

    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n结果摘要：')
    if 'summary' in report:
        for k, v in report['summary'].items():
            print('  %-14s %.4f +- %.4f' % (k, v['mean'], v['std']))
    else:
        st, rk = report['test']['state'], report['test']['risk']
        print('  测试集 MAE %.3f  覆盖区 %.3f  缺口区 %.3f'
              % (st['overall']['mae'], st['covered']['mae'], st['gap']['mae']))
        print('  准确率 %.4f  宏F1 %.4f  ECE %.4f'
              % (rk['accuracy'], rk['macro_f1'],
                 report['test']['calibration']['ece']))
    print('报告已写入 %s' % os.path.join(args.out, 'report.json'))


if __name__ == '__main__':
    main()
