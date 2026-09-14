# -*- coding: utf-8 -*-
"""生成合成数据集 Scenic-MM-synth 并保存到磁盘。

用法：
    python scripts/make_synth_data.py --out data/scenic_mm_synth.pt
    python scripts/make_synth_data.py --anchor-sparsity 0.5

生成的张量结构与论文 5.1 节的 Scenic-MM 一致，供训练与评估脚本读取。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthesize import (ScenarioConfig, build_scenario,  # noqa: E402
                                 describe)


def parse_args():
    p = argparse.ArgumentParser(description='生成合成景区多模态数据集')
    p.add_argument('--out', default='data/scenic_mm_synth.pt',
                   help='输出 .pt 路径')
    p.add_argument('--days', type=int, default=60, help='天数，论文为 60')
    p.add_argument('--regions', type=int, default=18, help='区域数 N，论文为 18')
    p.add_argument('--cameras', type=int, default=12, help='摄像头数 M，论文为 12')
    p.add_argument('--sectors', type=int, default=7, help='信令单元数 L，论文为 7')
    p.add_argument('--steps-per-day', type=int, default=96,
                   help='每日基本步数（15 min 步长 -> 96）')
    p.add_argument('--anchor-sparsity', type=float, default=1.0,
                   help='锚点稀疏化档位：1.0 / 0.5 / 0.25 / 0.0')
    p.add_argument('--seed', type=int, default=20260913)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ScenarioConfig(days=args.days, n_regions=args.regions,
                         n_cameras=args.cameras, n_sectors=args.sectors,
                         steps_per_day=args.steps_per_day, seed=args.seed)
    print('生成合成场景：%d 天 x %d 步/天 = %d 个基本步，%d 个区域'
          % (cfg.days, cfg.steps_per_day, cfg.days * cfg.steps_per_day,
             cfg.n_regions))
    sc = build_scenario(cfg, anchor_sparsity=args.anchor_sparsity)

    info = describe(sc)
    print('场景统计：')
    for k, v in info.items():
        print('  %-22s %s' % (k, v))

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'state_dict': {k: v for k, v in vars(sc).items()
                               if k != 'cfg'},
                'config': vars(cfg),
                'anchor_sparsity': args.anchor_sparsity}, out)
    print('已保存到 %s' % out)

    # 同时写一份可读的统计，便于核对与写进实验记录
    meta = out.replace('.pt', '_stats.json')
    with open(meta, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print('统计信息已保存到 %s' % meta)


if __name__ == '__main__':
    main()
