# -*- coding: utf-8 -*-
"""论文 6.5 节表 8 的消融实验。

表 8 按“观测通道”与“结构模块”两个维度组织，每一行明确标注其启用的通道
与模块。本脚本把表 8 的每一行翻译成一组 HOINetConfig 的 use_* 开关，逐行
训练并在测试集上报告风险分级准确率与宏平均 F1。

列与开关的对应关系
------------------
    表 8 的列        开关
    视频            use_video
    信令            use_signal
    渗透率校准      use_penetration
    空间图          use_graph
    时间模块        use_temporal
    锚点            use_anchor

表 8 中未出现的四个开关，即精度门控 use_gate、外部协变量 use_external、
人数守恒 use_conservation，在表 8 的全部行中一律保持为真。这正是表 8
“使信息量差异与结构差异可被区分开”的前提：若在 Visual-only 一行同时关掉
门控，该行的降幅就同时包含信息量差异与结构差异，无法归因。

用法：
    python scripts/run_ablation.py --data data/scenic_mm_synth.pt \\
        --seeds 0 1 2 3 4 --out runs/ablation
    python scripts/run_ablation.py --rows full visual_only --epochs 5

说明两处与代码的对应边界
------------------------
1. 表 8 的 Feature fusion 一行不在本脚本的行列中。该行是把视频与信令放到
   特征层做交融，属于融合层次的对照而非同一模型内的开关，本仓库的 HOINet
   只实现观测层的反演，故无法用 use_* 开关表示。该行对应 scripts 的基线
   对照（见 scripts/evaluate.py 与论文表 9 的口径），本脚本不伪造其数值。
2. 论文 6.5 节正文在表 8 之外还列出了“去除趋势辅助任务”一项。论文的方法
   章节并未定义趋势辅助任务这一损失项，本仓库按“只实现论文涉及的代码”的
   原则不提供该开关，正文这一项待论文补齐定义后再实现。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthesize import ScenarioConfig, build_scenario  # noqa: E402
from src.model.engine import TrainConfig, run_seeds              # noqa: E402
from src.model.losses import LossWeights                          # noqa: E402

# 表 8 未出现的四个开关在全部行中恒为真，理由见模块 docstring
_BASE = {'use_gate': True, 'use_external': True, 'use_conservation': True}

_ALL = ('use_video', 'use_signal', 'use_anchor', 'use_penetration',
        'use_graph', 'use_temporal', 'use_gate', 'use_external',
        'use_conservation')


def _row(name, video=False, signal=False, penetration=False, graph=False,
         temporal=False, anchor=False):
    """由表 8 的六个勾选列生成一组完整的开关字典。

    显式给全九个开关，使每一行互相独立——set_ablation 只设置被点名的键，
    逐行只传差异项会让上一行关闭的通道残留到下一行。
    """
    flags = dict(_BASE)
    flags.update({'use_video': video, 'use_signal': signal,
                  'use_penetration': penetration, 'use_graph': graph,
                  'use_temporal': temporal, 'use_anchor': anchor})
    assert set(flags) == set(_ALL), '开关字典应覆盖全部九个开关'
    return {'name': name, 'switches': flags}


# 表 8 的行，按论文中的顺序
TABLE8 = [
    # 名称                视频   信令   渗透率  空间图  时间    锚点
    _row('Visual-only',   video=True),
    _row('Signal-only',   signal=True),
    _row('HOINet w/o graph',    video=True, signal=True, penetration=True,
         temporal=True, anchor=True),
    _row('HOINet w/o temporal', video=True, signal=True, penetration=True,
         graph=True, anchor=True),
    _row('HOINet w/o anchor',   video=True, signal=True, graph=True,
         temporal=True),
    _row('Full HOINet',   video=True, signal=True, penetration=True,
         graph=True, temporal=True, anchor=True),
]

def _full_except(*off):
    """完整模型但关闭指定的开关。"""
    flags = {k: True for k in _ALL}
    for k in off:
        if k not in flags:
            raise ValueError('未知开关 %r' % k)
        flags[k] = False
    return flags


# 论文 6.5 节正文的三个内部组件消融。这三行关闭的是读出与损失侧的组件，
# 不改变观测通道，故与表 8 分开列出。
INTERNAL = [
    {'name': 'w/o gate', 'switches': _full_except('use_gate')},
    {'name': 'w/o conservation', 'switches': _full_except('use_conservation')},
    {'name': 'w/o external', 'switches': _full_except('use_external')},
]


def parse_args():
    p = argparse.ArgumentParser(description='论文表 8 的消融实验')
    p.add_argument('--data', default=None,
                   help='由 make_synth_data.py 生成的 .pt；不给则现生成')
    p.add_argument('--out', default='runs/ablation', help='输出目录')
    p.add_argument('--rows', nargs='+', default=None,
                   help='只跑指定的行；行名见脚本中的 TABLE8 与 INTERNAL，'
                        '另可用 full 表示完整模型')
    p.add_argument('--internal', action='store_true',
                   help='在表 8 之外追加正文的三个内部组件消融')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=96)
    p.add_argument('--days', type=int, default=60)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                   help='论文报告 5 个随机种子的均值与标准差')
    p.add_argument('--init-penetration-from-anchors', action='store_true',
                   help='式(10) 的初值改由训练集锚点与信令联合估计')
    p.add_argument('--device', default=None)
    return p.parse_args()


def load_scenario(path, days):
    """读取数据文件；未提供时现场生成合成场景。"""
    if path and os.path.exists(path):
        print('读取 %s' % path)
        blob = torch.load(path, map_location='cpu', weights_only=False)
        from src.data.synthesize import Scenario
        return Scenario(cfg=ScenarioConfig(**blob['config']),
                        **blob['state_dict'])
    print('未提供数据文件，现场生成合成场景')
    return build_scenario(ScenarioConfig(days=days))


def select_rows(args):
    rows = TABLE8 + (INTERNAL if args.internal else [])
    if not args.rows:
        return rows
    want = set(args.rows)
    picked = [r for r in rows if r['name'] in want]
    if 'full' in want:
        picked.append({'name': 'Full HOINet',
                       'switches': {k: True for k in _ALL}})
    known = {r['name'] for r in rows} | {'full'}
    missing = want - known
    if missing:
        raise SystemExit('未知的行名 %s；可选：%s'
                         % (sorted(missing), sorted(known)))
    return picked


def main():
    args = parse_args()
    sc = load_scenario(args.data, args.days)
    rows = select_rows(args)
    os.makedirs(args.out, exist_ok=True)

    base = TrainConfig(lr=args.lr, batch_size=args.batch_size,
                       max_epochs=args.epochs, patience=args.patience,
                       seq_len=args.seq_len, stride=args.stride,
                       init_penetration_from_anchors=args.init_penetration_from_anchors)
    if args.device:
        base.device = args.device
    w = LossWeights()

    table = []
    for row in rows:
        cfg = TrainConfig(**vars(base))
        cfg.ablation = row['switches']
        print('\n===== %s =====' % row['name'])
        result = run_seeds(sc, cfg, seeds=args.seeds, loss_weights=w,
                           verbose=True)

        def stat(key):
            return (result['summary'][key]['mean'],
                    result['summary'][key]['std'])

        acc, acc_sd = stat('准确率')
        f1, f1_sd = stat('宏F1')
        mae, mae_sd = stat('MAE_整体')
        entry = {'row': row['name'], 'switches': row['switches'],
                 'seeds': list(args.seeds),
                 'accuracy_mean': acc, 'accuracy_std': acc_sd,
                 'macro_f1_mean': f1, 'macro_f1_std': f1_sd,
                 'mae_mean': mae, 'mae_std': mae_sd}
        table.append(entry)
        print('  %s：准确率 %.2f ± %.2f，宏平均 F1 %.4f ± %.4f，MAE %.2f'
              % (row['name'], 100 * acc, 100 * acc_sd, f1, f1_sd, mae))

    full = [e for e in table if e['row'] == 'Full HOINet']
    if full:
        ref_acc = full[0]['accuracy_mean']
        ref_f1 = full[0]['macro_f1_mean']
        for e in table:
            e['d_accuracy_pp'] = 100.0 * (ref_acc - e['accuracy_mean'])
            e['d_macro_f1'] = ref_f1 - e['macro_f1_mean']

    out = {'scenario_days': sc.cfg.days, 'seeds': list(args.seeds),
           'rows': table}
    path = os.path.join(args.out, 'ablation.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('\n%-22s %-14s %-12s %s' % ('行', '准确率/%', '宏平均 F1', '较完整模型/百分点'))
    for e in table:
        d = e.get('d_accuracy_pp')
        print('%-22s %6.2f ± %-5.2f %8.4f    %s'
              % (e['row'], 100 * e['accuracy_mean'], 100 * e['accuracy_std'],
                 e['macro_f1_mean'],
                 '—' if d is None else '%+.2f' % d))
    print('\n结果已写入 %s' % path)


if __name__ == '__main__':
    main()
