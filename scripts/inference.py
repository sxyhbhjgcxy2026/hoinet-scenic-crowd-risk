# -*- coding: utf-8 -*-
"""在已训练好的 HOINet 上做推理，输出人数、密度、渗透率与风险等级。

训练与推理的区别（论文 5.2 节）
--------------------------------
训练阶段用真值人数计算式(23) 的总损失并反向传播。推理阶段没有真值，模型
只做一次前向：三路观测经式(18) 的反投影与式(19)—式(22) 的 S 级展开得到
状态，再由式(5) 的 Softmax 读出风险概率。因此本脚本不加载 n_true、risk
等标签字段，也不计算任何指标——它只产出预测。

渗透率的处理沿论文 5.2 节：测试时段的渗透率不使用任何真值，只由式(10) 的
有界低秩参数化给出；锚点可用时其观测参与反演，不可用时模型退回先验。

用法：
    # 用训练好的权重推理，结果写为 .pt 与 .json
    python scripts/inference.py --ckpt runs/hoinet/model_seed0.pt \\
        --data data/scenic_mm_synth.pt --split test --out runs/infer

    # 不提供权重时使用随机初始化的模型，仅用于检查流程是否连通
    python scripts/inference.py --days 6 --seq-len 96 --stride 96
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import dataset as ds                                  # noqa: E402
from src.data.synthesize import ScenarioConfig, build_scenario      # noqa: E402
from src.model.engine import TrainConfig, build_model               # noqa: E402

RISK_NAMES = ('低风险', '中风险', '高风险', '极高风险')


def parse_args():
    p = argparse.ArgumentParser(description='HOINet 推理')
    p.add_argument('--ckpt', default=None,
                   help='训练脚本产出的 .pt 权重；不给则用随机初始化的模型')
    p.add_argument('--data', default=None,
                   help='由 make_synth_data.py 生成的 .pt；不给则现生成')
    p.add_argument('--split', default='test',
                   choices=['train', 'val', 'test'],
                   help='在哪个划分区间上推理，论文 5.2 节为 42/9/9 天')
    p.add_argument('--out', default='runs/infer', help='输出目录')
    p.add_argument('--days', type=int, default=60, help='合成数据天数')
    p.add_argument('--seq-len', type=int, default=96, help='序列长度 T')
    p.add_argument('--stride', type=int, default=96,
                   help='滑窗步长；推理时取 seq_len 即为不重叠的整段推理')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default=None)
    p.add_argument('--save-probs', action='store_true',
                   help='是否把逐位置的四类概率一并写入结果（文件更大）')
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


def main():
    args = parse_args()
    sc = load_scenario(args.data, args.days)
    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    # 推理不需要标签，但 dataset 的 with_target 控制的是标签是否随样本给出，
    # 这里保持默认即可：模型前向只用观测字段，标签只落在不参与推理的键上。
    cfg = TrainConfig(seq_len=args.seq_len, batch_size=args.batch_size,
                      stride=args.stride, device=str(device))
    loaders, tensors, splits = ds.make_loaders(
        sc, device, seq_len=args.seq_len, batch_size=args.batch_size,
        stride=args.stride)
    if args.split not in loaders:
        raise SystemExit('未知划分 %r；可选 %s'
                         % (args.split, sorted(loaders)))

    model = build_model(sc, cfg, tensors)
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise SystemExit('权重文件不存在：%s' % args.ckpt)
        blob = torch.load(args.ckpt, map_location=device, weights_only=False)
        state = blob.get('model', blob)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print('提示：%d 个参数在权重文件中缺失（例如消融开关改变了结构）'
                  % len(missing))
        if unexpected:
            print('提示：%d 个参数在模型中没有对应（权重文件版本较旧？）'
                  % len(unexpected))
        print('已加载权重 %s' % args.ckpt)
    else:
        print('警告：未提供权重，使用随机初始化的模型，输出仅用于检查流程')
    model.eval()

    n_hat_all, rho_hat_all, pi_hat_all, probs_all = [], [], [], []
    with torch.no_grad():
        for batch in loaders[args.split]:
            batch = {k: (v.to(device).float() if v.is_floating_point()
                         else v.to(device)) for k, v in batch.items()}
            preds = model(batch['y_vis'], batch['m_vis'], batch['o_vis'],
                          batch['y_sig'], batch['y_anc'], batch['a_mask'],
                          batch['ext'])
            n_hat_all.append(preds['n_hat'].cpu())
            rho_hat_all.append(preds['rho_hat'].cpu())
            pi_hat_all.append(preds['pi_hat'].cpu())
            probs_all.append(preds['probs'].cpu())

    n_hat = torch.cat(n_hat_all, dim=0)
    rho_hat = torch.cat(rho_hat_all, dim=0)
    pi_hat = torch.cat(pi_hat_all, dim=0)
    probs = torch.cat(probs_all, dim=0)
    risk = probs.argmax(dim=-1) + 1                     # 回到 1..K 的记法

    os.makedirs(args.out, exist_ok=True)
    blob = {'n_hat': n_hat, 'rho_hat': rho_hat, 'pi_hat': pi_hat,
            'risk': risk, 'region_names': list(sc.region_names),
            'split': args.split, 'seq_len': args.seq_len,
            'stride': args.stride, 'checkpoint': args.ckpt}
    if args.save_probs:
        blob['probs'] = probs
    torch.save(blob, os.path.join(args.out, 'predictions.pt'))

    # 逐等级的占比，便于快速核对推理输出是否符合场景的风险构成
    counts = torch.bincount(risk.flatten(), minlength=4)[:4].float()
    share = counts / counts.sum().clamp(min=1.0)

    # 时间维按窗口拼接后会有重叠，故这里只给出窗口内的均值，不做时间还原
    summary = {
        'split': args.split,
        'n_windows': int(n_hat.shape[0]),
        'seq_len': args.seq_len,
        'stride': args.stride,
        'n_hat_mean': float(n_hat.mean()),
        'rho_hat_mean': float(rho_hat.mean()),
        'pi_hat_mean': float(pi_hat.mean()),
        'pi_hat_min': float(pi_hat.min()),
        'pi_hat_max': float(pi_hat.max()),
        'risk_share': {RISK_NAMES[i]: float(share[i]) for i in range(4)},
        'checkpoint': args.ckpt,
    }
    with open(os.path.join(args.out, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n划分 %s：%d 个窗口，窗长 %d，步长 %d'
          % (args.split, summary['n_windows'], args.seq_len, args.stride))
    print('人数均值 %.2f，密度均值 %.4f，渗透率均值 %.4f（范围 %.4f..%.4f）'
          % (summary['n_hat_mean'], summary['rho_hat_mean'],
             summary['pi_hat_mean'], summary['pi_hat_min'],
             summary['pi_hat_max']))
    print('风险等级占比：' + '  '.join(
        '%s %.1f%%' % (RISK_NAMES[i], 100 * share[i]) for i in range(4)))
    print('\n预测已写入 %s' % os.path.join(args.out, 'predictions.pt'))


if __name__ == '__main__':
    main()
