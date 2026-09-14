# -*- coding: utf-8 -*-
"""论文中可检验的不变量与结构约束。

这些测试检查的是论文的**结构性质**，不是训练精度：形状契约、观测模型的
代数性质、阈值的单调性、校准指标之间的不等式、数据划分的不重叠等。它们
不依赖训练是否收敛，因此可以在几秒内跑完。

运行方式（不需要 pytest）：
    python tests/test_invariants.py

也可以直接用 pytest 收集：
    pytest tests/
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import anchors as anc                        # noqa: E402
from src.data import dataset as ds                         # noqa: E402
from src.data import graph as gr                           # noqa: E402
from src.data import labels as lab                          # noqa: E402
from src.data import signaling as sig                       # noqa: E402
from src.data import video as vid                           # noqa: E402
from src.data.synthesize import ScenarioConfig, build_scenario  # noqa: E402
from src.evals import calibration as cal                    # noqa: E402
from src.evals import metrics as met                        # noqa: E402
from src.model.engine import TrainConfig, build_model, evaluate  # noqa: E402
from src.model.hoi_net import HOINetConfig                  # noqa: E402
from src.model.losses import LossWeights                    # noqa: E402
from src.model.ops import MonotoneReadout                   # noqa: E402

_SCENARIO = None


def scenario():
    """共用一个短场景：本文件不检验训练，短场景足以覆盖全部结构路径。"""
    global _SCENARIO
    if _SCENARIO is None:
        _SCENARIO = build_scenario(ScenarioConfig(days=3))
    return _SCENARIO


def _tensors():
    return ds.ScenarioTensors(scenario(), torch.device('cpu'))


def _batch(b=2, t=96):
    sc = scenario()
    return (sc.y_vis[:t].unsqueeze(0).expand(b, -1, -1),
            sc.m_vis[:t].unsqueeze(0).expand(b, -1, -1),
            sc.o_vis[:t].unsqueeze(0).expand(b, -1, -1),
            sc.y_sig[:t].unsqueeze(0).expand(b, -1, -1),
            sc.y_anc[:t].unsqueeze(0).expand(b, -1, -1),
            sc.anc_mask[:t].unsqueeze(0).expand(b, -1, -1),
            sc.ext[:t].unsqueeze(0).expand(b, -1, -1))


# ------------------------------------------------------- 1. 锚点：表 1 的算式
def test_anchor_events_match_table_1():
    """论文表 1：60 天 x 3 出入口 x 2 次 = 360 个锚点时刻。

    闸机锚点每天每点位触发 gate_times_per_day 次，60 天共
    n_gates * times_per_day * 60 = 3 * 2 * 60 = 360 次。
    """
    layout = anc.AnchorLayout(gate_regions=(0, 11, 5),
                              manual_regions=(3, 8),
                              manual_times_per_day=2, steps_per_day=96)
    n_gate, n_manual = len(layout.gate_regions), len(layout.manual_regions)
    t_total = 60 * 96
    mask = anc.anchor_mask(layout, n_gate, n_manual, t_total)
    gate_events = float(mask[:, :n_gate].sum())
    assert gate_events == 360.0, '闸机锚点次数应为 360，实际 %g' % gate_events
    # 每个闸机点位每日恰好两次
    per_day = mask[:96, :n_gate].sum()
    assert float(per_day) == 6.0, '每日闸机锚点应为 3 点位 x 2 次 = 6，实际 %g' % per_day


def test_anchor_mask_repeats_daily_not_stretched():
    """每日采样模式必须逐日重复，而不是把一天的模式拉伸到整条时间轴。

    把一天的 96 步模式逐日平移，则第 d 天的掩码应与第 0 天的完全相同。
    """
    layout = anc.AnchorLayout(gate_regions=(0, 11, 5), manual_regions=(3, 8),
                              steps_per_day=96)
    mask = anc.anchor_mask(layout, 3, 2, 10 * 96)
    day0 = mask[:96]
    for d in range(1, 10):
        assert torch.equal(mask[d * 96:(d + 1) * 96], day0), \
            '第 %d 天的锚点模式与第 0 天不一致' % d


def test_anchor_sparsity_levels_are_nested():
    """稀疏化档位应确实减少可稀释的锚点，且 0% 档位全为 0、100% 档位不变。

    这里必须检查“确实减少”，不能只检查单调性：一个什么都不做的实现同样满足
    单调不减，早先的版本正因如此把一个空操作的缺陷掩盖了过去。
    """
    layout = anc.AnchorLayout(gate_regions=(0, 11, 5), manual_regions=(3, 8),
                              steps_per_day=96)
    n_gates, n_manual = 3, 2
    mask = anc.anchor_mask(layout, n_gates, n_manual, 4 * 96)
    counts = [float(anc.sparsify(mask, lv, seed=0, n_gates=n_gates).sum())
              for lv in (1.0, 0.5, 0.25, 0.0)]
    assert counts[0] == float(mask.sum())
    assert counts[-1] == 0.0
    assert counts == sorted(counts, reverse=True), '稀疏化档位未单调: %s' % counts

    # 闸机列不参与稀释，可稀释的只有人工抽样列
    n_gates_total = float(mask[:, :n_gates].sum())
    dilutable = float(mask[:, n_gates:].sum())
    assert dilutable > 0, '人工抽样列应有锚点，否则本测试无效'
    for lv, got in zip((0.5, 0.25), counts[1:3]):
        expect = n_gates_total + round(dilutable * lv) if lv > 0 else 0.0
        assert abs(got - expect) <= n_manual, \
            '档位 %g 应保留约 %g 个锚点，实际 %g（未真正稀释？）' % (
                lv, expect, got)
        assert got < counts[0], '档位 %g 未减少任何锚点' % lv

    # 闸机列在非零档位下必须原样保留
    for lv in (0.5, 0.25):
        kept = anc.sparsify(mask, lv, seed=0, n_gates=n_gates)
        assert torch.equal(kept[:, :n_gates], mask[:, :n_gates]), \
            '档位 %g 不应稀释闸机锚点' % lv

    # n_gates 缺省时全部列参与稀释，此时 50% 档位必须少于 100% 档位
    allp = anc.sparsify(mask, 0.5, seed=0)
    assert float(allp.sum()) < float(mask.sum()), \
        'n_gates 缺省时应对全部列稀释'


# ------------------------------------------- 2. 观测模型的代数性质
def test_mixing_columns_sum_to_one():
    """空间混合矩阵 B 在列上归一化：每个区域的设备全部落到扇区上。

    式(10) 之后的渗透率估计依赖这一性质（见 anchors.estimate_penetration）。
    """
    b = scenario().mixing
    col = b.sum(dim=0)
    assert torch.allclose(col, torch.ones_like(col), atol=1e-5), \
        'B 的列和应为 1，实际范围 %.4f..%.4f' % (float(col.min()), float(col.max()))
    assert float(b.min()) >= 0.0, 'B 不应含负元'


def test_time_aggregation_is_identity_at_15min():
    """论文 5.1 节的基本步与信令窗口同为 15 min，此时 W 退化为单位阵。

    这是式(9) 的时间聚合在模型内不出现的原因。
    """
    w = sig.time_aggregation_matrix(96, step_minutes=15, window_minutes=15)
    assert tuple(w.shape) == (96, 96)
    assert torch.allclose(w, torch.eye(96), atol=1e-6), '15 min 窗口下 W 应为单位阵'


def test_sector_total_equals_region_device_total():
    """扇区设备总数恒等于各区域设备总数（由 B 的列和为 1 推出）。

        sum_l y_l = sum_l sum_k B_lk pi_k n_k = sum_k pi_k n_k

    这是式(9) 的一个严格代数恒等式，也是 estimate_penetration 的估计依据。
    注意该恒等式对**无噪观测**严格成立；有噪观测只在期望意义下成立，故这里
    先在无噪条件下验证恒等式本身，再单独检查实测观测的噪声水平。
    """
    sc = scenario()
    t = 96
    mixed = sc.pi_true[:t] * sc.n_true[:t]                 # (T, N) 设备数

    # (1) 无噪时严格成立。窗口数与基本步数在论文的 15 min 设定下相等，
    # 故时间聚合矩阵取前 t 行前 t 列即对应前 t 个基本步（W 退化为单位阵）。
    clean = sig.simulate(sc.n_true[:t], sc.pi_true[:t], sc.mixing,
                         sc.time_agg[:t, :t], noise=None)
    lhs = clean.sum(dim=1)                                  # (T,) 扇区总量
    rhs = mixed.sum(dim=1)                                  # (T,) 区域设备总量
    err = float((lhs - rhs).abs().max() / rhs.abs().max())
    assert err < 1e-5, '无噪观测下恒等式应严格成立，相对误差 %.3e' % err

    # (2) 有噪观测的偏差应由噪声水平解释：信令噪声为 3% 乘性 + 1 台加性，
    # 7 个扇区求和后乘性噪声部分被平均，故扇区总量的相对偏差应远小于 3%
    noisy = sc.y_sig[:t].sum(dim=1)
    rel = float((noisy - rhs).abs().max() / rhs.abs().max())
    assert rel < 0.05, \
        '有噪观测的扇区总量偏差 %.3f 超出噪声水平可解释的范围' % rel


def test_scenario_sparsity_actually_thins_anchors():
    """build_scenario 的 anchor_sparsity 必须真正改变锚点数量。

    该路径与 sparsify 共享同一个缺陷历史：档位为空操作时，按档位生成的
    四套数据会完全相同，稀疏化实验因此退化为重复同一条件。
    """
    counts = []
    for lv in (1.0, 0.5, 0.25, 0.0):
        sc = build_scenario(ScenarioConfig(days=10), anchor_sparsity=lv)
        counts.append(int(sc.anc_mask.sum()))
    assert counts == sorted(counts, reverse=True), '档位未单调: %s' % counts
    assert counts[-1] == 0, '0%% 档位应无任何可用锚点'
    assert counts[0] > counts[1] > counts[2] > 0, \
        '档位之间应严格递减且非零，实际 %s' % counts
    # 闸机列不参与稀释，故 100% 与 50% 的差值正是人工抽样列的一半
    full = build_scenario(ScenarioConfig(days=10), anchor_sparsity=1.0)
    half = build_scenario(ScenarioConfig(days=10), anchor_sparsity=0.5)
    assert torch.equal(half.anc_mask[:, :3], full.anc_mask[:, :3]), \
        '闸机锚点不应被稀释'


def test_estimate_penetration_recovers_order_of_magnitude():
    """由锚点与信令估计的渗透率应落在真值的同一量级内。

    该估计把锚点人数按面积份额外推到全域，故必然带有系统偏差；这里只
    检查量级正确（例如不会像按伪逆反摊那样给出真值的十几倍）。
    """
    sc = scenario()
    t = 96
    area = sc.areas.float()
    ac = sc.y_anc[:t].float() @ sc.anchor_sel.float()
    pi = anc.estimate_penetration(ac, sc.y_sig[:t].float(),
                                  sc.mixing.float(), sc.time_agg[:t, :t].float(),
                                  region_area=area)
    assert tuple(pi.shape) == (t, sc.cfg.n_regions)
    ratio = float(pi.median()) / float(sc.pi_true[:t].median())
    assert 0.3 < ratio < 3.0, '估计值与真值的量级不符，比值 %.3f' % ratio


def test_estimate_penetration_rejects_shape_mismatch():
    """mixing 的列数必须与 anchor_counts 的区域数一致。"""
    ac = torch.zeros(8, 5)
    try:
        anc.estimate_penetration(ac, torch.zeros(8, 7), torch.zeros(7, 9),
                                 torch.eye(8))
    except ValueError:
        return
    raise AssertionError('区域数不一致时应抛出 ValueError')


def test_estimate_penetration_handles_missing_anchors():
    """锚点全缺时不应崩溃，也不应产生 NaN。"""
    ac = torch.zeros(8, 5)
    pi = anc.estimate_penetration(ac, torch.zeros(8, 7), torch.zeros(7, 5),
                                  torch.eye(8))
    assert not bool(torch.isnan(pi).any()), '锚点全缺时不应产生 NaN'


# ------------------------------------------------- 3. 风险阈值与标签
def test_capacity_thresholds_are_strictly_increasing():
    """由承载量核定的三级阈值必须严格递增，否则风险等级无法区分。"""
    sc = scenario()
    tau = sc.tau_capacity
    assert bool((tau[:, 1:] > tau[:, :-1]).all()), \
        '阈值应沿风险等级严格递增，实际最小间隔 %.3e' \
        % float((tau[:, 1:] - tau[:, :-1]).min())


def test_risk_labels_are_in_range_and_ordered():
    """风险等级取值落在 1..K，且等级随风险分数单调不减。"""
    sc = scenario()
    k = sc.cfg.n_classes
    assert int(sc.risk.min()) >= 1 and int(sc.risk.max()) <= k, \
        '风险等级应在 1..%d 内' % k
    # 同一区域内，分数更高的时刻等级不应更低
    score = lab.risk_score(sc.rho_true, sc.ext, sc.tau[:, 0])
    flat_s = score.flatten()
    flat_r = sc.risk.flatten().float()
    order = torch.argsort(flat_s)
    bad = (flat_r[order][1:] - flat_r[order][:-1]) < -1e-9
    # 标签带噪声，允许少量逆序，但比例应很低
    frac = float(bad.float().mean())
    assert frac < 0.35, '等级与风险分数的逆序比例过高：%.3f' % frac


def test_gap_and_covered_masks_partition_regions():
    """覆盖区与缺口区应构成对区域的划分，且论文给出 6 个缺口区。"""
    sc = scenario()
    both = (sc.gap_mask.bool() & sc.covered_mask.bool()).sum()
    neither = (~sc.gap_mask.bool() & ~sc.covered_mask.bool()).sum()
    assert int(both) == 0, '覆盖区与缺口区不应重叠'
    assert int(neither) == 0, '每个区域应属于覆盖区或缺口区之一'
    assert int(sc.gap_mask.sum()) == 6, \
        '论文为 6 个缺口区，实际 %d' % int(sc.gap_mask.sum())


# ------------------------------------------------- 4. 数据划分
def test_splits_are_contiguous_and_non_overlapping():
    """42 / 9 / 9 天的划分首尾相接且不重叠，避免时段泄漏。"""
    splits = ds.default_splits(days=60)
    assert [s.name for s in splits] == ['train', 'val', 'test']
    spans = [(s.start_day, s.end_day) for s in splits]
    assert spans[0][0] == 0
    for (_, e), (s2, _) in zip(spans, spans[1:]):
        assert e == s2, '划分区间之间应有间断或重叠: %s' % (spans,)
    assert spans[-1][1] == 60
    assert spans[0][1] == 42 and spans[1][1] == 51, \
        '论文为 42/9/9 天，实际 %s' % (spans,)


def test_splits_scale_to_shorter_scenarios():
    """较短场景下按比例缩放，且总和仍等于场景天数、每段至少一天。"""
    for days in (6, 12, 30):
        spans = [(s.start_day, s.end_day) for s in ds.default_splits(days=days)]
        assert spans[0][0] == 0 and spans[-1][1] == days
        for s, e in spans:
            assert e > s, '划分区间应至少一天: %s' % (spans,)


# ------------------------------------------------- 5. 模型前向的形状与约束
def test_forward_shapes_and_bounds():
    """式(5) 的概率归一、式(10) 的渗透率有界、非负人数。"""
    sc = scenario()
    b, t, n, k = 2, 96, sc.cfg.n_regions, sc.cfg.n_classes
    model = build_model(sc, TrainConfig(seq_len=t, stride=8), _tensors())
    out = model(*_batch(b, t))

    assert tuple(out['n_hat'].shape) == (b, t, n)
    assert tuple(out['pi_hat'].shape) == (b, t, n)
    assert tuple(out['rho_hat'].shape) == (b, t, n)
    assert tuple(out['probs'].shape) == (b, t, n, k)

    assert float(out['n_hat'].min()) >= 0.0, '式(21) 的 ReLU 应保证人数非负'
    probs = out['probs']
    assert torch.allclose(probs.sum(dim=-1), torch.ones(b, t, n), atol=1e-5), \
        '式(5) 的 Softmax 应在等级维上归一'
    cfg = model.cfg
    assert float(out['pi_hat'].min()) >= cfg.pi_min - 1e-5
    assert float(out['pi_hat'].max()) <= cfg.pi_max + 1e-5, \
        '式(10) 的渗透率应落在 [%.2f, %.2f] 内' % (cfg.pi_min, cfg.pi_max)
    assert not bool(torch.isnan(out['n_hat']).any())
    assert not bool(torch.isnan(out['probs']).any())


def test_gates_form_distribution():
    """式(19) 的门控在每个阶段、每个位置都应是概率分布。"""
    sc = scenario()
    model = build_model(sc, TrainConfig(seq_len=96, stride=8), _tensors())
    out = model(*_batch(2, 96))
    g = out['aux']['gates']                                     # (S, B, T, 3)
    assert tuple(g.shape) == (model.cfg.n_stages, 2, 96, 3)
    assert torch.allclose(g.sum(dim=-1), torch.ones(2, 96), atol=1e-5), \
        '门控权重之和应为 1'


def test_backward_runs_and_gradients_are_finite():
    """式(23) 的总损失可反向传播，且梯度有限。"""
    from src.model.losses import total_loss
    sc = scenario()
    tensors = _tensors()
    model = build_model(sc, TrainConfig(seq_len=96, stride=8), tensors)
    batch = _batch(2, 96)
    keys = ('y_vis', 'm_vis', 'o_vis', 'y_sig', 'y_anc', 'a_mask', 'ext')
    preds = model(*batch)
    d = dict(zip(keys, batch))
    d['n_true'] = sc.n_true[:96].unsqueeze(0).expand(2, -1, -1)
    d['rho_true'] = sc.rho_true[:96].unsqueeze(0).expand(2, -1, -1)
    d['risk'] = (sc.risk[:96] - 1).unsqueeze(0).expand(2, -1, -1)
    d['rho_hat'] = preds['rho_hat']
    loss = total_loss(preds, d, LossWeights(), None, areas=tensors.areas)
    loss['total'].backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert bool(torch.isfinite(p.grad).all()), '%s 的梯度含 inf/nan' % name


# ------------------------------------------------- 6. 单调读出（式(5)）
def test_monotone_readout_weights_are_non_decreasing():
    """式(5) 的单调性约束：单调分量的权重沿风险等级非降。

    这是式(5) 的充分条件。
    """
    r = MonotoneReadout(n_monotone=3, n_free=9, n_classes=4)
    w = r.monotone_weight()
    assert tuple(w.shape) == (3, 4)
    assert bool((w[:, 1:] >= w[:, :-1] - 1e-7).all()), \
        '权重应沿等级非降，实际 %s' % w.tolist()


def test_monotone_readout_is_non_decreasing_in_input():
    """数值检验：单调分量增大时，高风险等级的累积概率不下降。"""
    torch.manual_seed(0)
    r = MonotoneReadout(n_monotone=3, n_free=9, n_classes=4)
    z_free = torch.randn(1, 1, 1, 9)
    base = torch.zeros(1, 1, 1, 3)
    probs = [torch.softmax(r(base + delta, z_free), dim=-1)
             for delta in (0.0, 0.5, 1.0, 2.0)]
    for c in range(4):
        for v in range(3):
            tail = [float(p[..., c:].sum()) for p in probs]
            for a, b in zip(tail, tail[1:]):
                assert b >= a - 1e-6, \
                    '等级 %d 的累积概率随输入下降: %s' % (c, tail)
            break       # 只需检查一个分量，三个分量同理
        break
    # 三个分量逐一检查
    for j in range(3):
        seq = []
        for delta in (0.0, 0.5, 1.0, 2.0):
            d = torch.zeros(1, 1, 1, 3)
            d[..., j] = delta
            seq.append(torch.softmax(r(base + d, z_free), dim=-1))
        for c in range(4):
            tail = [float(p[..., c:].sum()) for p in seq]
            for a, b in zip(tail, tail[1:]):
                assert b >= a - 1e-6, \
                    '分量 %d 在等级 %d 上的累积概率下降: %s' % (j, c, tail)


# ------------------------------------------------- 7. 评估口径
def test_mce_is_not_less_than_ece():
    """MCE 为各分箱偏差的最大值，故恒有 MCE >= ECE。"""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(500, 4), dim=-1)
    labels = torch.randint(0, 4, (500,))
    e, m = cal.ece(probs, labels)
    assert float(m) >= float(e) - 1e-9, 'MCE %.6f < ECE %.6f' % (float(m), float(e))


def test_perfect_predictions_have_zero_ece():
    """完全正确的预测且置信度为 1 时，ECE 与 Brier 都应为 0。"""
    probs = torch.zeros(10, 4)
    labels = torch.arange(4).repeat_interleave(3)[:10]
    probs[torch.arange(10), labels] = 1.0
    e, m = cal.ece(probs, labels)
    assert float(e) < 1e-6 and float(m) < 1e-6
    b = cal.brier_score(probs, labels)
    assert float(b) < 1e-6, 'Brier 应为 0，实际 %g' % float(b)


def test_gap_metric_uses_the_specified_regions():
    """缺口区指标的样本数应与缺口区区域数一致。"""
    sc = scenario()
    t, n = 48, sc.cfg.n_regions
    n_hat = torch.rand(2, t, n) * 100
    n_true = torch.rand(2, t, n) * 100
    rho_hat = n_hat / sc.areas
    rho_true = n_true / sc.areas
    out = met.state_metrics(n_hat, n_true, rho_hat, rho_true, sc.gap_mask)
    n_gap = int(sc.gap_mask.sum())
    # 缺口区 MAE 应等于只在缺口区区域上计算的 MAE
    manual = float((n_hat[:, :, sc.gap_mask.bool()]
                    - n_true[:, :, sc.gap_mask.bool()]).abs().mean())
    assert abs(out['gap']['mae'] - manual) < 1e-4, \
        '缺口区 MAE 与手工计算不符: %.6f vs %.6f' % (out['gap']['mae'], manual)
    assert out['gap_over_covered'] > 0
    assert n_gap == 6


def test_risk_metrics_on_perfect_prediction():
    """完美预测下准确率与宏 F1 应为 1，极端风险召回应为 1。"""
    torch.manual_seed(0)
    labels = torch.randint(0, 4, (4, 6, 3))
    probs = torch.zeros(4, 6, 3, 4)
    probs.scatter_(-1, labels.unsqueeze(-1), 1.0)
    out = met.risk_metrics(probs, labels)
    assert abs(out['accuracy'] - 1.0) < 1e-6
    assert abs(out['macro_f1'] - 1.0) < 1e-6
    assert abs(out['extreme_recall'] - 1.0) < 1e-6


# ------------------------------------------------- 8. 消融开关
# ------------------------------------------------- 式(18) 的预条件
def _precond_model(mode, t=96):
    sc = scenario()
    m = build_model(sc, TrainConfig(seq_len=t, stride=8), _tensors())
    m.cfg.precond = mode
    m.eval()
    return m, sc


def test_gradient_is_orthogonal_to_observation_null_space():
    """式(18) 的梯度恒正交于观测算子 A 的零空间：A v = 0 时 g . v = 0。

    这是完整求解优于对角近似的根据——零点在零空间内，故零空间方向既无
    梯度也无步长，不会被虚假更新。
    """
    m, sc = _precond_model('none')
    t = 96
    n0 = torch.zeros(1, t, sc.n_true.shape[1])
    with torch.no_grad():
        gv, _ = m._video_grad(n0, sc.y_vis[:t].unsqueeze(0),
                              sc.m_vis[:t].unsqueeze(0),
                              sc.o_vis[:t].unsqueeze(0))
        # 视频通道的 A = M H：缺口区的基向量属于零空间
        gap = (~(m.H.abs().sum(0) > 0))
        assert bool(gap.any()), '本场景应存在视频不可观测的缺口区'
        v = torch.zeros_like(gv)
        v[..., gap] = 1.0
        inner = (gv * v).sum(dim=-1).abs().max()
        # 缺口区列在 H 中恒为 0，故该方向上的梯度逐元素为零而非仅正交
        assert float(gv[..., gap].abs().max()) == 0.0, \
            '缺口区在视频通道上的梯度应为零，实测最大 %.3e' % \
            float(gv[..., gap].abs().max())
        assert float(inner) == 0.0
        assert v.abs().sum() > 0


def test_full_solve_gives_exact_newton_step_on_one_hot_operator():
    """A 为选择矩阵时，(H^T Omega^2 H)^{-1} g 应精确等于 n - n*。

    以锚点通道为对象：P 是 0/1 选择矩阵，有观测处的 Hessian 为 4x4 的对角阵，
    故两种预条件方式都应给出精确的牛顿步。
    """
    m, sc = _precond_model('full')
    t = 96
    n0 = torch.zeros(1, t, sc.n_true.shape[1])
    with torch.no_grad():
        ga, _ = m._anchor_grad(n0, sc.y_anc[:t].unsqueeze(0),
                               sc.anc_mask[:t].unsqueeze(0))
        resid = (n0 @ m.P.t() - sc.y_anc[:t].unsqueeze(0))
        # 把锚点空间的残差经 P 散回区域空间，才能与区域空间的梯度逐元素比
        resid_region = resid @ m.P
        act = (sc.anc_mask[:t].unsqueeze(0) @ m.P) > 0
        assert bool(act.any()), '本场景应有可用的锚点观测'
        # 有观测处的牛顿步应等于残差本身（即 n - n*）。阻尼 lambda 会带来
        # 量级为 eps_curv 的**相对**偏差，故按相对量设定判据；残差本身在
        # 10^3 量级，用绝对容差会把这一已知偏差误判为缺陷。
        scale = resid_region[act].abs().max().clamp(min=1e-12)
        rel = (ga[act] - resid_region[act]).abs().max() / scale
        assert float(rel) < 10.0 * m.cfg.eps_curv, \
            '锚点通道的完整求解应等于残差，相对偏差 %.3e 超过阻尼量级' \
            % float(rel)
        assert ga[act].abs().max() > 0


def test_full_solve_recovers_the_state_scale_where_diagonal_overshoots():
    """完整求解把信令通道折算到 n - n* 的量级，对角近似则放大约 7 倍。

    信令算子为 B diag(pi)，B 为 (7, 18) 且列和为 1，故对角元正比于
    sum_l B[l,k]^2 = 1/7 而分子正比于 sum_l B[l,k] = 1。
    """
    sc = scenario()
    t = 96
    n0 = torch.zeros(1, t, sc.n_true.shape[1])
    pi = sc.pi_true[:t].unsqueeze(0)
    target = (n0 - sc.n_true[:t].unsqueeze(0)).abs().mean()

    vals = {}
    for mode in ('full', 'diag'):
        m, _ = _precond_model(mode)
        with torch.no_grad():
            res, _ = m._signal_residual(n0, pi, sc.y_sig[:t].unsqueeze(0))
            gs, _ = m._signal_grad(pi, res)
        vals[mode] = float(gs.abs().mean())

    assert abs(vals['full'] / float(target) - 1.0) < 0.15, \
        '完整求解应落在 n - n* 的量级上，实测比值 %.3f' % \
        (vals['full'] / float(target))
    assert vals['diag'] > 5.0 * vals['full'], \
        '对角近似应显著偏大，实测 full=%.1f diag=%.1f' % \
        (vals['full'], vals['diag'])


def test_precond_none_is_literal_eq18():
    """precond='none' 必须原样返回式(18) 的梯度，供对照实验使用。"""
    sc = scenario()
    t = 96
    n0 = torch.zeros(1, t, sc.n_true.shape[1])
    out = {}
    for mode in ('none', 'diag', 'full'):
        m, _ = _precond_model(mode)
        with torch.no_grad():
            gv, _ = m._video_grad(n0, sc.y_vis[:t].unsqueeze(0),
                                  sc.m_vis[:t].unsqueeze(0),
                                  sc.o_vis[:t].unsqueeze(0))
        out[mode] = gv
    assert out['none'].abs().mean() < out['diag'].abs().mean(), \
        '未预条件的梯度应远小于预条件后的梯度'
    assert not torch.allclose(out['diag'], out['full']), \
        '对角近似与完整求解的结果不应相同'


def test_preconditioner_keeps_gap_regions_at_zero():
    """缺口区在两通道的算子中列恒为 0，预条件后仍应为 0，不得凭空产生更新。"""
    sc = scenario()
    t = 96
    n0 = torch.zeros(1, t, sc.n_true.shape[1])
    for mode in ('diag', 'full'):
        m, _ = _precond_model(mode)
        gap = ~(m.H.abs().sum(0) > 0)
        if not bool(gap.any()):
            return
        with torch.no_grad():
            gv, _ = m._video_grad(n0, sc.y_vis[:t].unsqueeze(0),
                                  sc.m_vis[:t].unsqueeze(0),
                                  sc.o_vis[:t].unsqueeze(0))
        assert float(gv[..., gap].abs().max()) == 0.0, \
            'precond=%s 时缺口区出现了非零视频更新' % mode


def test_precond_rejects_unknown_mode():
    """预条件方式写错时必须报错，否则会静默退回某个分支。"""
    m, _ = _precond_model('full')
    m.cfg.precond = 'ful'
    t = 96
    n0 = torch.zeros(1, t, 18)
    try:
        m._video_grad(n0, torch.zeros(1, t, 12), torch.ones(1, t, 12),
                      torch.zeros(1, t, 12))
    except (ValueError, KeyError):
        return
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError('应抛出 ValueError，实际 %s' % type(exc).__name__)
    raise AssertionError('未知的 precond 取值应抛出 ValueError')


def test_unknown_ablation_switch_is_rejected():
    """消融开关名写错时必须报错，否则实验会静默地跑成完整模型。"""
    sc = scenario()
    model = build_model(sc, TrainConfig(seq_len=96, stride=8), _tensors())
    try:
        model.set_ablation(use_vido=False)
    except ValueError:
        return
    raise AssertionError('未知消融开关应抛出 ValueError')


def test_ablation_switches_change_behaviour():
    """每个消融开关都应改变前向输出，否则该开关未接入计算图。"""
    sc = scenario()
    tensors = _tensors()
    batch = _batch(2, 96)
    model = build_model(sc, TrainConfig(seq_len=96, stride=8), tensors)
    base = model(*batch)
    ref_n, ref_p = base['n_hat'], base['probs']

    all_switches = ['use_video', 'use_signal', 'use_anchor', 'use_penetration',
                    'use_graph', 'use_temporal', 'use_gate', 'use_external',
                    'use_conservation']
    for sw in all_switches:
        model.set_ablation(**{k: (k != sw) for k in all_switches})
        out = model(*batch)
        changed = (float((out['n_hat'] - ref_n).abs().max()) > 1e-6
                   or float((out['probs'] - ref_p).abs().max()) > 1e-6)
        assert changed, '关闭 %s 后输出未变化，该开关可能未接入' % sw
    model.set_ablation(**{k: True for k in all_switches})


def test_full_model_is_the_default():
    """默认配置必须是完整模型，消融开关全部为真。"""
    c = HOINetConfig()
    for f in ('use_video', 'use_signal', 'use_anchor', 'use_penetration',
              'use_graph', 'use_temporal', 'use_gate', 'use_external',
              'use_conservation'):
        assert getattr(c, f) is True, '%s 的默认值应为 True' % f
    assert c.enabled_channels() == ['video', 'signal', 'anchor']


# ------------------------------------------------- 9. 端到端评估可跑通
def test_evaluate_returns_all_metric_groups():
    """5.5 节的评估口径应同时给出状态、风险与校准三组指标。"""
    sc = scenario()
    tensors = _tensors()
    model = build_model(sc, TrainConfig(seq_len=96, stride=8), tensors)
    loaders, _, _ = ds.make_loaders(sc, torch.device('cpu'), seq_len=96,
                                    batch_size=2, stride=48)
    out = evaluate(model, loaders['test'], tensors, LossWeights())
    for key in ('state', 'risk', 'calibration'):
        assert key in out, '评估结果缺少 %s' % key
    for key in ('overall', 'covered', 'gap', 'gap_over_covered'):
        assert key in out['state'], '状态指标缺少 %s' % key
    for key in ('accuracy', 'macro_f1', 'weighted_f1', 'extreme_recall'):
        assert key in out['risk'], '风险指标缺少 %s' % key
    for key in ('brier', 'ece', 'mce'):
        assert key in out['calibration'], '校准指标缺少 %s' % key
    assert out['calibration']['mce'] >= out['calibration']['ece'] - 1e-9


# ------------------------------------------------- 10. 配置文件
def test_default_config_matches_dataclass_defaults():
    """configs/default.yaml 的取值应与各 dataclass 的默认值一致。"""
    from src import config as cfgmod
    path = cfgmod.DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return
    sc, model, train, loss = cfgmod.load_config(path)
    d_sc, d_model, d_train, d_loss = cfgmod.load_config(None)
    import dataclasses
    for name, a, b in (('ScenarioConfig', sc, d_sc),
                       ('TrainConfig', train, d_train),
                       ('LossWeights', loss, d_loss)):
        for f in dataclasses.fields(a):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            assert va == vb, '%s.%s 配置文件为 %r，默认值为 %r' % (
                name, f.name, va, vb)
    # n_regions / n_cameras / n_sectors / n_ext / n_classes / n_anchors 不是
    # 模型的自由超参数，而是由场景规模继承而来（见 src/config.py），故不
    # 参与“与默认值一致”的比较；改为检查它们确实等于场景派生的取值。
    inherited = ('n_regions', 'n_cameras', 'n_sectors', 'n_ext',
                 'n_classes', 'n_anchors')
    assert model.n_anchors == sc.n_anchors_gate + sc.n_anchors_manual
    for f in dataclasses.fields(d_model):
        if f.name.startswith('use_') or f.name in inherited:
            continue                    # 消融开关不在配置文件中
        va, vb = getattr(model, f.name), getattr(d_model, f.name)
        assert va == vb, 'HOINetConfig.%s 配置文件为 %r，默认值为 %r' % (
            f.name, va, vb)


def test_config_rejects_unknown_field():
    """配置文件中的拼写错误必须报错，不能静默忽略。"""
    from src import config as cfgmod
    try:
        cfgmod.build_config({'train': {'learnig_rate': 1e-4}})
    except ValueError:
        return
    raise AssertionError('未知配置字段应抛出 ValueError')


# ------------------------------------------------- 运行入口
def _main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print('  PASS  %s' % name)
        except Exception as exc:                                # noqa: BLE001
            failed.append((name, exc))
            print('  FAIL  %s\n        %s: %s'
                  % (name, type(exc).__name__, exc))
    print('\n%d 项通过，%d 项失败（共 %d 项）'
          % (len(tests) - len(failed), len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
