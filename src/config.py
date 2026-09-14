# -*- coding: utf-8 -*-
"""把 YAML 配置文件读成各个 dataclass 配置对象。

论文 5.1 节与 5.4 节给出了完整的场景规模、模型结构与训练超参数。这些量在
代码中由三个 dataclass 承载：

    data.synthesize.ScenarioConfig     场景规模，论文 5.1 节
    model.hoi_net.HOINetConfig         模型结构，论文 3.5 节与 5.4 节
    model.engine.TrainConfig           训练设置，论文 5.4 节
    model.losses.LossWeights           式(23)—式(26) 的权重

本模块提供从 configs/default.yaml 到上述对象的映射，使超参数既可以直接用
dataclass 默认值复现，也可以通过配置文件集中修改。

字段名的对应关系是显式的：配置文件里的键名与 dataclass 字段名相同，只有
少量键为了可读性做了改名，记在 _SCENARIO_ALIASES 与 _LOSS_ALIASES 中。
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional

from .data.synthesize import ScenarioConfig
from .model.engine import TrainConfig
from .model.hoi_net import HOINetConfig
from .model.losses import LossWeights

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'configs', 'default.yaml')

# 配置文件的键名 -> dataclass 字段名
_SCENARIO_ALIASES = {'n_anchors_gate': 'n_anchors_gate'}   # 同名，仅为显式列出
_LOSS_ALIASES: Dict[str, str] = {}                          # 全部同名

# 配置文件的顶层段 -> dataclass
_SECTIONS = {
    'scenario': ScenarioConfig,
    'train': TrainConfig,
    'loss': LossWeights,
}


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。

    只依赖 PyYAML；未安装时给出明确的安装提示，而不是抛出难以定位的
    ImportError。
    """
    try:
        import yaml
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            '读取配置文件需要 PyYAML，请先执行 pip install -r requirements.txt'
        ) from exc
    if not os.path.exists(path):
        raise FileNotFoundError('配置文件不存在：%s' % path)
    with open(path, 'r', encoding='utf-8') as f:
        blob = yaml.safe_load(f) or {}
    if not isinstance(blob, dict):
        raise ValueError('配置文件的顶层应为映射，实际为 %s' % type(blob).__name__)
    return blob


def _build(cls, values: Optional[Dict[str, Any]], section: str,
           aliases: Optional[Dict[str, str]] = None):
    """用一段配置构造 dataclass，未知键直接报错。

    静默忽略未知键会让配置文件的拼写错误变成难以察觉的行为差异，因此这里
    显式检查：配置文件里出现 dataclass 没有的字段即报错。
    """
    if not values:
        return cls()
    aliases = aliases or {}
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, val in values.items():
        name = aliases.get(key, key)
        if name not in fields:
            raise ValueError(
                '配置段 [%s] 中存在未知字段 %r；%s 可用的字段为 %s'
                % (section, key, cls.__name__, sorted(fields)))
        kwargs[name] = val
    # target_ratios 在 YAML 中是列表，dataclass 声明为 tuple
    if 'target_ratios' in kwargs and isinstance(kwargs['target_ratios'], list):
        kwargs['target_ratios'] = tuple(kwargs['target_ratios'])
    if 'tcn_kernels' in kwargs and isinstance(kwargs['tcn_kernels'], list):
        kwargs['tcn_kernels'] = tuple(kwargs['tcn_kernels'])
    return cls(**kwargs)


def build_config(cfg: Dict[str, Any]):
    """由已读入的配置字典构造四个配置对象。

    Returns:
        (scenario, model, train, loss)
    """
    scenario = _build(ScenarioConfig, cfg.get('scenario'), 'scenario',
                      _SCENARIO_ALIASES)
    loss = _build(LossWeights, cfg.get('loss'), 'loss', _LOSS_ALIASES)

    model_cfg = cfg.get('model') or {}
    # HOINetConfig 的区域数等结构量必须与场景一致，配置文件里只写模型特有的
    # 部分（阶段数、嵌入维度等），其余从场景继承，避免两处取值不一致。
    model_fields = {f.name for f in dataclasses.fields(HOINetConfig)}
    inherited = {'n_regions': scenario.n_regions,
                 'n_cameras': scenario.n_cameras,
                 'n_sectors': scenario.n_sectors,
                 'n_ext': scenario.n_ext,
                 'n_classes': scenario.n_classes,
                 'n_anchors': scenario.n_anchors_gate + scenario.n_anchors_manual}
    merged = dict(inherited)
    for key, val in model_cfg.items():
        if key not in model_fields:
            raise ValueError('配置段 [model] 中存在未知字段 %r；%s 可用的字段为 %s'
                             % (key, 'HOINetConfig', sorted(model_fields)))
        merged[key] = tuple(val) if key == 'tcn_kernels' and isinstance(val, list) \
            else val
    if 'seq_len' not in model_cfg:
        merged['seq_len'] = (cfg.get('train') or {}).get('seq_len', 96)
    model = HOINetConfig(**merged)

    train = _build(TrainConfig, cfg.get('train'), 'train')
    if not (cfg.get('train') or {}).get('seq_len'):
        train.seq_len = model.seq_len
    return scenario, model, train, loss


def load_config(path: Optional[str] = None):
    """读取配置文件并构造四个配置对象；path 为 None 时用配置文件默认值。

    Returns:
        (scenario, model, train, loss)，path 为 None 时四个对象均取 dataclass
        的默认值，即论文 5.1 节与 5.4 节的设置。
    """
    if path is None:
        return (ScenarioConfig(), HOINetConfig(), TrainConfig(), LossWeights())
    return build_config(load_yaml(path))


def split_days(cfg: Dict[str, Any], days: int):
    """由配置文件的 splits 段给出三个划分区间的天数，并校验总和。

    论文 5.2 节的数据划分为 42 / 9 / 9 天，对应 days=60 时的 70% / 15% / 15%。
    天数与总天数不符时按比例缩放，保证三个区间首尾相接且不超出场景长度。
    """
    sp = cfg.get('splits') or {}
    tr = int(sp.get('train_days', 42))
    va = int(sp.get('val_days', 9))
    te = int(sp.get('test_days', 9))
    total = tr + va + te
    if total == days:
        return tr, va, te
    if total <= 0:
        raise ValueError('splits 段的天数之和必须为正')
    # 按配置的比例重新分配到 days 天上，至少各留 1 天
    scale = days / float(total)
    tr2 = max(1, int(round(tr * scale)))
    va2 = max(1, int(round(va * scale)))
    te2 = max(1, days - tr2 - va2)
    return tr2, va2, te2
