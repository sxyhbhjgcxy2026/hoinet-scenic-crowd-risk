# -*- coding: utf-8 -*-
"""数据侧：图结构、信令、视频、锚点、标签与合成数据。

模块与论文小节的对应：
    graph.py       3.4 节      功能区邻接与图拉普拉斯
    signaling.py   3.3.2 节    式(9) 信令的时空聚合与脱敏
    video.py       3.3.1 节    式(6)(7) 视频覆盖与异方差噪声
    anchors.py     3.3.3 节    式(11) 锚点观测与渗透率标定
    labels.py      3.1 节      式(3) 拥堵持续状态与风险阈值
    synthesize.py  5.1 节      按 Scenic-MM 结构生成合成数据
    dataset.py     5.2 节      42/9/9 天划分与序列滑窗
"""

from .dataset import (ScenarioTensors, ScenicWindowDataset, SplitIndex,
                      collate, default_splits, make_loaders)
from .synthesize import Scenario, ScenarioConfig, build_scenario, describe

__all__ = ['Scenario', 'ScenarioConfig', 'build_scenario', 'describe',
           'ScenicWindowDataset', 'ScenarioTensors', 'SplitIndex',
           'default_splits', 'make_loaders', 'collate']
