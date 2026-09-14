# HOINet

HOINet 把景区人流风险评估拆成两个问题：潜在状态的反演，与风险概率的读出。
状态量取区域人数而非密度，使信令的空间混合观测具有可加形式；反演过程由近端
梯度法的展开迭代构造，视频、信令与锚点三条观测通道的残差反投影分别对应一条
梯度通道。

## 目录结构

```
src/data/       数据层：结构矩阵、观测、标签、场景生成
  synthesize.py   合成场景生成（对应论文 5.1 节的数据描述）
  labels.py       式(1)(3) 密度与拥堵持续状态、阈值与风险等级
  video.py        式(6)—式(8)  带掩码的视频局部人数观测
  signaling.py    式(9)        空间混合与时间聚合的信令观测
  anchors.py      式(11)       锚点观测、可用性掩码与渗透率标定
  graph.py        式(12)       功能连通图与图拉普拉斯
  dataset.py      划分训练/验证/测试并滑窗
src/model/      模型层
  hoi_net.py      HOINet 主体：展开层、三通道反投影、门控、渗透率更新
  ops.py          切比雪夫图卷积、因果时间卷积、单调读出、软阈值
  losses.py       式(23)—式(26) 训练目标
  encoder.py      视频侧的视觉编码器
  engine.py       训练循环、评估与多种子重复实验
src/evals/      评估层
  metrics.py      状态反演指标（整体/覆盖区/缺口区）与风险分级指标
  calibration.py  ECE 与 MCE 校准指标
src/config.py   配置文件到各 dataclass 配置对象的映射
configs/        实验配置
  default.yaml    论文 5.1 节与 5.4 节的规模与超参数
scripts/        命令行入口
  make_synth_data.py   生成合成数据
  train.py             训练
  evaluate.py          评估
  inference.py         推理（无标签，只产出预测）
  run_ablation.py      论文 6.5 节表 8 的消融实验
  run_robustness.py    论文 6.6 节表 10—表 12 的观测缺陷实验
tests/          论文中可检验的不变量与结构约束
```

## 快速开始

```
pip install -r requirements.txt

python scripts/make_synth_data.py --out data/scenic_mm_synth.pt
python scripts/train.py --data data/scenic_mm_synth.pt --epochs 100
python scripts/evaluate.py --data data/scenic_mm_synth.pt --ckpt runs/hoinet/model.pt
```

默认超参数全部对应论文 5.4 节：Adam，学习率 1e-4，余弦退火，批大小 8，
最多 100 轮，验证损失连续 10 轮不降即早停，5 个随机种子报告均值与标准差。

```
python scripts/train.py --data data/scenic_mm_synth.pt --seeds 0 1 2 3 4
```

仅需验证流程是否跑通时，用较小的场景与较少的轮数即可：

```
python scripts/make_synth_data.py --days 6 --out data/small.pt
python scripts/train.py --data data/small.pt --epochs 5
```

## 参数说明

[`configs/default.yaml`](configs/default.yaml) 是全部参数的集中入口，取值与论文
一致；不提供配置文件时，各 dataclass 的默认值与之一致。下表按论文中的符号列出
取值与其在仓库中的位置。

### 场景规模（论文 5.1 节表 2）

| 符号 | 含义 | 取值 | 配置项 |
|------|------|------|--------|
| $N$ | 子区域数 | 18 | `scenario.n_regions` |
| $M$ | 摄像头路数 | 12 | `scenario.n_cameras` |
| — | 视频覆盖区 / 缺口区 | 12 / 6 | `src/data/video.py:gap_regions` |
| — | 摄像头有效覆盖面积占比 | 均值 41.6% | `src/data/synthesize.py` 的覆盖矩阵 |
| $L$ | 信令单元（基站扇区）数 | 7 | `scenario.n_sectors` |
| $A$ | 锚点点位数 | 3 个出入口闸机 + 2 个人工抽样点位 | `scenario.n_anchors_gate` / `n_anchors_manual` |
| — | 锚点观测次数 | 360（60 天 × 3 出入口 × 2 次） | `src/data/anchors.py` |
| $E$ | 外部变量维度 | 8 | `scenario.n_ext` |
| $K$ | 风险等级数 | 4（低/中/高/极高） | `scenario.n_classes` |
| — | 类别占比 | 45% / 30% / 18% / 7% | `scenario.target_ratios` |
| — | 连续观测天数 | 60 | `scenario.days` |
| — | 视频采样间隔 | 60 s | `src/data/video.py` |
| — | 信令聚合窗口 | 15 min | `model.window_minutes` |
| $T$ | 序列长度 | 96（15 min 步长，对应 24 h） | `model.seq_len` |

这些取值经 [`scripts/make_synth_data.py`](scripts/make_synth_data.py) 传给
`src/data/synthesize.py`，用于生成与表 2 结构一致的景区多模态场景。

### 模型结构（论文 3.5 节与 5.4 节）

| 符号 | 含义 | 取值 | 位置 |
|------|------|------|------|
| — | 视觉编码器卷积核 | 3×3 / 5×5 / 7×7 | `VisualConfig.branch_kernels` |
| — | 视觉编码器通道数 | 64 / 128 / 256 | `VisualConfig.branch_channels` |
| — | 三分支输出空间尺寸 | 输入的 1/4、1/8、1/16 | `src/model/encoder.py` |
| $d_v$ | 融合后视觉特征维度 | 256 | `VisualConfig.d_visual` |
| $N_s$ | 每个基本步内抽取的视频子步数 | 30 | `src/data/video.py` |
| $d$ | 反演层嵌入维度 | 256 | `model.d_embed` |
| $K$ | 切比雪夫阶数 | 3 | `model.cheb_k` |
| $K_1, K_2$ | 时间卷积核大小 | 3, 5 | `model.tcn_kernels` |
| $S$ | 展开阶段数 | 4 | `model.n_stages` |
| — | 空间传播与时间补全算子 | 参数在 $S$ 个阶段间共享 | `src/model/ops.py` |
| $d_\pi$ | 渗透率低秩维度 | 8 | `model.pi_rank` |
| $d_e$ | 环境特征维度 | 8 | `HOINetConfig.n_ext` |
| — | 风险读出头 | 两层 MLP，隐藏层 128 | `model.readout_hidden` |

视觉分支的参数集中在 `src/model/encoder.py:VisualConfig`，不在 `configs/default.yaml`
的 `model` 段内，修改需直接改该 dataclass。

### 训练设置（论文 5.4 节）

| 含义 | 取值 | 配置项 |
|------|------|--------|
| 优化器 | Adam | `src/model/engine.py` |
| 初始学习率 | $1 \times 10^{-4}$ | `train.lr` |
| 学习率调度 | 余弦退火 | `train.eta_min_ratio` |
| 批大小 | 8 | `train.batch_size` |
| 训练轮数 | 100 | `train.max_epochs` |
| 早停耐心值 | 10 | `train.patience` |
| 随机种子 | 5 个，报告均值 ± 标准差 | `scripts/train.py --seeds` |

论文报告的运行环境为 NVIDIA RTX 4090（24 GB）、Python 3.9 与 PyTorch 2.1.0。
本仓库只依赖 `torch` 与 `PyYAML`，全部脚本在 CPU 上即可运行。

### 损失权重

| 论文符号 | 含义 | 取值 | 配置项 |
|----------|------|------|--------|
| $\mu_1$ | 风险分类损失权重 | 0.1 | `loss.lambda_risk` |
| $\mu_2$ | 观测一致性权重 | 0.05 | `loss.lambda_obs` |
| $\mu_3$ | 跨模态一致性权重 | $1 \times 10^{-4}$ | `loss.lambda_cons` |
| $\beta$ | 式(24) 密度项系数 | 1.0（论文未列取值） | `loss.beta` |
| $\lambda_{\mathrm{rel}}$ | 式(24) 相对误差项权重 | 0.1（论文未列取值） | `loss.lambda_rel` |
| $\lambda_{\mathrm{ord}}$ | 式(25) 序数损失权重 | 0.1（论文未列取值） | `loss.lambda_ord` |
| $\varepsilon$ | 式(24) 相对误差分母的稳定项 | $10^{-3}$（论文未列取值） | `loss.eps` |

## 消融与鲁棒性实验

```
python scripts/run_ablation.py --data data/scenic_mm_synth.pt \
    --seeds 0 1 2 3 4 --internal --out runs/ablation
python scripts/run_robustness.py --data data/scenic_mm_synth.pt \
    --seeds 0 1 2 3 4 --out runs/robustness
```

消融脚本按论文表 8 的勾选列逐行设置 `use_*` 开关，报告准确率与宏平均 F1；
鲁棒性脚本覆盖表 10 的信令时间戳平移、表 11 的渗透率扰动与表 12 的锚点
稀疏化，三者都只在测试区间注入缺陷，训练与验证区间保持干净。

## 测试

```
python tests/test_invariants.py     # 也可用 pytest tests/ 收集
```

## 式(18) 的预条件

论文式(18) 给出的是数据项梯度，其量级由观测精度 `Omega^2` 决定，比待修正
的状态量小四个到十个数量级；式(20) 只有一个可学习步长供三条通道共用，无法
同时补偿三个不同的量级，故预条件是把式(18) 与式(20) 接通的前提。

实现取带阻尼的完整 Hessian 求解 `(Hessian + lambda I)^{-1} g`，其依据是
式(18) 的梯度恒正交于观测算子的零空间，因而求解结果在可辨识子空间内是精确
的牛顿步、在零空间内为零。`HOINetConfig.precond` 提供三种方式：

```
precond: full    # 默认，完整求解
precond: diag    # 对角（Jacobi）近似，作对照
precond: none    # 不做预条件，即式(18) 原样
```

对角近似在稠密观测算子下偏差较大——信令通道的 `B` 为 (7, 18) 且列和为 1，
对角近似会把该通道放大约 10 倍，完整求解则落在参照量的 0.999 倍。

## 引用

若本仓库对您的工作有帮助，请引用论文原文。
