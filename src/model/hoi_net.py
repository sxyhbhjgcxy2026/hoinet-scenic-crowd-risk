# -*- coding: utf-8 -*-
"""HOINet 是论文第 3 章的直接实现。整体结构与公式的对应关系：

    式(9)   信令时空聚合观测        -> _signal_residual()
    式(10)  渗透率有界参数化        -> _penetration()
    式(11)  锚点观测                -> _anchor_grad()
    式(13)  反演目标（网络结构的唯一来源）
    式(14)  视频数据项              -> _video_grad()
    式(15)  信令数据项              -> _signal_residual() 的平方和
    式(16)  锚点数据项              -> _anchor_grad()
    式(17)  信令重构残差            -> _signal_residual()
    式(18)  三通道反投影梯度        -> _video_grad / _signal_grad / _anchor_grad
    式(19)  观测精度门控            -> _gates()
    式(20)  梯度步                  -> 前向循环内的 n_tilde
    式(21)  图卷积 / 时间卷积 / 非负投影 -> 前向循环内的 n_next
    式(22)  渗透率更新              -> 前向循环内的 eta 更新
    式(3)   拥堵持续状态            -> cum_dwell_state()
    式(5)   风险概率读出            -> MonotoneReadout

"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import ChebConv, MonotoneReadout, TemporalConv


def _inv_softplus(x: float, eps: float = 1e-12) -> float:
    """softplus 的反函数，数值稳定版：返回 y 使 softplus(y) = x。

    直接写 log(expm1(x)) 在 x 较大时溢出——观测方差与均值之比在本数据集上
    约为 10^3，expm1(10^3) 超出双精度范围。x 较大时 softplus(y) = y 已足够
    精确，直接用 x 即可。
    """
    x = max(float(x), eps)
    return x if x > 20.0 else math.log(math.expm1(x))


@dataclass
class HOINetConfig:
    """模型与观测结构的配置。

    实验章节给出的取值写在行末注释中（论文 5.4 节／5.1 节）。
    """

    n_regions: int = 18              # N，景区子区域数
    n_cameras: int = 12              # M，摄像头路数
    n_sectors: int = 7               # L，信令单元（基站扇区）数
    n_windows: int = 6               # J，信令窗口数（每窗口 15 min，T=96 -> J=6）
    n_anchors: int = 3               # A，锚点点位数（3 个出入口闸机）
    n_ext: int = 8                   # 外部变量维度
    n_classes: int = 4               # K，风险等级（低/中/高/极高）

    n_stages: int = 4                # S，展开阶段数
    seq_len: int = 96                # T，序列长度（15 min 步长，96 步 = 24 h）
    d_embed: int = 256               # d，反演层嵌入维度
    cheb_k: int = 3                  # 切比雪夫阶数
    tcn_kernels: tuple = (3, 5)      # 时间卷积核大小 K1、K2
    pi_rank: int = 8                 # 渗透率低秩维度 dz
    pi_min: float = 0.05             # 渗透率物理下界
    pi_max: float = 0.85             # 渗透率物理上界
    readout_hidden: int = 128        # 风险读出头隐藏层维度
    eps_curv: float = 1e-6
    """式(18) 预条件的相对阻尼系数，见 HOINet._precondition。

    阻尼取为本通道 Hessian 对角元最大值的 eps_curv 倍，即 Levenberg-Marquardt
    的缩放方式。三条观测通道的量级相差若干量级，因此阻尼必须是相对的而非
    绝对值；取相对量后可同时服务于对角近似与完整求解两种预条件方式。
    """

    precond: str = 'full'
    """式(18) 的预条件方式，取值 'full' / 'diag' / 'none'。

    'full' 对 Hessian 作带阻尼的完整求解，'diag' 只取其对角元作 Jacobi
    近似，'none' 即式(18) 原样不做任何预条件。默认为 'full'，理由见
    HOINet._precondition 的 docstring：对角近似在稠密混合算子下会放大约 7
    倍，而完整求解在可辨识子空间内给出精确的牛顿步、在零空间内给出零步，
    与论文命题 2 的可辨识性结论一致。'diag' 与 'none' 保留供对照。
    """

    window_minutes: int = 15         # 信令聚合窗口长度

    # ---- 消融开关：论文 6.5 节表 8 的通道与模块消融 ----
    # 默认全部为 True，即完整模型。置 False 表示关闭对应的观测通道或结构
    # 模块，用于复现表 8 的消融行。各项的含义与作用位置：
    #
    #   use_video      式(14)、式(18) 视频通道      -> _video_grad 不参与反投影
    #   use_signal     式(15)、式(18) 信令通道      -> _signal_residual 不参与
    #   use_anchor     式(16)、式(18) 锚点通道      -> _anchor_grad 不参与
    #   use_penetration 式(10)、式(22) 渗透率校准   -> pi 固定为 1 且不更新
    #   use_graph      式(21) 的图卷积分支          -> 近端算子去掉 GCN 项
    #   use_temporal   式(21) 的时间卷积分支        -> 近端算子去掉 TCN 项
    #   use_gate       式(19) 的观测精度门控        -> 三条通道等权
    #   use_external   读出层的外部协变量           -> 从 z_free 中去掉 ext
    #   use_conservation 读出层的人数守恒项 v       -> 从 z_free 中去掉 v
    #
    # 表 8 中"HOINet w/o anchor"一行同时关闭锚点与渗透率校准（论文正文
    # 明确说明该行"同时去掉了渗透率校准与锚点"），因此复现该行需要同时
    # 置 use_anchor=False 与 use_penetration=False。
    #
    # 表 8 中的 Feature fusion 一行是特征层融合的对照方法，对应论文表 5 的
    # MulT，不是本网络的模块消融，故不由开关表达。
    use_video: bool = True
    use_signal: bool = True
    use_anchor: bool = True
    use_penetration: bool = True
    use_graph: bool = True
    use_temporal: bool = True
    use_gate: bool = True
    use_external: bool = True
    use_conservation: bool = True

    def enabled_channels(self) -> list:
        """当前启用的观测通道名，顺序与式(18) 的三条通道一致。"""
        names = []
        if self.use_video:
            names.append('video')
        if self.use_signal:
            names.append('signal')
        if self.use_anchor:
            names.append('anchor')
        return names


class HOINet(nn.Module):
    """深度展开的异构观测反演网络。

    Args:
        cfg: HOINetConfig。
        coverage: (M, N) 摄像头到区域的有效覆盖比例 H，式(6)。
        mixing: (L, N) 基站扇区到区域的空间混合矩阵 B，式(9)。
        anchor_sel: (A, N) 锚点选择矩阵 P，式(11)。
        areas: (N,) 各区域有效面积 s_k，式(1)。
        laplacian: (N, N) 已缩放的图拉普拉斯 L~，由 data.graph.build_laplacian 得到。

    Note:
        式(9) 的时间聚合矩阵 W 不在此处传入：它是数据组装的一部分，
        在 src/data/dataset.py 中已把信令观测摊平到基本步上。
    """

    def __init__(self, cfg: HOINetConfig,
                 coverage: torch.Tensor,
                 mixing: torch.Tensor,
                 anchor_sel: torch.Tensor,
                 areas: torch.Tensor,
                 laplacian: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        n, m, l, a = (cfg.n_regions, cfg.n_cameras, cfg.n_sectors,
                      cfg.n_anchors)

        # ---- 观测结构矩阵：由景区部署决定，不参与训练 ----
        # 式(9) 的三个结构量中，H（覆盖）、B（空间混合）、P（锚点选择）在模型内
        # 使用；时间聚合 W 在数据层使用（见 _signal_residual 的说明）。
        for name, mat, shape in (('coverage', coverage, (m, n)),
                                 ('mixing', mixing, (l, n)),
                                 ('anchor_sel', anchor_sel, (a, n))):
            if tuple(mat.shape) != shape:
                raise ValueError('%s 形状应为 %s，实际 %s'
                                 % (name, shape, tuple(mat.shape)))
        self.register_buffer('H', coverage.float())          # (M, N)
        self.register_buffer('Bmat', mixing.float())         # (L, N)
        self.register_buffer('P', anchor_sel.float())        # (A, N)
        self.register_buffer('areas', areas.float().clamp(min=1e-6))  # (N,)
        if laplacian is not None:
            self.register_buffer('L', laplacian.float())
        else:
            raise ValueError('必须提供 laplacian，'
                             '可用 data.graph.build_laplacian 从 spatial_adj 现算')

        # ---- 式(7) 视频观测方差：sigma^2 = s0^2 + s1^2 * y + s2^2 * o ----
        # 三个系数非负，用 softplus 参数化。初值按观测量的量级设定：人数观测
        # 为数百至数千，故方差应随观测值线性增长（s1 是相对误差项）。若把
        # s0、s1 初始化得过小，式(7) 给出的方差会远低于真实噪声，omega^2
        # 随之过大，式(18) 的反投影梯度被放大若干个量级，展开迭代在第一步
        # 就会把状态推到物理上不可能的量级。取 softplus(-2)=0.127、
        # softplus(0)=0.693 使 var 的量级与观测噪声相称。
        self.log_sigma0 = nn.Parameter(torch.tensor(-2.0))
        self.log_sigma1 = nn.Parameter(torch.tensor(-2.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))
        # 信令与锚点的精度同样可学习。论文给出了视频方差的具体形式（式(7)），
        # 未给出信令与锚点的噪声模型，此处按各信令单元一个精度、锚点一个精度处理。
        #
        # 初值必须与观测噪声的量级相称：信令观测是聚合后的设备数，量级可达
        # 数千，其噪声标准差约在几十的量级；若把 omega^2 初始化成 1（即认为
        # 噪声标准差为 1），式(15) 的数据项会比式(24) 的状态项大若干个数
        # 量级，式(23) 的各项权重随之失去意义，梯度裁剪也会把受监督的项
        # 完全压掉。锚点直接观测人数、计量误差为个位数，故其精度取 1 附近。
        self.log_omega_sig = nn.Parameter(torch.full((l,), -4.0))
        self.log_omega_anc = nn.Parameter(torch.tensor(0.0))

        # ---- 式(10) 渗透率的初始值：eta_t = U v_t + b + delta_t ----
        self.U = nn.Parameter(torch.randn(n, cfg.pi_rank) * 0.05)
        self.v_pi = nn.Parameter(torch.zeros(cfg.seq_len, cfg.pi_rank))
        self.b_pi = nn.Parameter(torch.zeros(n))
        self.delta_pi = nn.Parameter(torch.zeros(cfg.seq_len, n))

        # ---- 式(19) 门控：MLP(h_t) -> 3 -> Softmax ----
        self.gate_proj_state = nn.Linear(n, cfg.d_embed)
        self.gate_proj_ext = nn.Linear(cfg.n_ext, cfg.d_embed)
        self.gate_proj_res = nn.Linear(3, cfg.d_embed)
        self.gate_mlp = nn.Sequential(
            nn.Linear(cfg.d_embed, cfg.d_embed), nn.GELU(),
            nn.Linear(cfg.d_embed, 3))

        # ---- 式(20) 与式(22) 的可学习步长，每个阶段一个 ----
        # 初值取 softplus(-0.5) = 0.474，即每次展开走完当前误差的一半左右。
        # 梯度经对角预条件后（见 _precondition）已是“当前值减目标值”的量级，
        # 步长的合理范围因此是 (0, 1)：过小则收敛慢，过大则在前两个阶段就
        # 越过最优解并在后续阶段来回震荡。
        self.log_eta_step = nn.Parameter(torch.full((cfg.n_stages,), -0.5))
        self.log_gamma = nn.Parameter(torch.full((cfg.n_stages,), -0.5))

        # ---- 式(21) 近端算子替身的阻尼系数，每阶段一个 ----
        # 论文只说明用 GCN 与 TCN 近似近端算子，未给出该替身的参数化形式。
        # 本实现把它写成对输入的**有阻尼修正**而非自由加性项：
        #
        #     prox(n_tilde) = ReLU( n_tilde + beta * TCN(GCN(n_tilde)) )
        #
        # 这样当某区域没有任何观测通道提供梯度时（n_tilde = n），修正项虽然
        # 非零但被 beta 压住，迭代保持非扩张；若写成 n_tilde + TCN(GCN(n_tilde))，
        # 无观测区域也会被自由项逐阶段放大，实测四个阶段内最大估计值从
        # 1.4e4 涨到 7.0e4，超出真值上界数倍。初值 softplus(-2.25) = 0.105。
        self.log_beta_prox = nn.Parameter(torch.full((cfg.n_stages,), -2.25))

        # ---- 式(21) 的近端算子替身：参数在 S 个阶段间共享 ----
        self.gcn = ChebConv(n, cheb_k=cfg.cheb_k, in_dim=1, out_dim=1)
        self.tcn = TemporalConv(channels=1, kernels=cfg.tcn_kernels)

        # ---- 式(22) 的低秩与时间平滑模块 R_pi，论文按阶段给出，故每阶段一个 ----
        self.R_pi = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=3, padding=1) for _ in range(cfg.n_stages)])

        # ---- 式(3) 拥堵持续状态的遗忘系数 lambda_q ----
        self.logit_lambda_q = nn.Parameter(torch.tensor(1.0))   # sigmoid -> 0.73

        # ---- 净流入速率 v 的平滑系数（论文未给出 v 的观测方程，见 docs） ----
        self.logit_lambda_v = nn.Parameter(torch.tensor(1.0))

        # ---- 式(5) 风险读出：rho、q、dn 受单调约束，v 与 e 自由 ----
        self.readout = MonotoneReadout(n_monotone=3,
                                       n_free=1 + cfg.n_ext,
                                       n_classes=cfg.n_classes)

    # ------------------------------------------------------------------ 观测算子
    def set_ablation(self, **flags) -> 'HOINet':
        """设置论文 6.5 节表 8 的消融开关，返回自身以便链式调用。

        只接受 HOINetConfig 中 use_ 开头的布尔字段，拼写错误会直接报错，
        避免消融实验因开关名写错而静默地跑出与完整模型相同的结果。

            model.set_ablation(use_anchor=False, use_penetration=False)
        """
        valid = {f.name for f in dataclasses.fields(HOINetConfig)
                 if f.name.startswith('use_')}
        for k, v in flags.items():
            if k not in valid:
                raise ValueError('未知消融开关 %r；可用开关为 %s'
                                 % (k, sorted(valid)))
            setattr(self.cfg, k, bool(v))
        return self

    def _channel_keep(self, dtype=None, device=None) -> torch.Tensor:
        """三条观测通道的启用掩码，顺序与式(18) 一致：视频、信令、锚点。"""
        return torch.tensor([self.cfg.use_video, self.cfg.use_signal,
                             self.cfg.use_anchor],
                            dtype=dtype or torch.float32,
                            device=device)

    def ablation_tag(self) -> str:
        """当前消融配置的短标识，用于实验记录的行名。"""
        c = self.cfg
        off = []
        for name, label in (('video', '视频'), ('signal', '信令'),
                            ('anchor', '锚点'), ('penetration', '渗透率校准'),
                            ('graph', '空间图'), ('temporal', '时间模块'),
                            ('gate', '精度门控'), ('external', '外部协变量'),
                            ('conservation', '人数守恒')):
            if not getattr(c, 'use_' + name):
                off.append(label)
        if len(off) == 9:
            return '全部关闭'
        return '完整模型' if not off else '去掉' + '、'.join(off)

    @torch.no_grad()
    def set_observation_scales(self, video_mean: float, video_ms: float,
                               signal_ms: float, anchor_ms: float) -> None:
        """按观测的实际量级初始化式(7)、式(15)、式(16) 的观测精度。

        论文把 Omega 定义为观测精度的对角阵，即观测方差之逆。方差参数若只
        按“数量级 1”猜测，会与实际观测尺度相差数个数量级：本数据集中视频
        观测的量级约为 10^3、残差平方约为 10^6，而 sigma1 的初值使式(7) 的
        方差约为 30，于是式(14) 的数据项约为 10^7，是式(24) 状态项的 10^4
        倍。此时式(23) 的权重 mu1 与 mu2 形同虚设，训练完全由观测项主导，
        状态项与风险项的梯度可忽略。

        这里把方差尺度直接取为该类观测在训练集上的统计量，使初始状态下
        各项数据项与状态项量级相当，式(23) 的权重才具有论文所述的相对
        含义。真实部署时应当在训练前用一段标定数据做同样的估计。

        Args:
            video_mean: 视频观测 y^vis 的均值，用于把式(7) 的方差折算到
                        典型观测水平上。
            video_ms:   视频观测的均方值 E[y^2]，作为其方差尺度的估计。
            signal_ms:  信令观测的均方值。
            anchor_ms:  有效锚点观测的均方值。
        """
        eps = 1e-12
        # 式(7)：var = s0^2 + s1^2 * y + s2^2 * o。
        #
        # s1^2 必须是无量纲的过散布系数，而不是 E[y^2]/E[y]。后者量纲为 y，
        # 在本数据集上约为 1.5e3，于是 var = s1^2 * y 在 y ≈ E[y] 处约为
        # 1.5e3 * 1.5e3 ≈ 2e6、标准差约等于观测值本身，即隐式声明视频观测
        # 有 100% 的相对噪声。其后果是观测精度 omega^2 约为 1e-10，式(18)
        # 的视频反投影梯度比状态量级小四个数量级，式(20) 的可学习步长必须
        # 放大一万倍才能补偿，而 Adam 对单个标量参数每步只更新约 lr 的量，
        # 训练期无法收敛到该量级，反演人数因而停滞在零附近。
        #
        # 除以 E[y]^2 之后，s1^2 = E[y^2]/E[y]^2 = 1 + CV^2，即“过散布系数”，
        # 此时 s1^2 * y 是与计数型观测相符的散粒噪声项，s1^2 取 O(1)。这与
        # 式(7) 把方差写成“常数项 + 随人数增长项 + 随遮挡增长项”的形式一致。
        s1_sq = max(video_ms / max(video_mean ** 2, eps), eps)
        self.log_sigma0.data.fill_(_inv_softplus(1e-3))
        self.log_sigma1.data.fill_(0.5 * _inv_softplus(s1_sq))
        self.log_sigma2.data.fill_(_inv_softplus(1e-3))
        self.log_omega_sig.data.fill_(-0.5 * math.log(max(signal_ms, eps)))
        self.log_omega_anc.data.fill_(-0.5 * math.log(max(anchor_ms, eps)))

    @torch.no_grad()
    def set_penetration_init(self, pi_init: torch.Tensor) -> None:
        """把式(10) 的初始 eta 置为由锚点与信令联合估计的渗透率。

        论文 5.2 节说明渗透率的初值“由训练集锚点与信令联合估计”。式(10) 的
        参数化是 pi = pi_min + (pi_max - pi_min) * sigmoid(eta)，故需要把
        估计出的物理渗透率折算回 eta 空间再写入初值参数：

            frac = (pi - pi_min) / (pi_max - pi_min)
            eta  = logit(frac)

        低秩项 U v_pi 在初始化时为零（v_pi 初始化为零），因此把 eta 拆成
        “逐区域常数 b_pi”与“逐时刻偏差 delta_pi”两段即可精确表示该初值：

            b_pi     = mean_t eta_t
            delta_pi = eta_t - b_pi

        之后 v_pi 与 U 仍可学习，模型可在训练中偏离这一初值。

        Args:
            pi_init: (T, N) 物理渗透率的估计值，取值应落在 (pi_min, pi_max)
                     内；越界值会被截断到区间边缘。
        """
        span = self.cfg.pi_max - self.cfg.pi_min
        t_steps = min(int(pi_init.shape[0]), self.cfg.seq_len)
        frac = (pi_init[:t_steps].float() - self.cfg.pi_min) / span
        # 截断到开区间，避免 logit 发散
        frac = frac.clamp(min=1e-4, max=1.0 - 1e-4)
        eta = torch.log(frac / (1.0 - frac))
        eta = eta.to(self.b_pi.device)
        b = eta.mean(dim=0)
        self.b_pi.data.copy_(b)
        self.delta_pi.data.zero_()
        self.delta_pi.data[:t_steps] = eta - b.unsqueeze(0)

    def _video_grad(self, n: torch.Tensor, y_vis: torch.Tensor,
                    m_vis: torch.Tensor, o_vis: torch.Tensor):
        """式(14) 与式(18) 视频通道。

        Returns:
            grad: (B, T, N) 反投影梯度 H^T M^T Omega^2 (M H n - y)
            loss: 标量，式(14) 的视频数据项

        Note:
            loss 取均值是为了让式(23) 的各项权重可比；grad 按对角预条件
            （见 _precondition）缩放，使式(20) 的步长成为 O(1) 的修正系数。
        """
        s0 = F.softplus(self.log_sigma0)
        s1 = F.softplus(self.log_sigma1)
        s2 = F.softplus(self.log_sigma2)
        # 式(7)：方差随估计人数与遮挡程度增大
        var = s0 ** 2 + s1 ** 2 * y_vis.clamp(min=0) + s2 ** 2 * o_vis
        omega2 = 1.0 / var.clamp(min=1e-6)                     # (B, T, M)
        # 式(6)：带掩码的局部人数观测
        pred = m_vis * (n @ self.H.t())                        # (B, T, M)
        resid = pred - y_vis
        loss = (omega2 * resid ** 2).mean()
        # 式(18) 第一项：H^T M^T Omega^2 (M H n - y)，再作预条件
        raw = (m_vis * omega2 * resid) @ self.H                # (B, T, N)
        # Hess = H^T diag(m * omega^2) H，形状 (B, T, N, N)
        w = m_vis * omega2                                     # (B, T, M)
        hess = (self.H.t() * w.unsqueeze(-2)) @ self.H          # (B, T, N, N)
        grad = self._precondition(raw, hess)
        return grad, loss

    def _precondition(self, raw: torch.Tensor,
                      hess: torch.Tensor) -> torch.Tensor:
        """对反投影梯度作预条件：g <- (Hessian + lambda I)^{-1} g。

        按 cfg.precond 分三种方式：'full' 作完整求解，'diag' 只取 Hessian 的
        对角元作 Jacobi 近似，'none' 直接返回原梯度即式(18) 原样。

        Args:
            raw:  (B, T, N) 式(18) 的反投影梯度。
            hess: (B, T, N, N) 对应数据项的 Hessian（半正定）。

        **预条件并非可选项，而是式(18) 与式(20) 能够训练的前提。** 这一条
        是与论文的偏离，需明确说明。式(18) 的数据项梯度为

            g = A^T Omega^2 (A n - y),   A = M H（视频）、B diag(pi)（信令）、P（锚点）

        Omega^2 取自式(14) 的数据项尺度（见 set_observation_scales），其量级
        远小于 1。实测 n = 0 时三条通道原始梯度的均值为

            视频 2.0e-1       信令 1.9e-4       锚点 8.0e-7
            参照量 |n - n*| 约 3.2e3

        即原始梯度比待修正量小四个到十个数量级。式(20) 只有**一个**可学习步长
        step = softplus(log_eta_step) 供三条通道共用，其初值为 0.474；要补偿
        视频通道需 step 约为 7.6e3，即参数 log_eta_step 由 -0.5 增至约 7.6e3，
        而 Adam 对单个标量参数每步只移动约 lr = 1e-4，需要约 1e8 步，在论文
        5.4 节的 100 轮预算内不可能达到。三条通道所需的量级又互不相同，单一
        步长也无法同时补偿，仅靠式(20) 前的 Softmax 门控（取值在单纯形上、
        会饱和）无法弥合十万倍以上的比例差。故若不预条件，视频通道将独占
        梯度而信令与锚点通道近乎失效。

        ---- 为何取完整求解而非对角近似 ----

        本方法取 (Hess + lambda I)^{-1} g，其中 Hess = A^T Omega^2 A 为数据项
        Hessian，lambda 为阻尼。这一取法有一个精确性质，正是它优于对角近似
        的原因。设 v 属于 A 的零空间，即 A v = 0，则

            g . v = (Omega^2 (A n - y)) . (A v) = 0

        即**原始梯度恒正交于 A 的零空间**。于是解 (Hess + lambda I) g' = g 在
        可辨识子空间 range(A) 内给出精确的牛顿步，而在零空间内 g' 的步长为零，
        不会产生任何虚假更新。若 y = A n* 且 n 与 n* 之差落在 range(A) 内，
        range(A) 上的步长恰为 n - n*，式(20)

            n <- n - eta * g'

        即退化为 n <- n - eta (n - n*)，eta 取 (0, 1) 内即可稳定收敛。这正是
        可学习步长容易学到的量级，也是式(20) 设计该步长的本意。

        对角近似 g_k / Hess_kk 不具备这一性质，在稠密观测算子下偏差可以很大。
        实测信令通道：B 为 (7, 18) 且列和为 1（见 test_mixing_column_sums_
        are_one），每个区域分属 7 个信令单元，对角近似使其预条件后的梯度为
        31051，约为参照量 3.2e3 的 10 倍；原因是用于相除的对角元正比于
        sum_l B[l,k]^2 = 1/7，而分子正比于 sum_l B[l,k] = 1，相差 7 倍。改为
        完整求解后该通道回落到其秩所允许的量级。视频通道的对角近似为 2219
        （参照量的 0.70 倍），完整求解后同样回落。

        零空间的维数由论文命题 2 给定：B 为 (7, 18)，故信令通道最多可辨识
        L = 7 个方向，其余 11 维不可辨识；H 为 (12, 18)，视频通道最多 12 维；
        缺口区的 6 个区域在两个通道的算子中列恒为 0，因而必属零空间。这与
        论文表 8 中"去掉空间图模块代价最大"的结论一致：缺口区只能由式(12)
        的图先验与锚点补足，不能指望观测梯度。

        预条件不改变式(13) 的最优解，只改变迭代的到达路径。

        ---- 数值下界 ----

        阻尼必须取相对量。Hessian 含观测精度 omega^2 的因子，而三条通道的
        omega^2 相差若干量级（视频约 5e-4、信令约 5e-8），因此**不能**用固定
        的绝对下界。早先的实现取 clamp(min=1e-6)，该值大于锚点与信令通道的
        真实曲率，凡低于 1e-6 的位置都被抬到 1e-6，预条件后的梯度被额外乘上
        omega^2/1e-6 这个因子，锚点与信令通道因而被整体压低若干数量级。现在
        统一取 lambda = eps_curv * max_k Hess_kk，即 Levenberg-Marquardt 的
        缩放方式，使阻尼只承担避免除以零与抑制近零奇异值的作用。

        锚点通道的 Hessian 为 P^T Omega^2 P，P 是 0/1 选择矩阵，故该矩阵本身
        就是对角阵，两种预条件方式在它上面完全等价。
        """
        mode = getattr(self.cfg, 'precond', 'full')
        if mode not in ('full', 'diag', 'none'):
            raise ValueError("precond 只能取 'full' / 'diag' / 'none'，"
                             '实际为 %r' % (mode,))
        if mode == 'none':
            return raw

        diag = hess.diagonal(dim1=-2, dim2=-1)                   # (B, T, N)
        scale = diag.detach().abs().amax(dim=-1, keepdim=True)
        floor = (self.cfg.eps_curv * scale).clamp(min=torch.finfo(
            diag.dtype).tiny)

        if mode == 'diag':
            return raw / (diag + floor)

        n = raw.shape[-1]
        eye = torch.eye(n, dtype=raw.dtype, device=raw.device)
        # hess 由 A^T Omega^2 A 构造，恒为半正定；加上正阻尼后正定。
        # 相除前先对称化，消除构造过程中浮点误差带来的不对称。
        a = hess.detach()
        a = 0.5 * (a + a.transpose(-1, -2)) + floor.unsqueeze(-1) * eye
        return torch.linalg.solve(a, raw.unsqueeze(-1)).squeeze(-1)

    def _signal_residual(self, n: torch.Tensor, pi: torch.Tensor,
                         y_sig: torch.Tensor):
        """式(9) 与式(17)：信令的空间混合观测与重构残差。

        式(9) 的完整形式同时含空间混合 B 与时间聚合 W：

            y^sig_j = sum_t W_jt * B diag(pi_t) n_t + eps_j

        其中时间聚合 W 把分钟级状态累加到信令窗口上。**本实现在数据层完成
        时间聚合**（见 src/data/dataset.py 与 src/data/signaling.py 的
        time_aggregation_matrix），模型接收的 y_sig 已经是摊平到基本步上的
        (B, T, L) 序列，故此处只保留空间混合 B。

        这样划分的理由：论文 5.1 节报告的基本时间步为 15 min、信令窗口亦为
        15 min，此时 W 退化为单位阵，式(9) 的时间聚合不产生任何作用；而在
        分钟级时间步的非退化设定下，W 的形状随窗口位置变化，属于数据组装
        而非模型结构。把 W 留在数据层可以让模型对两种设定都保持同一个接口。

        Args:
            n:     (B, T, N) 当前人数估计。
            pi:    (B, T, N) 当前渗透率估计。
            y_sig: (B, T, L) 已摊平到基本步的信令设备数观测。

        Returns:
            resid: (B, T, L) 信令单元级的重构残差
            loss: 标量，式(15) 的信令数据项
        """
        mixed = pi * n                                           # (B, T, N)
        pred = mixed @ self.Bmat.t()                             # (B, T, L)
        resid = pred - y_sig
        loss = (self.omega_sig2() * resid ** 2).mean()           # 取均值，理由同式(14)
        return resid, loss

    def omega_sig2(self) -> torch.Tensor:
        """信令观测精度矩阵（对角）的平方项，形状 (1, 1, L)。

        形状按 (B, T, L) 的残差张量广播；每个信令单元一个精度，
        不随时间与样本变化。
        """
        return torch.exp(2.0 * self.log_omega_sig).view(1, 1, -1)

    def _signal_grad(self, pi: torch.Tensor, resid: torch.Tensor):
        """式(18) 信令通道：diag(pi) B^T Omega^2 e。

        时间聚合 W 已在数据层完成，故这里没有 W 项（见 _signal_residual）。

        Returns:
            grad: (B, T, N)
            grad_pi_raw: (B, T, N) 式(15) 对 pi 的梯度的一半，
                         用于式(22) 的渗透率更新（见 _penetration_update）
        """
        u = self.omega_sig2() * resid                            # (B, T, L)
        raw = u @ self.Bmat                                      # B^T Omega^2 e
        # 观测算子为 B diag(pi)，故真实梯度为 diag(pi) B^T Omega^2 e，且
        # Hess = diag(pi) B^T Omega^2 B diag(pi)。注意必须把 diag(pi) 并入
        # 梯度之后再求解：完整求解不是逐分量相除，pi * solve(H, raw) 与
        # solve(H, pi * raw) 并不相等。B^T Omega^2 B 与时间无关，先算成
        # (N, N) 再按 pi 两侧加权。
        grad = pi * raw
        bwb = (self.Bmat * self.omega_sig2().view(-1, 1)).t() @ self.Bmat   # (N, N)
        hess = pi.unsqueeze(-1) * bwb * pi.unsqueeze(-2)          # (B, T, N, N)
        step = self._precondition(grad, hess)
        # 单区域观测（B 退化为选择矩阵）时有 step = (pi n - y)/pi = n - n*，
        # 与视频、锚点两通道的牛顿步一致。
        return step, step

    def _anchor_grad(self, n: torch.Tensor, y_anc: torch.Tensor,
                     anc_mask: torch.Tensor):
        """式(11) 与式(18) 锚点通道。

        Args:
            anc_mask: (B, T, A) 该时刻该点位是否有锚点观测，1 有 0 无。

        Returns:
            grad: (B, T, N)
            loss: 标量，式(16) 的锚点数据项
        """
        omega2 = torch.exp(2.0 * self.log_omega_anc)
        pred = n @ self.P.t()                                    # (B, T, A)
        resid = (pred - y_anc) * anc_mask
        # 只在**有观测**的位置取均值：锚点掩码下大量位置为 0，若按全部位置
        # 取均值，损失会被未观测位置稀释，且稀释比例随锚点稀疏化档位变化，
        # 使不同档位下的损失不可比。
        n_active = anc_mask.sum().clamp(min=1.0)
        loss = (omega2 * resid ** 2).sum() / n_active
        raw = (omega2 * resid) @ self.P                          # (B, T, N)
        # 观测算子为 P，Hess = P^T Omega^2 P。P 是 0/1 选择矩阵，故 Hess 本身
        # 就是对角阵：有锚点的区域为 omega^2、其余为 0，两种预条件方式在此
        # 完全等价。预条件后只有被锚点观测到的区域得到梯度，缺口区不因锚点
        # 项而产生虚假更新。
        curv = omega2 * (self.P ** 2).sum(dim=0).view(1, 1, -1).expand_as(raw)
        grad = self._precondition(raw, torch.diag_embed(curv))
        return grad, loss

    # ------------------------------------------------------------------ 渗透率
    def _penetration(self, eta: torch.Tensor) -> torch.Tensor:
        """式(10) 的有界参数化：pi = pi_min + (pi_max - pi_min) * sigmoid(eta)。"""
        span = self.cfg.pi_max - self.cfg.pi_min
        return self.cfg.pi_min + span * torch.sigmoid(eta)

    def _init_eta(self, t_steps: int, batch: int) -> torch.Tensor:
        """式(10) 的初始 eta：U v_t + b + delta_t。

        论文中 U v_t^pi + b + delta_t 不含样本维，故初始值与样本无关；
        随后每个样本按式(22) 各自更新。
        """
        u = self.U                                              # (N, r)
        v = self.v_pi[:t_steps]                                 # (T, r)
        low_rank = v @ u.t()                                    # (T, N)
        eta0 = low_rank + self.b_pi + self.delta_pi[:t_steps]    # (T, N)
        return eta0.unsqueeze(0).expand(batch, -1, -1).contiguous()

    def _penetration_update(self, eta: torch.Tensor, grad_pi_raw: torch.Tensor,
                            stage: int) -> torch.Tensor:
        """式(22)：eta <- eta - gamma * grad_eta L_sig + R_pi(eta)。

        grad_eta L_sig = dL_sig/dpi * dpi/deta
        dL_sig/dpi    = 2 * grad_pi_raw（式(15) 对 pi 求导）
        dpi/deta      = (pi_max - pi_min) * sigmoid(eta) * (1 - sigmoid(eta))
        """
        span = self.cfg.pi_max - self.cfg.pi_min
        sig = torch.sigmoid(eta)
        dpi_deta = span * sig * (1.0 - sig)
        grad_eta = 2.0 * grad_pi_raw * dpi_deta
        gamma = F.softplus(self.log_gamma[stage])
        r_pi = self.R_pi[stage](eta.transpose(1, 2)).transpose(1, 2)
        return eta - gamma * grad_eta + r_pi

    # ------------------------------------------------------------------ 门控
    def _channel_log_prior(self, y_vis: torch.Tensor, m_vis: torch.Tensor,
                           o_vis: torch.Tensor,
                           anc_mask: torch.Tensor) -> torch.Tensor:
        """三条观测通道的精度占比取对数，形状 (B, T, 3)。

        式(19) 说明门控的等价形式是“各通道精度在总精度中的占比”，故把
        式(7)、式(15)、式(16) 的精度分别沿各自的观测维取均值，作为门控的
        先验，再由 MLP(h_t) 在此先验之上作修正。三者的观测维分别是摄像头
        M、信令单元 L 与锚点 A，故先各自降到 (B, T, 1) 再拼接。

        用精度而非梯度幅度作先验是必要的：三条通道的梯度量级由各自的观测
        尺度决定，信令观测是聚合后的设备数（本数据集中量级达数千），其梯度
        幅度天然比视频通道大一个数量级。幅度大只反映尺度差异、不反映可靠性
        差异，直接幅度做门控输入时 Softmax 会锁死在幅度最大的通道上。
        """
        with torch.no_grad():                                    # 门控是可靠性统计量，不回传梯度
            s0 = F.softplus(self.log_sigma0)
            s1 = F.softplus(self.log_sigma1)
            s2 = F.softplus(self.log_sigma2)
            var = s0 ** 2 + s1 ** 2 * y_vis.clamp(min=0) + s2 ** 2 * o_vis
            p_vis = (m_vis / var.clamp(min=1e-6)).mean(dim=-1, keepdim=True)
            p_sig = torch.exp(2.0 * self.log_omega_sig).mean().expand_as(p_vis)
            # 锚点通道只在该时刻该点位有观测时贡献精度，故按掩码取均值
            p_anc = (torch.exp(2.0 * self.log_omega_anc)
                     * anc_mask).mean(dim=-1, keepdim=True)
            p = torch.cat([p_vis, p_sig, p_anc], dim=-1)          # (B, T, 3)
            return torch.log(p.clamp(min=1e-12))

    def _gates(self, n: torch.Tensor, ext: torch.Tensor,
               g_vis: torch.Tensor, g_sig: torch.Tensor,
               g_anc: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
        """式(19)：alpha = Softmax(MLP(h_t) + log 精度占比)。

        论文只说明 h_t 为“当前状态的嵌入表示”，未给出具体取法。本实现的
        h_t 取三项之和：状态 n_t 的线性投影、外部变量 e_t 的线性投影，
        以及三条观测通道残差幅度（各取沿区域维的均值）的线性投影。

        残差幅度必须先化为无量纲量再送入 MLP，否则这一路特征同样会被尺度
        主导，门控退化成“选量级最大的通道”。这里把每个通道的幅度除以它
        自己的批次均值，得到的是相对波动，三条通道因而可比。
        """
        with torch.no_grad():                                    # 门控是可靠性统计量，不回传梯度
            mag = torch.stack([
                g_vis.abs().mean(dim=-1),
                g_sig.abs().mean(dim=-1),
                g_anc.abs().mean(dim=-1)], dim=-1)               # (B, T, 3)
            scale = mag.mean(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            res_mag = mag / scale

        h = (self.gate_proj_state(n)
             + self.gate_proj_ext(ext)
             + self.gate_proj_res(res_mag))
        return F.softmax(self.gate_mlp(F.gelu(h)) + log_prior,    # (B, T, 3)
                         dim=-1)

    # ------------------------------------------------------------------ 前向
    def forward(self,
                y_vis: torch.Tensor,
                m_vis: torch.Tensor,
                o_vis: torch.Tensor,
                y_sig: torch.Tensor,
                y_anc: torch.Tensor,
                anc_mask: torch.Tensor,
                ext: torch.Tensor,
                return_states: bool = False):
        """
        Args:
            y_vis:    (B, T, M) 摄像头局部人数观测
            m_vis:    (B, T, M) 有效观测权重 m^vis，[0, 1]
            o_vis:    (B, T, M) 遮挡比例或图像质量指标
            y_sig:    (B, J, L) 信令设备数观测
            y_anc:    (B, T, A) 锚点人数观测，无锚点处填 0
            anc_mask: (B, T, A) 锚点可用性掩码，1 有 0 无
            ext:      (B, T, E) 外部变量

        Returns:
            dict，键为
              n_hat   (B, T, N)   反演的各区域人数
              pi_hat  (B, T, N)   估计的设备渗透率
              rho_hat (B, T, N)   密度，式(1)
              q_hat   (B, T, N)   拥堵持续状态，式(3)
              logits  (B, T, N, K) 风险读出的未归一化分数
              probs   (B, T, N, K) 风险概率，式(5)
              aux     dict        数据项损失与门控，供训练与可解释性分析
        """
        b, t, _ = y_vis.shape
        c = self.cfg
        n = y_vis.new_zeros(b, t, c.n_regions)
        eta = self._init_eta(t, b)

        aux = {'loss_vis': y_vis.new_zeros(()),
               'loss_sig': y_vis.new_zeros(()),
               'loss_anc': y_vis.new_zeros(()),
               'gates': [], 'states': []}
        zero = y_vis.new_zeros(b, t, c.n_regions)

        # 式(19) 门控的精度先验只依赖观测与结构，与展开阶段无关，故在循环外算一次
        log_prior = self._channel_log_prior(y_vis, m_vis, o_vis, anc_mask)

        for s in range(c.n_stages):
            # 关闭渗透率校准时，式(10) 退化为常数 pi = 1，即不做渗透率补偿
            pi = (self._penetration(eta) if c.use_penetration
                  else y_vis.new_ones(b, t, c.n_regions))

            # 式(18) 三条观测通道的反投影；被消融的通道返回零梯度与零数据项
            if c.use_video:
                g_vis, loss_vis = self._video_grad(n, y_vis, m_vis, o_vis)
            else:
                g_vis, loss_vis = zero, y_vis.new_zeros(())
            if c.use_signal:
                resid_sig, loss_sig = self._signal_residual(n, pi, y_sig)
                g_sig, grad_pi_raw = self._signal_grad(pi, resid_sig)
            else:
                g_sig, grad_pi_raw = zero, zero
                loss_sig = y_vis.new_zeros(())
            if c.use_anchor:
                g_anc, loss_anc = self._anchor_grad(n, y_anc, anc_mask)
            else:
                g_anc, loss_anc = zero, y_vis.new_zeros(())

            aux['loss_vis'] = aux['loss_vis'] + loss_vis
            aux['loss_sig'] = aux['loss_sig'] + loss_sig
            aux['loss_anc'] = aux['loss_anc'] + loss_anc

            # 式(19) 门控；式(20) 梯度步
            if c.use_gate:
                alpha = self._gates(n, ext, g_vis, g_sig, g_anc,
                                    log_prior)                   # (B, T, 3)
            else:
                # 关闭门控时三条通道等权，即式(19) 的 alpha 恒为 1/3
                alpha = torch.full((b, t, 3), 1.0 / 3.0,
                                   dtype=n.dtype, device=n.device)
            # 被消融的通道权重置零后重新归一化。这一步不能省：被关闭的通道
            # 其反投影梯度为零，但式(19) 的精度先验仍会按观测精度给它较大的
            # 份额，实测单开视频通道时门控把 98% 的权重给了已关闭的信令通道，
            # 反演因此几乎不更新（n_hat 恒为 0），消融结果失去意义。
            keep = self._channel_keep(dtype=n.dtype, device=n.device)
            if not bool(keep.all()):
                alpha = alpha * keep.view(1, 1, -1)
                alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            step = F.softplus(self.log_eta_step[s])
            n_tilde = n - step * (alpha[..., 0:1] * g_vis
                                  + alpha[..., 1:2] * g_sig
                                  + alpha[..., 2:3] * g_anc)
            aux['gates'].append(alpha.detach())

            # 式(21) 的近端算子替身：GCN 与 TCN 两支可分别关闭
            prox = n_tilde
            if c.use_graph or c.use_temporal:
                feat = n_tilde
                if c.use_graph:
                    feat = self.gcn(feat.unsqueeze(-1), self.L).squeeze(-1)
                if c.use_temporal:
                    feat = self.tcn(feat)
                beta = F.softplus(self.log_beta_prox[s])
                prox = n_tilde + beta * feat
            # ReLU 为非负投影，始终保留
            n = F.relu(prox)

            # 式(22) 渗透率更新，再经式(10) 映射回物理区间
            if c.use_penetration:
                eta = self._penetration_update(eta, grad_pi_raw, s)

            if return_states:
                aux['states'].append(n.detach())
            # 末阶段的单通道反投影，供式(26) 的跨模态一致性损失使用
            aux['g_vis'] = g_vis
            aux['g_sig'] = g_sig
            aux['step'] = step

        # 关闭渗透率校准时 pi 恒为 1，返回值与式(18) 中实际使用的量保持一致
        pi_hat = (self._penetration(eta) if c.use_penetration
                  else y_vis.new_ones(b, t, c.n_regions))
        rho_hat = n / self.areas.view(1, 1, -1)                  # 式(1)
        q_hat = self.cum_dwell_state(rho_hat)                    # 式(3)
        dn = self._delta(n)
        v_hat = self._net_inflow(dn)                             # 净流入速率

        z_mono = torch.stack([rho_hat, q_hat, dn], dim=-1)       # (B, T, N, 3)
        # 外部变量按区域广播：天气、活动、节假日等对整个景区是同一组取值，
        # 但读出头是逐区域作用的，故在区域维上展开到 (B, T, N, E)
        ext_r = ext.unsqueeze(2).expand(-1, -1, c.n_regions, -1)
        # 消融时把对应分量置零：w_free 的形状保持不变，被关闭的分量以 0 进入
        # 读出层，等效于该项不参与式(5)，同时 state_dict 与完整模型兼容
        if not c.use_conservation:
            v_hat = torch.zeros_like(v_hat)
        if not c.use_external:
            ext_r = torch.zeros_like(ext_r)
        z_free = torch.cat([v_hat.unsqueeze(-1), ext_r], dim=-1)  # (B, T, N, E+1)
        logits = self.readout(z_mono, z_free)                    # 式(5)
        probs = F.softmax(logits, dim=-1)

        aux['gates'] = torch.stack(aux['gates'], dim=0)          # (S, B, T, 3)
        return {'n_hat': n, 'pi_hat': pi_hat, 'rho_hat': rho_hat,
                'q_hat': q_hat, 'v_hat': v_hat,
                'logits': logits, 'probs': probs, 'aux': aux}

    # ------------------------------------------------------------------ 状态派生
    def cum_dwell_state(self, rho: torch.Tensor, tau_low=None) -> torch.Tensor:
        """式(3)：q_{k,t} = lambda_q q_{k,t-1} + (1 - lambda_q) ReLU(rho - tau_1)。

        Args:
            rho: (B, T, N) 密度。
            tau_low: (N,) 各区域低风险阈值 tau_{k,1}；None 时取默认值。
        """
        lam = torch.sigmoid(self.logit_lambda_q)
        tau = (torch.full_like(self.areas, 0.5) if tau_low is None
               else tau_low.to(rho.device).float())
        excess = F.relu(rho - tau.view(1, 1, -1))                # (B, T, N)
        q = torch.zeros_like(rho[:, 0])                          # q_0 由零起算
        seq = []
        for step in range(rho.shape[1]):
            q = lam * q + (1.0 - lam) * excess[:, step]
            seq.append(q)
        return torch.stack(seq, dim=1)

    @staticmethod
    def _delta(n: torch.Tensor) -> torch.Tensor:
        """Delta n_{k,t} = n_{k,t} - n_{k,t-1}，首步取 0（式(5) 的人数变化量）。"""
        out = torch.zeros_like(n)
        out[:, 1:] = n[:, 1:] - n[:, :-1]
        return out

    def _net_inflow(self, dn: torch.Tensor) -> torch.Tensor:
        """净流入速率 v（式(2) 状态向量的第二分块）。

        论文把 v 定义为各区域净流入速率，但未给出它的观测方程与估计方式。
        按 3.4 节的守恒关系 n_{t+1} = n_t + F_in - F_out + A_move，净流入
        在数值上就是人数差分，故本实现取人数差分的一阶平滑作为 v：

            v_t = lam_v * v_{t-1} + (1 - lam_v) * Delta n_t

        lam_v 可学习。这一取法的含义是：瞬时差分对检测噪声敏感，平滑后的
        净流入更能反映疏散与聚集的持续趋势。
        """
        lam = torch.sigmoid(self.logit_lambda_v)
        out = torch.zeros_like(dn)
        prev = torch.zeros_like(dn[:, 0])
        for step in range(dn.shape[1]):
            prev = lam * prev + (1.0 - lam) * dn[:, step]
            out[:, step] = prev
        return out
