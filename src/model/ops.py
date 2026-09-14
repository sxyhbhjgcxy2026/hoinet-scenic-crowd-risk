# -*- coding: utf-8 -*-
"""基础算子：切比雪夫图卷积、时间卷积、单调权重参数化、软阈值。

对应论文：
  - 式(12) 图拉普拉斯 L = D - W_g 与空间先验
  - 式(21) GCN 近似空间先验的近端算子、TCN 近似时间项的软阈值近端算子
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebConv(nn.Module):
    """切比雪夫图卷积（式(21) 的 GCN 模块）。

    在功能连通图上传播邻域人数状态，近似式(12) 空间先验 R_spa 的近端算子。
    使用 K 阶截断的切比雪夫多项式展开，避免对 L 做特征分解：

        T_0(x) = x
        T_1(x) = L~ x
        T_k(x) = 2 L~ T_{k-1}(x) - T_{k-2}(x)
        y = sum_{k=0}^{K-1} theta_k T_k(x)

    其中 L~ = 2L/lambda_max - I 为缩放后的拉普拉斯矩阵。

    Args:
        n_nodes: 区域数 N。
        cheb_k: 切比雪夫阶数 K，论文取 3。
        in_dim / out_dim: 通道数。
        use_bias: 是否加偏置。
    """

    def __init__(self, n_nodes: int, cheb_k: int = 3, in_dim: int = 1,
                 out_dim: int = 1, use_bias: bool = True):
        super().__init__()
        self.cheb_k = cheb_k
        self.n_nodes = n_nodes
        self.weight = nn.Parameter(torch.empty(cheb_k, in_dim, out_dim))
        # 切比雪夫基的线性组合系数，按 PyTorch 惯例初始化
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (..., N, in_dim) 节点特征。本文中 x 为 (B, T, N, 1) 的人数状态。
            laplacian: (N, N) 已缩放的拉普拉斯矩阵 L~。

        Returns:
            (..., N, out_dim)
        """
        lead = x.shape[:-2]
        h = x.reshape(-1, self.n_nodes, x.shape[-1])
        lx = laplacian.to(h.dtype).to(h.device)

        # 逐阶递推切比雪夫多项式
        tx = [h, torch.einsum('nm,bmc->bnc', lx, h)]
        for _ in range(2, self.cheb_k):
            tx.append(2.0 * torch.einsum('nm,bmc->bnc', lx, tx[-1]) - tx[-2])

        out = torch.stack(
            [torch.einsum('bnc,cq->bnq', tx[k], self.weight[k])
             for k in range(self.cheb_k)], dim=0).sum(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*lead, self.n_nodes, -1)


class TemporalConv(nn.Module):
    """因果时间卷积（式(21) 的 TCN 模块）。

    在时间维上恢复信令 15 分钟窗口内的分钟级波动，近似时间平滑项与稀疏突变项
    组合的软阈值近端算子。默认两层、核大小 K1=3 与 K2=5，与论文一致。

    Args:
        channels: 特征通道数（本文为 1，即人数）。
        kernels: 各层核大小。
        dilations: 各层膨胀系数；None 时全为 1。
    """

    def __init__(self, channels: int = 1, kernels=(3, 5), dilations=None):
        super().__init__()
        dilations = dilations or [1] * len(kernels)
        self.layers = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            # 因果卷积：左侧补足 (k-1)*d 个时间步，输出长度与输入一致
            self.layers.append(nn.Conv1d(channels, channels, kernel_size=k,
                                         dilation=d, padding=(k - 1) * d))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, N) -> Returns: (B, T, N)。

        实现对每个区域的时间序列施加**同一组**时间卷积核：把区域维折进批维，
        以单通道卷积处理 (B*N, 1, T)，再折回 (B, T, N)。这样做的依据是，
        式(21) 的时间平滑与稀疏突变算子在所有区域上形制相同，只有输入序列
        不同，故权重跨区域共享；这也使参数量与区域数无关，避免 N=18 时
        时间分支的参数远超图分支。
        """
        b, t, n = x.shape
        h = x.permute(0, 2, 1).reshape(b * n, 1, t)          # (B*N, 1, T)
        for conv in self.layers:
            y = conv(h)
            y = y[..., :t]                         # 裁掉右侧多余的补零，保持因果
            h = h + self.act(y)                    # 残差连接
        return h.reshape(b, n, t).permute(0, 2, 1)           # (B, T, N)


class MonotoneReadout(nn.Module):
    """带单调性约束的风险概率读出头（式(5)）。

    式(5) 要求 z 的三个分量（密度 rho、拥堵持续 q、人数增量 dn）增加时，
    较高风险等级的累积概率 p(r >= c) 不下降。

    实现方式：把这三个分量对应的权重约束为沿风险等级 c 非降，即
    W[j, 1] <= W[j, 2] <= ... <= W[j, K]。该条件是论文约束的充分条件：
    权重沿等级非降时，输入增大只会把 Softmax 的概率质量推向更高等级，
    因而 p(r >= c) 不会下降；而若权重沿等级递减，p(r >= c) 会随输入增大
    而下降，违反式(5)。

    非降权重用累积 Softplus 参数化：
        W[j, 1] = w0[j]
        W[j, c] = W[j, c-1] + softplus(dw[j, c])   (c >= 2)
    这样 W 沿 c 严格非降，且对参数可微、无需求解带约束优化。

    Args:
        n_monotone: 受单调约束的输入维数（本文为 3：rho、q、dn）。
        n_free: 不受约束的输入维数（本文为 v 与外部变量）。
        n_classes: 风险等级数 K，本文为 4。
    """

    def __init__(self, n_monotone: int = 3, n_free: int = 1 + 8,
                 n_classes: int = 4):
        super().__init__()
        self.n_monotone = n_monotone
        self.n_free = n_free
        self.n_classes = n_classes
        self.w0 = nn.Parameter(torch.zeros(n_monotone))
        # 等级间增量，softplus 后加到上一等级上，保证非降
        self.dw = nn.Parameter(torch.zeros(n_monotone, n_classes - 1)
                               if n_classes > 1 else torch.zeros(n_monotone, 0))
        self.w_free = nn.Parameter(torch.empty(n_free, n_classes))
        nn.init.xavier_uniform_(self.w_free)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def monotone_weight(self) -> torch.Tensor:
        """返回 (n_monotone, K) 的非降权重矩阵。"""
        if self.n_classes == 1:
            return self.w0.unsqueeze(1)
        inc = F.softplus(self.dw)                     # (n_monotone, K-1) > 0
        w = torch.cat([self.w0.unsqueeze(1),
                       self.w0.unsqueeze(1) + torch.cumsum(inc, dim=1)], dim=1)
        return w

    def forward(self, z_mono: torch.Tensor, z_free: torch.Tensor) -> torch.Tensor:
        """Args: z_mono (..., n_monotone), z_free (..., n_free) -> logits (..., K)。"""
        w = self.monotone_weight()
        out = z_mono @ w + z_free @ self.w_free + self.bias
        return out


class SoftThreshold(nn.Module):
    """可学习软阈值算子，近似 l1 时间项的近端算子。

    prox_{t*||.||_1}(x) = sign(x) * max(|x| - t, 0)。
    论文用 TCN 近似该算子；本模块提供精确形式，用于消融与对照实验。
    """

    def __init__(self, n_features: int = 1, init_thresh: float = 0.1):
        super().__init__()
        self.log_thresh = nn.Parameter(
            torch.full((n_features,), float(torch.log(torch.tensor(init_thresh)))
                       if init_thresh > 0 else -8.0))

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        t = F.softplus(self.log_thresh)
        return torch.sign(x) * F.relu(torch.abs(x) - t)
