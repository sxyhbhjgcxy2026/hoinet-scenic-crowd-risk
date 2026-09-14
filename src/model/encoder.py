# -*- coding: utf-8 -*-
"""视觉分支：多感受野编码与密度估计，对应论文 3.2 节与 5.4 节的实现细节。

论文 5.4 节给出的视觉分支配置：
    三个并行分支的卷积核为 3x3、5x5、7x7，通道数为 64、128、256，
    输出分辨率分别为输入的 1/4、1/8、1/16，三路特征经转置卷积上采样后
    融合为 d_v = 256 维视觉特征，再经密度头回归局部人数密度图。
    每个基本步内抽取 Ns = 30 个视频子步，子步特征聚合后得到该步的
    局部人数观测 y^vis 与其有效观测权重 m^vis。

本模块的输出正是 HOINet 前向所需的 y_vis 与 m_vis，因此两个模块可以直接
串接。为避免把 ResNet 等骨干的具体实现也一并引入（论文只声明了分支核大小
与通道数），本模块按论文声明的配置从零实现，不依赖任何预训练权重。

关于上采样方式的说明
--------------------
论文 5.4 节写的是“经转置卷积上采样”，因此**特征通路**一律使用
nn.ConvTranspose2d，不使用双线性插值。转置卷积的输出尺寸与目标分辨率
最多差一个像素，按右下裁剪对齐，避免出现奇偶尺寸错位。

需要区分的是：区域掩码与点标注密度图的对齐用的是最近邻缩放，那是对
标签/掩码的尺寸对齐，不是特征通路里的可学习上采样，两者性质不同，
故不违背上述约束。

关于密度图到人数的换算
----------------------
论文使用密度图回归的思想：密度图在区域上的积分即该区域人数。
y_vis 由密度图在摄像头视野覆盖范围内积分得到，这与式(6) 的
y^vis = m^vis * sum_k H_ik n_k 在形式上是同一件事——只是式(6) 的
H_ik 由摄像头部署几何给出，而这里的积分范围由密度图自身的空间支撑决定。
本实现把两者对接：密度头输出每像素的人数密度，覆盖范围内的积分即 y_vis。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisualConfig:
    """视觉分支配置，取值对应论文 5.4 节。"""

    in_channels: int = 3
    branch_channels: Tuple[int, int, int] = (64, 128, 256)   # 3x3 / 5x5 / 7x7
    branch_kernels: Tuple[int, int, int] = (3, 5, 7)
    branch_strides: Tuple[int, int, int] = (4, 8, 16)        # 输出降采样倍数
    d_visual: int = 256                                      # d_v，融合后维度
    n_substeps: int = 30                                     # Ns，每基本步的子步数
    density_stride: int = 1                                  # 密度头输出与输入的像素比


class _Branch(nn.Module):
    """单个感受野分支：逐级下采样到 1/s 分辨率，通道数逐级加倍。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        n_down = {4: 2, 8: 3, 16: 4}[stride]
        layers = []
        c_in = in_ch
        for i in range(n_down):
            c_out = out_ch // (2 ** (n_down - 1 - i))
            c_out = max(c_out, 16)
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True)]
            c_in = c_out
        # 用该分支声明的核大小做一次大核卷积，实现对应尺度的感受野
        layers += [nn.Conv2d(c_in, out_ch, kernel_size=kernel, padding=kernel // 2),
                   nn.BatchNorm2d(out_ch),
                   nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    """转置卷积上采样到目标分辨率，再与低层特征相加。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None):
        y = F.relu(self.bn(self.up(x)))
        if target is not None:
            # 目标分辨率与上采样结果最多差一个像素，按右下裁剪对齐
            if y.shape[-2:] != target.shape[-2:]:
                y = y[..., :target.shape[-2], :target.shape[-1]]
            y = y + target
        return y


class VisualEncoder(nn.Module):
    """多感受野视觉编码器与密度头。

    Args:
        cfg: VisualConfig。
        region_masks: (N, h, w) 各区域在像素网格上的掩码；提供后即可由密度图
                      直接积分出各区域人数，用于与式(6) 对接。
    """

    def __init__(self, cfg: Optional[VisualConfig] = None,
                 region_masks: Optional[torch.Tensor] = None,
                 camera_masks: Optional[torch.Tensor] = None):
        super().__init__()
        cfg = cfg or VisualConfig()
        self.cfg = cfg
        c1, c2, c3 = cfg.branch_channels
        k1, k2, k3 = cfg.branch_kernels
        s1, s2, s3 = cfg.branch_strides

        # ---- 三分支并行编码 ----
        self.branch_lo = _Branch(cfg.in_channels, c1, k1, s1)    # 1/4
        self.branch_mid = _Branch(cfg.in_channels, c2, k2, s2)   # 1/8
        self.branch_hi = _Branch(cfg.in_channels, c3, k3, s3)    # 1/16

        # ---- 转置卷积上采样，逐级融合到 1/4 分辨率 ----
        self.up_hi = _Up(c3, c2)                                 # 1/16 -> 1/8
        self.up_mid = _Up(c2, c1)                                # 1/8  -> 1/4
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * 2, cfg.d_visual, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.d_visual), nn.ReLU(inplace=True))

        # ---- 密度头：逐像素回归人数密度 ----
        self.density_head = nn.Sequential(
            nn.Conv2d(cfg.d_visual, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1))
        # 输出非负：密度不可能为负，用 Softplus 而非 ReLU 以保留零附近的梯度
        self.density_act = nn.Softplus(beta=1.0)

        # ---- 质量头：预测 m^vis（在线、清晰、目标在视野内的综合权重）----
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(cfg.d_visual, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid())

        if region_masks is not None:
            self.register_buffer('region_masks', region_masks.float())
        else:
            self.region_masks = None
        if camera_masks is not None:
            self.register_buffer('camera_masks', camera_masks.float())
        else:
            self.camera_masks = None

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """编码到 1/4 分辨率的融合特征。

        Args:
            frames: (B, C=3, H, W) 单帧，或 (B, Ns, 3, H, W) 一个基本步内的子步序列。

        Returns:
            (B, d_v, H/4, W/4) 融合特征。
        """
        squeeze = frames.dim() == 5
        if squeeze:
            b, ns = frames.shape[:2]
            frames = frames.reshape(b * ns, *frames.shape[2:])
        lo = self.branch_lo(frames)
        mid = self.branch_mid(frames)
        hi = self.branch_hi(frames)
        mid = self.up_hi(hi, mid)
        lo = self.up_mid(mid, lo)
        feat = self.fuse(torch.cat([lo, mid], dim=1))
        if squeeze:
            feat = feat.reshape(b, ns, *feat.shape[1:]).mean(dim=1)
        return feat

    def forward(self, frames: torch.Tensor) -> dict:
        """完整前向：编码 -> 密度图 -> 各区域人数与质量权重。

        Args:
            frames: (B, 3, H, W) 或 (B, Ns, 3, H, W)。

        Returns:
            dict，键为
              density   (B, 1, H/4, W/4)  人数密度图
              counts    (B, N)            各区域人数（region_masks 提供时）
              y_vis     (B, M)            各摄像头的局部人数观测
              m_vis     (B, M)            各摄像头的有效观测权重
        """
        feat = self.encode(frames)
        density = self.density_act(self.density_head(feat))
        out = {'density': density, 'feat': feat,
               'm_vis_total': self.quality_head(feat)}

        if self.region_masks is not None:
            # (N, h, w) 上采样到密度图分辨率后与密度图逐像素相乘再求和
            m = self.region_masks.unsqueeze(1).to(density.dtype)
            if m.shape[-2:] != density.shape[-2:]:
                m = F.interpolate(m, size=density.shape[-2:], mode='nearest')
            out['counts'] = (density * m).sum(dim=(-1, -2)).squeeze(1)  # (B, N)

        if self.camera_masks is not None:
            cm = self.camera_masks.unsqueeze(0).to(density.dtype)        # (M,h,w)
            if cm.shape[-2:] != density.shape[-2:]:
                cm = F.interpolate(cm, size=density.shape[-2:], mode='nearest')
            local = (density * cm).sum(dim=(-1, -2))                     # (B, M)
            q = self.quality_head(feat)                                  # (B, 1)
            out['y_vis'] = local
            out['m_vis'] = q.expand(-1, local.shape[-1])
        return out

    def density_loss(self, density: torch.Tensor,
                     point_maps: torch.Tensor) -> torch.Tensor:
        """密度图的监督损失。

        论文 5.1 节说明视频标注为逐帧的人头点标注，5.4 节说明密度头回归
        局部人数密度图。这里用点标注图与密度图的逐像素 L2 损失，这是密度
        回归的标准做法（点标注经固定高斯核展宽为密度图后与预测逐像素比较）。

        Args:
            density:   (B, 1, h, w) 预测密度图。
            point_maps:(B, 1, h, w) 点标注展宽后的密度图。

        Returns:
            标量损失。
        """
        if density.shape[-2:] != point_maps.shape[-2:]:
            point_maps = F.interpolate(point_maps, size=density.shape[-2:],
                                       mode='nearest')
        return F.mse_loss(density, point_maps)
