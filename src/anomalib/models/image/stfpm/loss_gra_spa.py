# SPDX-License-Identifier: Apache-2.0
# LTD-Net loss; release change: expose existing constants without changing defaults.
import torch
from torch import nn
from torch.nn import functional as F

class STFPMLoss(nn.Module):
    """
    SPA-STFPM Loss Function (三元损失)
    
    包含三部分:
    1. Hard Mining MSE (Student <-> Adapted Teacher): 学习纹理细节 (针对 20% 困难像素)
    2. Gradient Loss   (Student <-> Adapted Teacher): 学习边缘结构 (针对螺纹)
    3. Preservation Loss (Adapted Teacher <-> Original Teacher): 防止模型坍塌，保留语义
    """

    def __init__(self, mining_ratio=0.2, gradient_weight=0.2, preservation_weight=1.0) -> None:
        super().__init__()
        if not 0 < mining_ratio <= 1 or min(gradient_weight, preservation_weight) < 0:
            raise ValueError("Invalid loss weights or mining_ratio")
        self.mining_ratio = mining_ratio
        self.gradient_weight = gradient_weight
        self.preservation_weight = preservation_weight
        # 注册 Sobel 算子 (不可训练，用于梯度损失)
        # 定义为 buffer 以便随模型保存和移动设备 (CPU/GPU)
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def compute_gradient_loss(self, feat1, feat2):
        """计算特征图的梯度差异 (边缘损失)"""
        b, c, h, w = feat1.shape
        # 扩展 kernel 到当前通道数
        kx = self.kernel_x.expand(c, 1, 3, 3)
        ky = self.kernel_y.expand(c, 1, 3, 3)
        
        # 计算 x 方向和 y 方向的梯度
        # group=c 保证对每个通道独立计算
        g1_x = F.conv2d(feat1, kx, groups=c, padding=1)
        g1_y = F.conv2d(feat1, ky, groups=c, padding=1)
        g2_x = F.conv2d(feat2, kx, groups=c, padding=1)
        g2_y = F.conv2d(feat2, ky, groups=c, padding=1)
        
        # 梯度的 MSE
        return F.mse_loss(g1_x, g2_x) + F.mse_loss(g1_y, g2_y)

    def forward(
        self,
        t_orig: dict[str, torch.Tensor],
        t_adp: dict[str, torch.Tensor],
        s_feats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            t_orig:  原始 ImageNet 教师特征 (锚点, Frozen)
            t_adp:   适配后的工业域教师特征 (Trainable Adapter Output)
            s_feats: 学生网络特征 (Student Output)
        """
        total_loss = 0
        
        # 遍历每一层 (layer1, layer2, layer3)
        for layer in t_orig.keys():
            # 1. 统一进行 L2 归一化
            # 归一化后关注特征的方向(模式)，而非数值大小，对光照鲁棒
            norm_t_orig = F.normalize(t_orig[layer], p=2, dim=1)
            norm_t_adp  = F.normalize(t_adp[layer], p=2, dim=1)
            norm_s      = F.normalize(s_feats[layer], p=2, dim=1)
            
            # =========================================================
            # Part A: Distillation Loss (Student 拟合 Adapted Teacher)
            # =========================================================
            
            # A1. Hard Pixel Mining (像素级纹理)
            # 计算差异图
            dist_map = torch.sum((norm_t_adp - norm_s) ** 2, dim=1)
            B, H, W = dist_map.shape
            
            # 只取误差最大的前 20% (Top-K)
            mining_ratio = self.mining_ratio
            k = max(int(H * W * mining_ratio), 1)
            
            # 拉平并取 TopK
            topk_vals, _ = torch.topk(dist_map.view(B, -1), k, dim=1)
            loss_pixel = 0.5 * torch.mean(topk_vals)
            
            # A2. Gradient Loss (边缘结构)
            # 强迫学生学习螺纹的边缘
            loss_grad = self.gradient_weight * self.compute_gradient_loss(norm_t_adp, norm_s)
            
            # 蒸馏总损失
            loss_distill = loss_pixel + loss_grad

            # =========================================================
            # Part B: Preservation Loss (Adapted Teacher 锚定 Original)
            # =========================================================
            
            # 核心机制：防止 Adapter 漂移过远导致模型坍塌
            # 强迫 T_adp 保持 T_orig 的语义结构
            loss_preserve = F.mse_loss(norm_t_adp, norm_t_orig)
            
            # =========================================================
            # Total Layer Loss
            # =========================================================
            # 1.0 是保留权重的系数。
            # 如果发现模型学不动工业特征，可以降低为 0.5 
            total_loss += loss_distill + self.preservation_weight * loss_preserve  #!!!!!
            
        return total_loss