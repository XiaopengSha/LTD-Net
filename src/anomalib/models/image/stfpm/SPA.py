# SPDX-License-Identifier: Apache-2.0
# LTD-Net architecture from the working copy. Release changes: input guards,
# optional pretrained loading, ImageBatch access and ONNX score ordering.
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock

from anomalib.data import InferenceBatch
from anomalib.models.components import TimmFeatureExtractor
from .anomaly_map import AnomalyMapGenerator

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler

# --- 基础组件 (EMA, Fusion) 保持不变 ---
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // factor, channels // factor)
        self.conv1x1 = nn.Conv2d(channels // factor, channels // factor, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // factor, channels // factor, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        if b * self.groups == 0: return x
        group_x = x.view(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # group_x = x.view(b * self.groups, -1, h, w)
        # x_h = torch.mean(group_x, dim=-1, keepdim=True) 
        # x_w = torch.mean(group_x, dim=-2, keepdim=True).permute(0, 1, 3, 2)
        # y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1x1(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        att_h = self.sigmoid(x_h)
        att_w = self.sigmoid(x_w)
        out = group_x * att_h * att_w
        return out.view(b, c, h, w)

class FeatureFusionBlock(nn.Module):
    def __init__(self, high_channels: int, low_channels: int, use_ca: bool = True):
        super().__init__()
        self.lat_conv = nn.Conv2d(high_channels, low_channels, kernel_size=1, bias=False)
        self.bn_lat = nn.BatchNorm2d(low_channels)
        self.smooth_conv = nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1, bias=False)
        self.bn_smooth = nn.BatchNorm2d(low_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_ca = use_ca
        if self.use_ca: self.ca = EMA(low_channels, factor=8)
    def forward(self, high_feature: Tensor, low_feature: Tensor) -> Tensor:
        high_projected = self.relu(self.bn_lat(self.lat_conv(high_feature)))
        high_upsampled = F.interpolate(high_projected, scale_factor=2, mode='bilinear', align_corners=False)
        if high_upsampled.shape != low_feature.shape:
             high_upsampled = F.interpolate(high_projected, size=low_feature.shape[-2:], mode='bilinear', align_corners=False)
        fused = low_feature + high_upsampled
        out = self.relu(self.bn_smooth(self.smooth_conv(fused)))
        if self.use_ca: out = self.ca(out)
        return out

class ResNet10WithFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet(BasicBlock, [1, 1, 1, 1])
        del self.backbone.layer4; del self.backbone.fc; del self.backbone.avgpool
        self.fusion_3to2 = FeatureFusionBlock(high_channels=256, low_channels=128, use_ca=True) #！！！！！
        self.fusion_2to1 = FeatureFusionBlock(high_channels=128, low_channels=64, use_ca=True)
        self.att3 = EMA(256, factor=32) #!!!!!!!
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x); c2 = self.backbone.layer2(c1); c3 = self.backbone.layer3(c2)
        p3 = self.att3(c3) #!!!!!!!!
        #p3 = c3
        p2 = self.fusion_3to2(high_feature=p3, low_feature=c2)
        p1 = self.fusion_2to1(high_feature=p2, low_feature=c1)
        return {"layer1": p1, "layer2": p2, "layer3": p3}
    
class ResNet18WithFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # ⚠️ 核心修改：将 [1, 1, 1, 1] 改为 ResNet18 标配的 [2, 2, 2, 2]
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
        del self.backbone.layer4; del self.backbone.fc; del self.backbone.avgpool
        
        # 你的创新模块 (Fusion 和 EMA) 保持完全一致
        self.fusion_3to2 = FeatureFusionBlock(high_channels=256, low_channels=128, use_ca=True)
        self.fusion_2to1 = FeatureFusionBlock(high_channels=128, low_channels=64, use_ca=True)
        self.att3 = EMA(256, factor=32)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        
        p3 = self.att3(c3)
        p2 = self.fusion_3to2(high_feature=p3, low_feature=c2)
        p1 = self.fusion_2to1(high_feature=p2, low_feature=c1)
        
        return {"layer1": p1, "layer2": p2, "layer3": p3}
# =========================================================================
# 1. 核心模块: GatedAdapter (论文创新点)
# =========================================================================
class GatedAdapter(nn.Module):
    """
    [SPA Module] Structure-Preserving Adapter with Gating
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        
        self.adapter = nn.Sequential(
            # 1. 降维
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 2. 特征提取 (DWConv)
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 3. 升维 [!!! 关键修改 !!!]
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
            
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        adapted_feature = self.adapter(x)
        # 动态融合：结构保留 + 领域适配
        return  x + self.alpha * adapted_feature
# =========================================================================
# 2. 主模型: SPA-STFPM
# =========================================================================
class STFPMModel(nn.Module):
    def __init__(self, layers: Sequence[str], backbone: str = "resnet18", teacher_pretrained: bool = True) -> None:
        super().__init__()
        if backbone != "resnet18" or tuple(layers) != ("layer1", "layer2", "layer3"):
            raise ValueError("LTD-Net requires resnet18 teacher and layer1/layer2/layer3 in order")
        self.tiler: Tiler | None = None
        self.layers = layers

        # 1. Teacher Backbone (冻结的锚点)
        self.teacher_model = TimmFeatureExtractor(
            backbone=backbone, pre_trained=teacher_pretrained, layers=layers 
        ).eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 2. Trainable Adapters (SPA)
        self.t_adapter1 = GatedAdapter(64)
        self.t_adapter2 = GatedAdapter(128)
        self.t_adapter3 = GatedAdapter(256)

        # 3. Student
        self.student_model = ResNet10WithFPN()  #！！！！

        print(f"\n[INFO] SPA Architecture Initialized.")
        print(f"[INFO] Gated Adapters ready for domain adaptation.\n")

        self.anomaly_map_generator = AnomalyMapGenerator()

    # 保留历史 AdaBN 接口。当前特征提取器前向强制 eval，因此此循环不会改变 BN。
    def update_teacher_bn_stats(self, dataloader, device, max_batches=51):
        print("[INFO] Running legacy warm-up (extractor remains in eval; BN is not recalibrated)...")
        self.teacher_model.train()
        for param in self.teacher_model.parameters(): param.requires_grad = False
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches: break
                _ = self.teacher_model((batch.image if hasattr(batch, 'image') else batch['image']).to(device))
        self.teacher_model.eval()

    def forward(self, images: torch.Tensor):
        output_size = images.shape[-2:]
        if self.tiler:
            raise NotImplementedError("LTD-Net tiled inference has no untile implementation")

        # 1. 获取原始 Teacher 特征 (T_orig) - 锚点
        with torch.no_grad():
            t_orig = self.teacher_model(images)
        
        # 2. 应用 Adapter (T_adp) - 开启梯度
        # 注意: 我们需要 t_orig 用于 Loss，也需要它作为 Adapter 的输入
        t_adp = {}
        # 必须显式开启梯度，即使在 eval 模式下
        with torch.set_grad_enabled(self.training): 
            if "layer1" in t_orig: t_adp["layer1"] = self.t_adapter1(t_orig["layer1"])
            if "layer2" in t_orig: t_adp["layer2"] = self.t_adapter2(t_orig["layer2"])
            if "layer3" in t_orig: t_adp["layer3"] = self.t_adapter3(t_orig["layer3"])

        # 3. Student Forward (S)
        s_feats = self.student_model(images)



        # [修改这里]：训练时，或者正在导出 ONNX 时，都直接返回特征给 Wrapper
        if self.training: #or torch.onnx.is_in_onnx_export():
            return t_orig, t_adp, s_feats

        # 推理时：计算 S 和 T_adp 的差异
        anomaly_map = self.anomaly_map_generator(
            teacher_features=t_adp,
            student_features=s_feats,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        if torch.onnx.is_in_onnx_export():
            # 在导出 ONNX 时，只返回基础的 Tensor 元组
            return anomaly_map, pred_score
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)