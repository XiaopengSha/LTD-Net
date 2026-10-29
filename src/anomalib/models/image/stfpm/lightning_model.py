# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Modified for LTD-Net release preparation: configurable training and checkpoint-safe loading.


from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss_gra_spa import STFPMLoss
from .SPA import STFPMModel

__all__ = ["Stfpm"]

class Stfpm(AnomalibModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        teacher_pretrained: bool = True,
        image_size: tuple[int, int] = (256, 256),
        student_lr: float = 0.4,
        adapter_lr: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 0.001,
        eta_min: float = 1e-6,
        mining_ratio: float = 0.2,
        gradient_weight: float = 0.2,
        preservation_weight: float = 1.0,
        adabn_batches: int = 51,
    ) -> None:
        if len(image_size) != 2 or any(x <= 0 or x % 16 for x in image_size):
            raise ValueError("image_size must contain two positive multiples of 16")
        if min(student_lr, adapter_lr) <= 0 or weight_decay < 0 or adabn_batches < 0:
            raise ValueError("Invalid optimizer or AdaBN parameters")
        if pre_processor is True:
            pre_processor = self.configure_pre_processor(tuple(image_size))
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.image_size = tuple(image_size)
        self.student_lr = student_lr
        self.adapter_lr = adapter_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.adabn_batches = adabn_batches
        self.model = STFPMModel(backbone=backbone, layers=layers, teacher_pretrained=teacher_pretrained)
        self.loss = STFPMLoss(mining_ratio, gradient_weight, preservation_weight)

    # ============================================================
    # 1. AdaBN 热身 (Warm-up)
    # 保留历史预热接口；当前 TimmFeatureExtractor 强制 eval，不会更新 BN 统计量。
    # ============================================================
    def on_train_start(self) -> None:
        if self.adabn_batches and hasattr(self.model, "update_teacher_bn_stats"):
            train_loader = self.trainer.train_dataloader
            if isinstance(train_loader, Sequence): 
                train_loader = train_loader[0]
            
            # 执行历史预热循环（不会校准 BN）
            self.model.update_teacher_bn_stats(train_loader, self.device, self.adabn_batches)

    # ============================================================
    # 2. 训练步 (Training Step) - 适配 SPA 三元架构
    # ============================================================
    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs 

        # SPA 模型返回三个输出:
        # t_orig:  原始 ImageNet 教师 (锚点)
        # t_adp:   适配后的教师 (可训练)
        # s_feats: 学生特征
        t_orig, t_adp, s_feats = self.model(batch.image)
        
        # 计算三元损失 (Distillation + Preservation)
        loss = self.loss(t_orig, t_adp, s_feats)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.image.shape[0])
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs 

        # 验证/推理模式下，模型只返回 Anomaly Map 结果
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    # ============================================================
    # 3. 优化器配置 (Optimizer) - 参数分组
    # ============================================================
    def configure_optimizers(self):
        """
        配置参数组：
        - Student: 高学习率 (从头学)
        - Teacher Adapters: 低学习率 (微调)
        """
        
        # 收集 3 个 Adapter 的参数
        adapter_params = (
            list(self.model.t_adapter1.parameters()) +
            list(self.model.t_adapter2.parameters()) +
            list(self.model.t_adapter3.parameters())
        )

        parameters = [
            # 学生网络: 保持 0.4 的高学习率
            {
                "params": self.model.student_model.parameters(), 
                "lr": self.student_lr
            },
            # 教师适配器: 给较小的学习率 (如 0.05 或 0.1)
            # 因为有 Preservation Loss 约束，不容易崩，但也不要太大
            {
                "params": adapter_params, 
                "lr": self.adapter_lr 
            }
        ]

        optimizer = torch.optim.SGD(
            params=parameters,
            momentum=self.momentum,
            dampening=0.0,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100, 
            eta_min=self.eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS