# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LTD-Net entry retaining the Stfpm class name for checkpoint compatibility.

Modified from anomalib STFPM. The active implementation is SPA.py with
loss_gra_spa.py; torch_model.py is the preserved upstream architecture reference.
See the repository README for training and reproducibility limits.
"""

from .lightning_model import Stfpm

__all__ = ["Stfpm"]
