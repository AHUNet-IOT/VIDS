#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model creation module. / 模型创建模块。"""

import os
import sys

# Add the project root (VeReMi) to sys.path so the shared module common can be imported / 将项目根目录（VeReMi）加入 sys.path，以便导入共享模块 common
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.models import SEResNet, count_parameters


def load_model(num_classes=5, channels=(256, 512, 512), reduction=16, seq_len=10):
    """Create the complex model (teacher model SE-ResNet) used in the federated learning stage. / 创建联邦学习阶段使用的复杂模型（教师模型 SE-ResNet）。"""
    return SEResNet(in_channels=1, num_classes=num_classes, seq_len=seq_len,
                    channels=channels, reduction=reduction)


def model_num_parameters(model):
    return count_parameters(model)
