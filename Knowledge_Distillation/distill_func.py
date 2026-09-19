"""Knowledge distillation utilities: learnable temperature + curriculum-scheduled soft-label distillation with a gradient reversal layer. / 知识蒸馏工具函数：可学习温度 + 梯度反转层的课程化软标签蒸馏。"""

import math

import torch
import torch.nn.functional as F


def soft_ce(logits_s, logits_t, temperature):
    """Soft cross-entropy distillation loss (no T² scaling; equivalent to the cross-entropy term of the temperature-normalized KL). / 软交叉熵蒸馏损失（不含 T² 缩放，等价于温度归一化 KL 的交叉熵项）。"""
    pt = F.softmax(logits_t / temperature, dim=1)
    log_ps = F.log_softmax(logits_s / temperature, dim=1)
    return -(pt * log_ps).sum(dim=1).mean()


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer: identity in the forward pass; multiplies gradients by lambda_ in the backward pass. / 梯度反转层：前向恒等，反向把梯度乘以 lambda_。"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        return lambda_ * grads, None


def curriculum_decay(epoch, num_loops):
    """Temperature curriculum scheduling: λ cosine-decays from 0 to -1 (stays at -1 after epoch reaches num_loops). / 温度课程调度：λ 由 0 余弦递减到 -1（epoch 到达 num_loops 后保持 -1）。"""
    if num_loops <= 0:
        return 0.0
    i = min(max(epoch, 0), num_loops)
    v = (math.cos(i * math.pi / num_loops) + 1.0) * 0.5
    return v - 1.0
