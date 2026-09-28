"""Knowledge distillation utilities: learnable temperature + soft-label distillation with a gradient reversal layer. / 知识蒸馏工具函数：可学习温度 + 梯度反转层的软标签蒸馏。"""

import math

import torch
import torch.nn.functional as F


def KL_loss(logits_s, logits_t, temperature):
    """KL distillation loss between the student and teacher logits under the given temperature. / 学生与教师 logits 在给定温度下的 KL 蒸馏损失。"""
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


def decay_schedule(epoch, num_loops, em=0):
    """Temperature scheduling: λ stays at 0 for the first em epochs (flat period), then cosine-decays from 0 to -1 (stays at -1 after epoch reaches num_loops). / 温度调度：前 em 个 epoch 保持 0（平坦期），之后 λ 由 0 余弦递减到 -1（epoch 到达 num_loops 后保持 -1）。"""
    if num_loops <= 0 or em >= num_loops:
        return 0.0
    if epoch <= em:
        return 0.0
    i = min(epoch - em, num_loops - em)
    v = (math.cos(i * math.pi / (num_loops - em)) + 1.0) * 0.5
    return v - 1.0
