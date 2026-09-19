#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def get_hyperparams(quick=False, max_samples=None):
    """Return the hyperparameter dictionary for the federated learning experiment.

    When quick=True, use fewer samples/rounds for a quick smoke test.
    When max_samples is not None, randomly sample that many samples from the training set (for quick experiments).

    返回联邦学习实验的超参数字典。

    quick=True 时使用少量样本/轮次，便于快速冒烟测试。
    max_samples 不为 None 时，从训练集中随机抽取该数量的样本（用于快速实验）。
    """
    hp = {
        # Data / 数据
        'max_samples': max_samples,
        # Fraction of Normal (class0) in the sampling result (0.5 = reduced to about 50%, remaining classes share the remaining budget by natural proportion) / Normal(class0) 在采样结果中的占比（0.5 = 降到约 50%，其余类按自然比例分配剩余预算）
        'normal_frac': 0.5,
        'test_ratio': 0.2,
        'public_size': 100000,
        # Public set subset size for learning quality evaluation (the full public set is used for distillation; evaluation uses a fixed subset for speed) / 学习质量评估使用的公共集子集大小（全量公共集用于蒸馏，评估用固定子集提速）
        'public_eval_size': 20000,
        # Training set subset size for overall loss evaluation (each round computes global loss on a fixed subset to avoid slow full evaluation) / 总体损失评估使用的训练集子集大小（每轮用固定子集算全局损失，避免全量评估过慢）
        'loss_eval_size': 50000,
        'seq_len': 10,
        # Model (complex model SE-ResNet) / 模型（复杂模型 SE-ResNet）
        'num_classes': 5,
        'teacher_channels': (371, 414, 442),
        'reduction': 16,
        # Federated learning / 联邦学习
        'n_clients': 30,
        'n_clusters': 9,
        'dirichlet_alpha': 3,
        'beta': 0.5,
        'rounds': 100,
        'local_epochs': 2,
        'batch_size': 256,
        'lr': 0.0001,
        'optimizer': 'sgd',
        'samples_per_cluster': 1,
        'use_class_weights': True,
        'class_weight_mult': [1.0, 1.0, 2, 1.0, 1.0],
        # Differential privacy (optional, disabled by default) / 差分隐私（可选，默认关闭）
        'use_dp': False,
        # Strict (ε, δ)-DP upper bound: δ is fixed, σ̂ is derived from the exact RDP formula based on data size / 严格 (ε, δ)-DP 上界：δ 固定，σ̂ 由精确 RDP 公式按数据规模反推
        'dp_epsilon': 8.0,
        'dp_delta': 1e-5,
        # Mask retention ratio r and clipping threshold X / 掩码保留比例 r、裁剪阈值 X
        'dp_retention_ratio': 0.6,
        'dp_clip_threshold': 1.5,
        'dp_gamma1': 0.9,
        'dp_gamma2': 0.999,
    }

    if quick:
        hp['max_samples'] = 50000
        hp['n_clients'] = 5
        hp['n_clusters'] = 3
        hp['rounds'] = 3
        hp['local_epochs'] = 1

    return hp


def get_file_name(hp):
    """Generate an experiment identifier file name from hyperparameters (for saving/distinguishing experiment artifacts). / 根据超参数生成实验标识文件名（用于保存/区分实验产物）。"""
    c = hp['teacher_channels']
    return (
        f"CarHacking_i{hp['rounds']}_N{hp['n_clients']}_C{hp['n_clusters']}"
        f"_E{hp['local_epochs']}_{hp['optimizer']}_lr{hp['lr']}"
        f"_B{hp['batch_size']}_beta{hp['beta']}"
        f"_ch{c[0]}-{c[1]}-{c[2]}"
    )
