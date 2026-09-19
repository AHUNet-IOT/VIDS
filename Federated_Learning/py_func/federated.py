#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Federated learning training logic module.
1. Initial client clustering (round 0, hierarchical clustering based on cosine distance of representative gradients)
2. Gain-weighted intra-cluster sampling (learning quality + gradient consistency -> sampling probability)
3. Inter-cluster dynamic update scheduling (high/medium/low gain clusters upload at 1/2/3 round frequency)
4. Quality-aware aggregation (weighted average by individual weights)

联邦学习训练逻辑模块。
1. 初始客户端聚类（第 0 轮，基于代表梯度的余弦距离做层次聚类）
2. 增益加权 簇内采样（学习质量 + 梯度一致性 -> 采样概率）
3. 簇间动态更新调度（高/中/低增益簇按 1/2/3 轮频率上传）
4. 质量感知聚合（按个体权重加权平均）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from py_func.clustering import (
    flatten_params, cosine_distance, hierarchical_clustering,
    compute_learning_quality,
)
from py_func.create_model import load_model


class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy samples via (1-p_t)^γ to focus on hard samples. / Focal Loss：通过 (1-p_t)^γ 降低易分类样本的权重，聚焦难分类样本。"""

    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')  # -log(p_t)
        pt = torch.exp(-ce)
        focal_weight = (1.0 - pt) ** self.gamma
        if self.weight is not None:
            focal_weight = focal_weight * self.weight[targets]
        loss = focal_weight * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def evaluate_accuracy(model, x, y, device, batch_size=512):
    """Compute the model's classification accuracy on the given data (without updating parameters). / 在给定数据上计算模型的分类准确率（不更新参数）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(device).unsqueeze(1)
            yb = torch.from_numpy(y[i:i + batch_size]).to(device)
            logits, _ = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    model.train()
    return correct / max(total, 1)


def evaluate_loss(model, x, y, device, batch_size=512, class_weights=None):
    """Compute the model's average cross-entropy loss on the given data (without updating parameters). / 在给定数据上计算模型的平均交叉熵损失（不更新参数）。"""
    model.eval()
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(device).unsqueeze(1)
            yb = torch.from_numpy(y[i:i + batch_size]).to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            total += len(yb)
    model.train()
    return total_loss / max(total, 1)


def train_local(model, x, y, epochs, batch_size, lr, device, optimizer='adam',
                class_weights=None, use_focal_loss=False, focal_gamma=2.0):
    """Train the model for several epochs on the client's local data and return (updated model, average loss). / 在客户端本地数据上训练模型若干轮，返回 (更新后的模型, 平均损失)。"""
    if optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights).to(device)
    elif class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    model.train()
    n = len(x)
    total_loss = 0.0
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(x[idx]).to(device).unsqueeze(1)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
    avg_loss = total_loss / max(n * epochs, 1)
    return model, avg_loss


def train_local_dp(model, x, y, epochs, batch_size, lr, device, optimizer='adam',
                   class_weights=None, sanitizer=None, client_id=0):
    """DP local training: per-batch DP-SGD.

    Independent from the non-DP train_local; called only when use_dp=True, and
    does not affect the existing non-DP federated training path. For each batch:
    per-sample gradients -> masking/normalization/clipping/noising/recovery ->
    write back model gradients -> optimizer.step().

    DP 版本地训练：per-batch DP-SGD。

    与无 DP 的 train_local 相互独立，仅在 use_dp=True 时调用，不影响原有
    无 DP 联邦训练路径。对每个 batch：逐样本计算梯度 -> 掩码/归一化/裁剪/
    加噪/恢复 -> 写回模型梯度 -> optimizer.step()。
    """
    if sanitizer is None:
        raise ValueError("DP 训练需要 AdaptiveDPSanitizer 实例")
    if optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    weight = class_weights.to(device) if class_weights is not None else None
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    n = len(x)
    total_loss = 0.0
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(x[idx]).to(device).unsqueeze(1)
            yb = torch.from_numpy(y[idx]).to(device)
            B = xb.shape[0]

            # One forward pass; BN uses full batch statistics (training mode) / 一次前向，BN 使用完整 batch 统计量（训练模式）
            opt.zero_grad()
            logits, _ = model(xb)
            losses = F.cross_entropy(logits, yb, weight=weight, reduction='none')
            batch_loss = losses.detach().sum().item()

            # Per-sample gradients (retain_graph on the same graph, including BN's per-sample contribution) / 逐样本梯度（对同一计算图 retain_graph，含 BN 的 per-sample 贡献）
            per_sample_grads = []
            for j in range(B):
                g = torch.autograd.grad(losses[j], params, retain_graph=True)
                per_sample_grads.append(
                    torch.cat([gg.detach().reshape(-1) for gg in g]))
            del logits, losses

            # Write back model gradients after DP protection / DP 保护后写回模型梯度
            final_flat = sanitizer.sanitize_batch(per_sample_grads, client_id, device)
            offset = 0
            for p in params:
                num = p.numel()
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                p.grad.data.copy_(final_flat[offset:offset + num].view_as(p.data))
                offset += num
            opt.step()
            total_loss += batch_loss

    avg_loss = total_loss / max(n * epochs, 1)
    return model, avg_loss


def pretrain_collect_grads(models, x_train, y_train, client_indices,
                           batch_size, device, n_batches=5):
    """Pretraining phase: each client trains several steps on local data and collects parameter differences Δw.

    The parameter difference (Δw = θ^{t+1}_i − θ^t_g) is the representative gradient, used to:
      1) generate the global importance mask (top r% of accumulated difference magnitudes across all clients)
      2) initialize each client's first/second moments μk/vk

    预训练阶段：每个客户端在本地数据上训练若干步，收集参数差分 Δw。

    参数差分（Δw = θ^{t+1}_i − θ^t_g）即代表梯度，用于：
      1) 生成全局重要性掩码（所有客户端累积差分幅值 top r%）
      2) 初始化每个客户端的一阶/二阶矩 μk/vk
    """
    samples_per_client = {}
    for i, m in enumerate(models):
        if len(client_indices[i]) == 0:
            continue
        init_params = flatten_params(m).clone()
        opt = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
        idx = client_indices[i]
        collected = []
        for b in range(n_batches):
            lo = (b * batch_size) % len(idx)
            hi = min(lo + batch_size, len(idx))
            xb = torch.from_numpy(x_train[idx[lo:hi]]).to(device).unsqueeze(1)
            yb = torch.from_numpy(y_train[idx[lo:hi]]).to(device)
            opt.zero_grad()
            logits, _ = m(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            delta = flatten_params(m) - init_params  # Parameter difference Δw (representative gradient) / 参数差分 Δw（代表梯度）
            collected.append(delta)
        samples_per_client[i] = torch.stack(collected)  # [n_batches, n_params]
    return samples_per_client


def unflatten_params(model, flat_vec):
    """Write the flattened vector back to the model's parameters (in-place update, returns the model). / 把扁平向量写回模型各参数（原地更新，返回模型）。"""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat_vec[offset:offset + n].reshape(p.shape).to(p.device, p.dtype))
            offset += n
    return model


def run_federated_training(
        x_train, y_train, client_indices, x_public, y_public,
        num_classes=5, seq_len=10, n_clusters=10, rounds=60, local_epochs=1,
        batch_size=256, lr=0.001, beta=0.5, device='cuda',
        channels=(256, 512, 512), reduction=16, optimizer='adam',
        samples_per_cluster=2, use_class_weights=True, class_weight_mult=None,
        use_focal_loss=False, focal_gamma=2.0,
        public_eval_size=20000, loss_eval_size=50000, verbose=True,
        use_dp=False, dp_epsilon=8.0, dp_delta=1e-5,
        dp_retention_ratio=0.6, dp_clip_threshold=4.0,
        dp_gamma1=0.9, dp_gamma2=0.999):
    """Run federated learning and return (trained global complex model, per-round loss list). / 运行联邦学习，返回 (训练好的全局复杂模型, 每轮损失列表)。"""
    n_clients = len(client_indices)
    global_model = load_model(num_classes=num_classes, channels=channels,
                              reduction=reduction, seq_len=seq_len).to(device)

    # Class-weighted loss: inverse class frequency to alleviate class imbalance (minority attack classes being drowned out by majority classes) / 类别加权损失：按类别频率反比，缓解类别不平衡（少数攻击类被多数类淹没）
    if use_class_weights:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        # Further amplified by per-class multipliers (focus on classes with severe miss detection, e.g. class 2) / 按类别倍率进一步放大（重点提升漏检严重的类别，如类别2）
        if class_weight_mult is not None:
            w *= np.asarray(class_weight_mult, dtype=np.float64)
        w = w / w.sum() * num_classes
        class_weights = torch.tensor(w, dtype=torch.float32)
    else:
        class_weights = None

    loss_history = []              # Overall loss of the global model on the training set per round / 每轮全局模型在训练集上的总体损失
    cluster_labels = None          # Determined at round 0 and kept fixed thereafter / 第 0 轮确定后保持不变
    global_grad_ref = None         # Global gradient reference used to compute gradient consistency / 用于计算梯度一致性的全局梯度参考
    omega1, omega2 = None, None    # Inter-cluster scheduling thresholds / 簇间调度阈值
    cluster_tier = None            # Participation tier of each cluster (1=every round/2=every 2 rounds/3=every 3 rounds), fixed every 30 rounds / 各簇参与档位(1=每轮/2=每2轮/3=每3轮)，每30轮固定一次

    # Initially prepare a local model copy for each client (loaded with global parameters first) / 初始时给每个客户端准备一个本地模型副本（先加载全局参数）
    local_models = []
    for i in range(n_clients):
        m = load_model(num_classes=num_classes, channels=channels,
                       reduction=reduction, seq_len=seq_len).to(device)
        m.load_state_dict(global_model.state_dict())
        local_models.append(m)

    # ===== Differential privacy initialization (optional, off by default) / 差分隐私初始化（可选，默认关闭）=====
    sanitizer = None
    if use_dp:
        import math as _math
        import sys as _sys
        from pathlib import Path as _Path
        _dp_root = _Path(__file__).resolve().parents[2]
        if str(_dp_root) not in _sys.path:
            _sys.path.insert(0, str(_dp_root))
        from Differential_Privacy.dp_mechanism import AdaptiveDPSanitizer
        from Differential_Privacy.rdp_accountant import (
            get_noise_multiplier_exact, compute_rdp_exact, rdp_to_epsilon,
            _EXACT_ORDERS,
        )

        # Strictly satisfy (ε, δ)-DP: infer the noise multiplier σ from the client with the fewest samples (worst privacy). / 严格满足 (ε, δ)-DP：按样本最少的客户端（隐私最差）反推噪声乘数 σ。
        # Sampling rate q = batch_size / N_min; steps = rounds × epochs × ceil(N_min/batch). / 采样率 q = batch_size / N_min，步数 = rounds × epochs × ceil(N_min/batch)。
        n_list = [len(client_indices[i]) for i in range(n_clients)
                  if len(client_indices[i]) > 0]
        n_min = min(n_list) if n_list else batch_size
        q = batch_size / float(max(n_min, 1))
        steps = _math.ceil(max(n_min, 1) / batch_size) * rounds * local_epochs
        sigma = get_noise_multiplier_exact(dp_epsilon, dp_delta, q, steps)
        eps_spent = rdp_to_epsilon(
            _EXACT_ORDERS, compute_rdp_exact(q, sigma, steps), dp_delta)

        sanitizer = AdaptiveDPSanitizer(
            retention_ratio=dp_retention_ratio, clip_threshold=dp_clip_threshold,
            noise_multiplier=sigma, gamma1=dp_gamma1, gamma2=dp_gamma2)
        print(f"[DP] 开启差分隐私: ε={dp_epsilon}(严格上界), δ={dp_delta}, "
              f"σ̂={sigma:.4f}, 核算ε={eps_spent:.4f}, "
              f"R={dp_retention_ratio}")
        # Pretraining phase collects gradients → generate the global importance mask / 预训练阶段收集梯度 → 生成全局重要性掩码
        print("[DP] 预训练阶段收集梯度样本 ...")
        dp_samples = pretrain_collect_grads(
            local_models, x_train, y_train, client_indices, batch_size, device)
        if dp_samples:
            accumulated = sum(s.sum(dim=0) for s in dp_samples.values())
            layer_sizes = [p.numel() for p in global_model.parameters()]
            sanitizer.build_mask(accumulated, layer_sizes=layer_sizes)
        # Pretraining does not update the model; re-sync local models to global parameters / 预训练不更新模型，重新同步本地模型到全局参数
        for i in range(n_clients):
            local_models[i].load_state_dict(global_model.state_dict())

    # Overall loss evaluation uses only a fixed subset of the training set to avoid re-evaluating on the full training set every round / 总体损失评估只使用训练集的一个固定子集，避免每轮在完整训练集上重复评估
    loss_eval_size = min(loss_eval_size, len(x_train))
    if loss_eval_size < len(x_train):
        _loss_idx = np.random.RandomState(0).choice(
            len(x_train), size=loss_eval_size, replace=False)
        x_loss_eval = x_train[_loss_idx]
        y_loss_eval = y_train[_loss_idx]
    else:
        x_loss_eval = x_train
        y_loss_eval = y_train

    # Loss of the initial global model on the training set (starting point of the round-1 curve, ≈ln(num_classes)) / 初始全局模型在训练集上的损失（第一轮曲线起点，≈ln(num_classes)）
    init_loss = evaluate_loss(global_model, x_loss_eval, y_loss_eval, device,
                              class_weights=class_weights)

    # Learning quality evaluation uses only a fixed subset of the public set to avoid re-evaluating 30 clients × full public set every round / 学习质量评估只使用公共集的一个固定子集，避免每轮 30 客户端 × 全量公共集 的重复评估
    public_eval_size = min(public_eval_size, len(x_public))
    if public_eval_size < len(x_public):
        _eval_idx = np.random.RandomState(0).choice(
            len(x_public), size=public_eval_size, replace=False)
        x_public_eval = x_public[_eval_idx]
        y_public_eval = y_public[_eval_idx]
    else:
        x_public_eval = x_public
        y_public_eval = y_public

    for t in range(rounds):
        grads = []
        accs = []

        # 1) Each client trains on local data and computes representative gradients and public-set accuracy / 1) 每个客户端在本地数据上训练，计算代表梯度与公共集准确率
        for i in range(n_clients):
            if len(client_indices[i]) == 0:
                local_models[i].load_state_dict(global_model.state_dict())
                grads.append(flatten_params(local_models[i]) * 0)
                accs.append(0.0)
                continue
            if use_dp:
                # DP path: per-batch DP-SGD, does not affect non-DP training / DP 路径：per-batch DP-SGD，不影响无 DP 训练
                train_local_dp(local_models[i], x_train[client_indices[i]],
                               y_train[client_indices[i]], epochs=local_epochs,
                               batch_size=batch_size, lr=lr, device=device,
                               optimizer=optimizer, class_weights=class_weights,
                               sanitizer=sanitizer, client_id=i)
            else:
                train_local(local_models[i], x_train[client_indices[i]],
                            y_train[client_indices[i]], epochs=local_epochs,
                            batch_size=batch_size, lr=lr, device=device,
                            optimizer=optimizer,
                            class_weights=class_weights,
                            use_focal_loss=use_focal_loss, focal_gamma=focal_gamma)
            grads.append(flatten_params(local_models[i]) - flatten_params(global_model))
            accs.append(evaluate_accuracy(
                local_models[i], x_public_eval, y_public_eval, device))

        # 2) Perform hierarchical clustering at round 0 / 2) 第 0 轮做层次聚类
        if t == 0:
            cluster_labels = hierarchical_clustering(grads, n_clusters)

        # 3) Learning quality; the global gradient reference is initialized at round 0 with the mean of client gradients / 3) 学习质量；全局梯度参考在第 0 轮用客户端梯度均值初始化
        lq = compute_learning_quality(accs)
        if global_grad_ref is None:
            global_grad_ref = torch.stack(grads).mean(0)

        # 4) Compute each client's global gain / 4) 计算每个客户端的全局增益
        gains = [0.0] * n_clients
        for i in range(n_clients):
            s = cosine_distance(grads[i], global_grad_ref).item()
            gains[i] = lq[i] * np.exp(-beta * s)

        # 5) Cluster-average gain; update scheduling thresholds every 30 rounds / 5) 簇平均增益，每 30 轮更新调度阈值
        cluster_members = {}
        for i, c in enumerate(cluster_labels):
            cluster_members.setdefault(int(c), []).append(i)
        gbar = {c: float(np.mean([gains[i] for i in members]))
                for c, members in cluster_members.items()}

        if t % 30 == 0:
            vals = list(gbar.values())
            omega1 = np.percentile(vals, 70) if vals else 0.0
            omega2 = np.percentile(vals, 30) if vals else 0.0
            # After threshold update, each cluster's participation tier stays fixed within the 30-round window / 阈值更新后，各簇参与档位在 30 轮窗口内保持固定
            cluster_tier = {}
            for c in cluster_members:
                if gbar[c] > omega1:
                    cluster_tier[c] = 1      # High gain: participate every round / 高增益：每轮参与
                elif gbar[c] > omega2:
                    cluster_tier[c] = 2      # Medium gain: participate every 2 rounds / 中增益：每2轮参与
                else:
                    cluster_tier[c] = 3      # Low gain: participate every 3 rounds / 低增益：每3轮参与

        # 6) Inter-cluster scheduling + intra-cluster sampling + aggregation / 6) 簇间调度 + 簇内采样 + 聚合
        vec_before = flatten_params(global_model)
        selected_states = []
        selected_weights = []
        for c, members in cluster_members.items():
            tier = cluster_tier[c]
            if tier == 1:
                eligible = True
            elif tier == 2 and t % 2 == 0:
                eligible = True
            elif tier == 3 and t % 3 == 0:
                eligible = True
            else:
                eligible = False
            if not eligible:
                continue
            # Sample representative clients within a cluster with gain-normalized probabilities / 簇内按增益归一化采样代表客户端
            w = np.array([gains[i] for i in members], dtype=np.float64)
            w = w / (w.sum() + 1e-12)
            n_pick = min(samples_per_cluster, len(members))
            picks = np.random.choice(members, size=n_pick, replace=False, p=w)
            for pick in picks:
                selected_states.append(local_models[pick].state_dict())
                selected_weights.append(gains[pick])

        if selected_states:
            if t == 0:
                # Round-1 aggregation follows TON-IOT: equal-weight average of all selected clients (1/n_selected) / 第一轮聚合参照 TON-IOT：所有被选中客户端等权平均（1/n_selected）
                sw = np.ones(len(selected_states), dtype=np.float64) / len(selected_states)
            else:
                # Subsequent rounds keep gain-weighted aggregation / 后续轮保持增益加权聚合
                sw = np.array(selected_weights, dtype=np.float64)
                sw = sw / sw.sum()
            new_state = {}
            for k in selected_states[0].keys():
                stacked = torch.stack(
                    [s[k].float() for s in selected_states])
                w = torch.tensor(sw, dtype=torch.float32).to(stacked.device)
                new_state[k] = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(0)
            global_model.load_state_dict(new_state)

        # Update the "latest global gradient reference" for the next round / 更新“最新全局梯度参考”供下一轮使用
        global_grad_ref = flatten_params(global_model) - vec_before

        # Sync global parameters to all local models / 同步全局参数到所有本地模型
        for i in range(n_clients):
            local_models[i].load_state_dict(global_model.state_dict())

        # Record this round's overall loss: the global model's loss on the training set / 记录本轮总体损失：全局模型在训练集上的损失
        if t == 0:
            loss_history.append(init_loss)
        else:
            loss_history.append(evaluate_loss(
                global_model, x_loss_eval, y_loss_eval, device,
                class_weights=class_weights))

        if verbose:
            acc = evaluate_accuracy(global_model, x_public, y_public, device)
            rl = loss_history[-1] if loss_history[-1] is not None else 0.0
            print(f"[FL] round {t:3d}/{rounds-1}  "
                  f"selected={len(selected_states):2d}  "
                  f"loss={rl:.4f}  public_acc={acc:.4f}")

    return global_model, loss_history
