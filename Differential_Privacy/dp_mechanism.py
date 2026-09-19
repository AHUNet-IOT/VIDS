# -*- coding: utf-8 -*-
"""Adaptive differential privacy sanitizer.

Applies the following to gradients in order: importance mask -> normalization -> adaptive clipping -> Gaussian noise injection -> sparse recovery,
and maintains each client's first/second moment state with an exponential moving average (EMA).

自适应差分隐私 sanitizer。
对梯度依次执行：重要性掩码 -> 归一化 -> 自适应裁剪 -> 注入高斯噪声 -> 稀疏恢复，
并用指数滑动平均（EMA）维护每个客户端的一阶/二阶矩状态。
"""
import torch


class AdaptiveDPSanitizer:
    def __init__(self, retention_ratio=0.6, clip_threshold=4.0,
                 noise_multiplier=0.8, gamma1=0.9, gamma2=0.999,
                 kappa=1e-6):
        self.r = float(retention_ratio)           # Mask retention ratio / 掩码保留比例
        self.X = float(clip_threshold)            # Uniform clipping threshold / 统一裁剪阈值
        self.noise_multiplier = float(noise_multiplier)  # Noise multiplier σ / 噪声乘数 σ
        self.beta1 = float(gamma1)                # First-moment decay coefficient / 一阶矩衰减系数
        self.beta2 = float(gamma2)                # Second-moment decay coefficient / 二阶矩衰减系数
        self.kappa = float(kappa)                 # Numerical stability constant / 数值稳定常数
        self.mask = None                          # Flattened importance mask (bool) / 扁平重要性掩码（bool）
        self.layer_sizes = None                   # Number of parameters per layer / 各层参数数量
        self.m_vec = {}                           # client_id -> μk (first moment) / client_id -> μk（一阶矩）
        self.s_vec = {}                           # client_id -> vk (second moment) / client_id -> vk（二阶矩）

    def build_mask(self, accumulated_grads, layer_sizes=None):
        """Generate an importance mask by retaining the top r% elements based on accumulated gradient magnitude. / 基于累积梯度幅值保留 top r% 元素，生成重要性掩码。"""
        if not isinstance(accumulated_grads, torch.Tensor):
            accumulated_grads = torch.as_tensor(accumulated_grads, dtype=torch.float32)
        g = accumulated_grads.flatten().float().cpu()
        if layer_sizes is not None:
            layer_sizes = [int(n) for n in layer_sizes]
            if sum(layer_sizes) != g.numel():
                raise ValueError(
                    f"layer_sizes 之和 {sum(layer_sizes)} != 参数量 {g.numel()}")
            self.layer_sizes = layer_sizes
        else:
            self.layer_sizes = None
        k = max(1, int(self.r * g.numel()))
        _, topk_idx = torch.topk(g.abs(), k)
        mask = torch.zeros_like(g, dtype=torch.bool)
        mask[topk_idx] = True
        self.mask = mask
        return mask

    def initialize(self, client_id, grad_samples=None):
        """Initialize the first/second moments μk/vk using the collected gradient samples.

        Use the sample mean and variance as initial values so that the first normalization maps the gradients to unit scale;
        fall back to μ=0, v=1 when there are no samples.

        用收集到的梯度样本初始化一阶/二阶矩 μk/vk。
        用样本均值和方差作为初始值，使第一次归一化就能把梯度映射到单位尺度；
        无样本时回退为 μ=0、v=1。
        """
        if self.mask is None:
            raise RuntimeError("请先调用 build_mask() 生成掩码")
        total_dim = int(self.mask.numel())
        if grad_samples is not None:
            if not isinstance(grad_samples, torch.Tensor):
                grad_samples = torch.as_tensor(grad_samples, dtype=torch.float32)
            gs = grad_samples.float().cpu()
            if gs.ndim == 1:
                gs = gs.unsqueeze(0)
            if gs.ndim >= 2 and gs.shape[0] > 0 and gs.shape[1] == total_dim:
                m_init = gs.mean(dim=0)
                var_init = gs.var(dim=0, unbiased=False)
                self.m_vec[client_id] = m_init
                self.s_vec[client_id] = var_init.clamp(min=self.kappa)
                return
        # fallback: m_vec=0, s_vec=1 / fallback：m_vec=0、s_vec=1
        self.m_vec[client_id] = torch.zeros(total_dim)
        self.s_vec[client_id] = torch.ones(total_dim)

    def sanitize(self, gradient, client_id):
        """Apply to a single flattened gradient: mask -> normalization -> clipping -> noise injection -> recovery.

        Return the protected gradient with the same shape as the input.

        对单个扁平梯度执行：掩码 -> 归一化 -> 裁剪 -> 加噪 -> 恢复。
        返回与输入同形状的受保护梯度。
        """
        if self.mask is None:
            raise RuntimeError("请先调用 build_mask() 生成掩码")

        g = gradient.flatten().float()
        device = g.device
        mask_f = self.mask.float().to(device)

        m = self.m_vec.get(client_id)
        s = self.s_vec.get(client_id)
        if m is None or s is None:
            self.initialize(client_id)
            m = self.m_vec[client_id]
            s = self.s_vec[client_id]
        m = m.to(device)
        s = s.to(device)

        # 1) Mask / 1) 掩码
        g_masked = g * mask_f

        # 2) Normalization / 2) 归一化
        centered = (g_masked - m) / (torch.sqrt(s) + self.kappa)

        # 3) Adaptive clipping (uniform threshold X) / 3) 自适应裁剪（统一阈值 X）
        clip_factor = torch.clamp(centered.norm(2) / self.X, min=1.0)
        hat = centered / clip_factor

        # 4) Inject Gaussian noise N(0, (σ̂·X)²) / 4) 注入高斯噪声 N(0, (σ̂·X)²)
        hat = hat + torch.randn_like(hat) * (self.noise_multiplier * self.X)

        # 5) Sparse recovery / 5) 稀疏恢复
        final = (hat * (torch.sqrt(s) + self.kappa) + m) * mask_f

        # 6) Update the moment estimates with the recovered gradient without noise / 6) 用未加噪的恢复梯度更新矩估计
        recovered = (centered * (torch.sqrt(s) + self.kappa) + m) * mask_f
        self._update_m_s(client_id, recovered, device)

        return final

    def sanitize_batch(self, per_sample_grads, client_id, device=None):
        """Perform per-batch processing on a batch of per-sample flattened gradients.

        Flow: mask -> normalization -> per-sample clipping -> add noise to the clipped mean (σ = noise_multiplier·X/B) -> restore scale.
        `per_sample_grads` is the list of flattened gradients (D-dimensional tensors) for each sample in the batch.

        Return the protected batch-averaged gradient (D-dimensional) for `optimizer.step()`.

        对一批 per-sample 扁平梯度执行 per-batch 处理。
        流程：掩码 -> 归一化 -> 逐样本裁剪 -> 对裁剪均值加噪（σ = noise_multiplier·X/B）-> 恢复尺度。
        `per_sample_grads` 为该 batch 内每个样本的扁平梯度列表（D 维张量）。
        返回受保护的 batch 平均梯度（D 维），供 `optimizer.step()` 使用。
        """
        if self.mask is None:
            raise RuntimeError("请先调用 build_mask() 生成掩码")
        if not per_sample_grads:
            raise ValueError("per_sample_grads 不能为空")

        if device is None:
            device = per_sample_grads[0].device

        # Initialize first/second moments m=0, s=1 (not initialized from pre-training samples) / 一阶/二阶矩初始化 m=0、s=1（不经预训练样本初始化）
        if client_id not in self.m_vec:
            self.m_vec[client_id] = torch.zeros(self.mask.numel())
            self.s_vec[client_id] = torch.ones(self.mask.numel())

        m = self.m_vec[client_id].to(device)
        s = self.s_vec[client_id].to(device)
        mask_f = self.mask.float().to(device)
        sqrt_s = torch.sqrt(s) + self.kappa

        B = len(per_sample_grads)
        clipped_sum = torch.zeros_like(m)
        recovered_sum = torch.zeros_like(m)

        for g in per_sample_grads:
            g = g.flatten().float().to(device)
            # Mask + normalization / 掩码 + 归一化
            centered = (g * mask_f - m) / sqrt_s
            # Per-sample adaptive clipping to the L2 threshold X / 逐样本自适应裁剪到 L2 阈值 X
            norm = centered.norm(2)
            clip = min(1.0, self.X / (norm + 1e-6))
            centered = centered * clip
            clipped_sum += centered
            # Restore scale (no noise), used to update the moment estimates / 恢复尺度（无噪声），用于更新矩估计
            recovered_sum += centered * sqrt_s + m

        clipped_mean = clipped_sum / float(B)
        # Noise standard deviation = noise_multiplier * clip_norm / batch_size / 噪声标准差 = noise_multiplier * clip_norm / batch_size
        sigma = self.noise_multiplier * self.X / float(B)
        noisy_mean = clipped_mean + torch.randn_like(clipped_mean) * sigma
        # Sparse recovery, applying the mask again / 稀疏恢复，再次施加掩码
        final = (noisy_mean * sqrt_s + m) * mask_f

        recovered_mean = recovered_sum / float(B)
        self._update_m_s(client_id, recovered_mean, device)
        return final

    def _update_m_s(self, client_id, recovered, device):
        """Update moment estimates: μk←β1·μk+(1-β1)·g; vk←β2·vk+(1-β2)·(g-μk)². / 更新矩估计：μk←β1·μk+(1-β1)·g；vk←β2·vk+(1-β2)·(g-μk)²。"""
        m_old = self.m_vec[client_id].to(device).clone()
        m_new = self.beta1 * m_old + (1.0 - self.beta1) * recovered
        grad_diff = (recovered - m_old) ** 2
        var_est = torch.clamp(grad_diff, min=self.kappa, max=1e6)
        s_old = self.s_vec[client_id].to(device)
        s_new = self.beta2 * s_old + (1.0 - self.beta2) * var_est
        # Store the state back to CPU uniformly to avoid cross-device dirty reads / 状态统一存回 CPU，避免跨设备脏读
        self.m_vec[client_id] = m_new.detach().cpu()
        self.s_vec[client_id] = s_new.detach().cpu()
