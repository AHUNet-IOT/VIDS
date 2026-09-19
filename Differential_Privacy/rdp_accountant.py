# -*- coding: utf-8 -*-
"""Privacy budget accounting for Rényi differential privacy (RDP).

Used to derive the Gaussian noise multiplier σ̂ from a preset (ε, δ)-DP budget and verify the accumulated privacy loss.
Uses the exact RDP formula for the Poisson subsampled Gaussian mechanism.

Rényi 差分隐私（RDP）隐私预算核算。
用于根据预设的 (ε, δ)-DP 预算反推高斯噪声乘数 σ̂，并验证累积隐私损失。
采用 Poisson 子采样高斯机制（Subsampled Gaussian Mechanism）的精确 RDP 公式。
"""
import math

import numpy as np

try:
    from scipy import special
except ImportError:  # pragma: no cover
    special = None

# Common RDP order (α) values, used to pick the optimal α when converting to (ε, δ)-DP / 常用的 RDP 阶数（α）取值，用于在转 (ε, δ)-DP 时取最优 α
_DEFAULT_ORDERS = np.array([
    1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 32.0, 64.0
], dtype=np.float64)


def compute_rdp_single(sample_rate, noise_multiplier, alpha):
    """Compute the order-α RDP for a single step of the Poisson subsampled Gaussian mechanism.

    Formula: RDP(α) = 1/(α-1) · log[ (1-q) + q · exp(α(α-1)/(2σ²)) ]
    where q is the sampling rate and σ is the noise multiplier (noise standard deviation = σ · sensitivity).

    计算单步 Poisson 子采样高斯机制的 α 阶 RDP。
    公式：RDP(α) = 1/(α-1) · log[ (1-q) + q · exp(α(α-1)/(2σ²)) ]
    其中 q 为采样率，σ 为噪声乘数（噪声标准差 = σ · 敏感度）。
    """
    q = float(sample_rate)
    sigma = float(noise_multiplier)
    alpha = float(alpha)
    if alpha == 1.0:
        return 0.0
    # log[(1-q) + q·exp(t)], explicitly handling the q=0/1 boundaries to avoid log(0) and exp overflow / log[(1-q) + q·exp(t)]，显式处理 q=0/1 边界，避免 log(0) 与 exp 溢出
    t = alpha * (alpha - 1.0) / (2.0 * sigma ** 2)
    if q >= 1.0:
        log_inner = t            # (1-q)=0 → inner = exp(t)
    elif q <= 0.0:
        log_inner = 0.0          # inner = 1
    else:
        log_inner = np.logaddexp(np.log1p(-q), np.log(q) + t)
    return log_inner / (alpha - 1.0)


def compute_rdp(sample_rate, noise_multiplier, steps, orders=None):
    """Compute the RDP vector accumulated over T steps.

    Return an array of the same length as orders, where each element is the accumulated RDP at the corresponding order α.

    计算 T 轮累积的 RDP 向量。
    返回与 orders 等长的数组，每个元素为对应 α 阶的累积 RDP。
    """
    if orders is None:
        orders = _DEFAULT_ORDERS
    rdp = np.array([compute_rdp_single(sample_rate, noise_multiplier, a) for a in orders])
    return rdp * steps


def rdp_to_epsilon(orders, rdp, delta):
    """Convert an RDP vector to the ε upper bound for (ε, δ)-DP (taking the optimal α). / 将 RDP 向量转换为 (ε, δ)-DP 的 ε 上界（取最优 α）。"""
    orders = np.asarray(orders, dtype=np.float64)
    rdp = np.asarray(rdp, dtype=np.float64)
    delta = float(delta)
    eps = rdp - np.log(delta) / (orders - 1.0)
    return float(np.min(eps))


def get_noise_multiplier(target_eps, target_delta, sample_rate, steps,
                         orders=None, sigma_low=0.01, sigma_high=100.0, tol=1e-3):
    """Binary-search the minimum noise multiplier σ̂ that satisfies (target_eps, target_delta)-DP.

    The returned σ̂ satisfies: ε ≤ target_eps after converting the T-step accumulated RDP to (ε, δ)-DP.

    二分搜索满足 (target_eps, target_delta)-DP 的最小噪声乘数 σ̂。
    返回的 σ̂ 满足：T 轮累积 RDP 转 (ε, δ)-DP 后 ε ≤ target_eps。
    """
    if orders is None:
        orders = _DEFAULT_ORDERS

    def feasible(sigma):
        rdp = compute_rdp(sample_rate, sigma, steps, orders)
        return rdp_to_epsilon(orders, rdp, target_delta) <= target_eps

    lo, hi = float(sigma_low), float(sigma_high)
    if feasible(lo):
        return lo
    if not feasible(hi):
        # The upper bound still does not satisfy the budget; return it with a hint (the caller can enlarge sigma_high) / 上界仍不满足，返回上界并提示（调用方可扩大 sigma_high）
        return hi

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ====================================================================
# Exact RDP formula (exact accounting for the subsampled Gaussian mechanism).
# The compute_rdp_single above is a loose upper bound and severely overestimates ε when the sampling rate is small,
# so the exact version below should be used when deriving σ.
# 精确 RDP 公式（子采样高斯机制的精确核算）。
# 上文的 compute_rdp_single 是宽松上界，采样率很小时会严重高估 ε，
# 因此反推 σ 时应使用下面的精确版本。
# ====================================================================

# RDP orders used for exact accounting / 精确核算使用的 RDP 阶数
_EXACT_ORDERS = np.arange(2.0, 64.0, 0.5)


def _log_add(logx, logy):
    """Log-domain addition: log(exp(logx) + exp(logy)), numerically stable implementation. / 对数域加法：log(exp(logx) + exp(logy))，数值稳定实现。"""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx, logy):
    """Log-domain subtraction: log(exp(logx) - exp(logy)), numerically stable implementation. / 对数域减法：log(exp(logx) - exp(logy))，数值稳定实现。"""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:
        return logx
    if logx == logy:
        return -np.inf
    try:
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx


def _log_erfc(x):
    """Log-domain complementary error function log(erfc(x)), preferring scipy's log_ndtr for accuracy. / 对数域互补误差函数 log(erfc(x))，优先用 scipy 的 log_ndtr 保证精度。"""
    if special is not None and hasattr(special, "log_ndtr"):
        return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)
    r = special.erfc(x)
    if r == 0.0:
        return (-math.log(math.pi) / 2 - math.log(x) - x ** 2 - .5 * x ** -2 +
                .625 * x ** -4 - 37. / 24. * x ** -6 + 353. / 64. * x ** -8)
    return math.log(r)


def _compute_log_a_int(q, sigma, alpha):
    """Accumulated log(A_α) terms of the exact RDP for integer α (subsampled Gaussian mechanism). / 整数 α 时精确 RDP 的 log(A_α) 累加项（子采样高斯机制）。"""
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef_i = (math.log(math.comb(alpha, i)) + i * math.log(q) +
                      (alpha - i) * math.log(1 - q))
        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)
    return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
    """log(A_α) of the exact RDP for non-integer α (using the infinite series expansion of the error function). / 非整数 α 时精确 RDP 的 log(A_α)（用误差函数无穷级数展开）。"""
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0
    z0 = sigma ** 2 * math.log(1 / q - 1) + .5
    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)
        log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1
        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30:
            break
    return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
    """Dispatch to the corresponding log(A_α) computation branch depending on whether α is an integer. / 按 α 是否为整数分发到对应的 log(A_α) 计算分支。"""
    if float(alpha).is_integer():
        return _compute_log_a_int(q, sigma, int(alpha))
    return _compute_log_a_frac(q, sigma, alpha)


def _compute_rdp_exact(q, sigma, alpha):
    """Single-step exact order-α RDP, handling the q=0/1 boundary cases. / 单步精确 α 阶 RDP，处理 q=0/1 边界情况。"""
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    if np.isinf(alpha):
        return np.inf
    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp_exact(sample_rate, noise_multiplier, steps, orders=None):
    """Exact accumulated RDP vector. / 精确累积 RDP 向量。"""
    if orders is None:
        orders = _EXACT_ORDERS
    q = float(sample_rate)
    sigma = float(noise_multiplier)
    rdp = np.array([_compute_rdp_exact(q, sigma, a) for a in orders])
    return rdp * steps


def get_noise_multiplier_exact(target_eps, target_delta, sample_rate, steps,
                               orders=None, sigma_low=0.01, sigma_high=1000.0,
                               tol=1e-3):
    """Binary-search the minimum noise multiplier σ̂ satisfying (ε, δ)-DP using the exact RDP formula. / 用精确 RDP 公式二分搜索满足 (ε, δ)-DP 的最小噪声乘数 σ̂。"""
    if orders is None:
        orders = _EXACT_ORDERS

    def feasible(sigma):
        rdp = compute_rdp_exact(sample_rate, sigma, steps, orders)
        return rdp_to_epsilon(orders, rdp, target_delta) <= target_eps

    lo, hi = float(sigma_low), float(sigma_high)
    if feasible(lo):
        return lo
    if not feasible(hi):
        return hi
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi
