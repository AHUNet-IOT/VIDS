"""Data partitioning utilities.

Partition the training set among multiple clients using a Dirichlet distribution (simulating non-IID RSUs), and generate a public dataset for evaluating the learning quality.

数据划分工具。

用 Dirichlet 分布将训练集划分给多个客户端（模拟非独立同分布的 RSU），并生成一个用于评估“学习质量”的公共数据集。
"""

import numpy as np


def dirichlet_partition(y, n_clients, alpha=0.5, seed=0):
    """Partition training samples into n_clients clients according to a Dirichlet distribution.

    Return client_indices, where client_indices[i] is the array of sample indices held by the i-th client.

    按 Dirichlet 分布将训练样本划分到 n_clients 个客户端。

    返回 client_indices，其中 client_indices[i] 为第 i 个客户端拥有的样本下标数组。
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]

    for c in classes:
        cidx = np.where(y == c)[0]
        rng.shuffle(cidx)
        if len(cidx) == 0:
            continue
        props = rng.dirichlet([alpha] * n_clients)
        props = np.round(props * len(cidx)).astype(int)
        props[-1] = len(cidx) - props[:-1].sum()
        start = 0
        for i in range(n_clients):
            cnt = max(0, props[i])
            client_indices[i].append(cidx[start:start + cnt])
            start += cnt

    for i in range(n_clients):
        client_indices[i] = np.concatenate(client_indices[i]) if client_indices[i] else np.array([], dtype=np.int64)
    return client_indices


def make_public_set(x, y, size=5000, seed=0):
    """Generate a public dataset (sampled with balanced classes) used to evaluate each client's learning quality. / 生成公共数据集（按类别均衡采样），用于评估各客户端学习质量。"""
    rng = np.random.default_rng(seed)
    idx = []
    per_class = max(1, size // len(np.unique(y)))
    for c in np.unique(y):
        cidx = np.where(y == c)[0]
        take = min(per_class, len(cidx))
        idx.append(rng.choice(cidx, size=take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return x[idx], y[idx]
