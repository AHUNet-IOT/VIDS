#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clustering utility functions.

Contains parameter flattening, cosine distance, hierarchical clustering, and learning quality computation.

聚类相关函数模块。

包含：参数展平、余弦距离、层次聚类、学习质量计算。
"""

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


def flatten_params(model):
    """Flatten model parameters into a 1-D vector (CPU). / 把模型参数展开为一维向量（CPU）。"""
    vec = []
    for p in model.parameters():
        vec.append(p.detach().cpu().reshape(-1))
    return torch.cat(vec)


def cosine_distance(a, b):
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    na = a.norm() + 1e-12
    nb = b.norm() + 1e-12
    return 1.0 - torch.dot(a, b) / (na * nb)


def hierarchical_clustering(grads, n_clusters):
    """Perform hierarchical clustering on cosine distances between representative gradients and return cluster labels. / 基于代表梯度之间的余弦距离做层次聚类，返回簇标签列表。"""
    n = len(grads)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(grads[i], grads[j]).item()
            dist[i, j] = dist[j, i] = d
    n_clusters = min(n_clusters, n)
    if n_clusters <= 1 or n == 1:
        return list(range(n))
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average')
    return model.fit_predict(dist)


def compute_learning_quality(accs):
    """Compute normalized learning quality based on accuracy on the public set. / 按公共数据集上的准确率计算归一化学习质量。"""
    accs = np.asarray(accs, dtype=np.float64)
    if accs.std() < 1e-12:
        return np.ones_like(accs) / len(accs)
    z = (accs - accs.mean()) / accs.std()
    shifted = z + abs(z.min()) + 1.0
    return shifted / shifted.sum()
