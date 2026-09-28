#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Add the project root (VeReMi) to sys.path so the shared module common can be imported / 将项目根目录（VeReMi）加入 sys.path，以便导入共享模块 common
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from common.data_utils import dirichlet_partition, make_public_set


def data_get(hp, seed=0, train_npz=None, test_npz=None, public_npz=None):
    """Load the training/test/public sets and partition clients, returning the data dictionary required for federated training. / 加载训练/测试/公共集并划分客户端，返回联邦训练所需的数据字典。"""
    td = np.load(train_npz, allow_pickle=True)
    x_train = td['x'].astype(np.float32)
    y_train = td['y'].astype(np.int64)

    ted = np.load(test_npz, allow_pickle=True)
    x_test = ted['x'].astype(np.float32)
    y_test = ted['y'].astype(np.int64)

    # Optional: limit the number of training samples (for quick experiments), sampling randomly without affecting the test set / 可选：限制训练样本量（用于快速实验），随机抽取且不影响测试集
    max_samples = hp.get('max_samples')
    if max_samples is not None and max_samples < len(y_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_train), size=max_samples, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    # Public set: if public_npz is provided (sharing the same data as the distillation training set), load it directly, / 公共集：若提供 public_npz（与蒸馏训练集共用同一份数据）则直接加载，
    # otherwise sample from the training set in a class-balanced way (backward compatible) / 否则从训练集中按类别均衡抽取（向后兼容）
    if public_npz is not None:
        pd = np.load(public_npz, allow_pickle=True)
        x_public = pd['x'].astype(np.float32)
        y_public = pd['y'].astype(np.int64)
    else:
        x_public, y_public = make_public_set(
            x_train, y_train, size=hp['public_size'], seed=seed)

    client_indices = dirichlet_partition(
        y_train, hp['n_clients'], alpha=hp['dirichlet_alpha'], seed=seed)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'x_public': x_public, 'y_public': y_public,
        'client_indices': client_indices,
        'class_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
    }
