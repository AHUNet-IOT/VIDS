# 两阶段车载入侵检测系统 代码文档

[English](README.md) | 中文

## 数据集下载

本项目适用于多种车载/车联网入侵检测数据集（如 CIC-EVSE、Car-Hacking、VeReMi 等）。请自行下载所需数据集，并完成数据清洗与特征选择：

- **Car-Hacking**：<http://ocslab.hksecurity.net/Datasets/car-hacking-dataset>
- **VeReMi**：<https://veremi-dataset.github.io/veremi>
- **CIC-EVSE**：<https://www.unb.ca/cic/datasets/evse-dataset-2024.html>

> 说明：原始数据集通常包含时间戳、ID、十六进制字段、缺失值、类别不平衡等问题，需先进行数据清洗（去重、缺失值处理、类别合并、归一化）与特征选择，再转换为本项目 `.npz` 格式（`x` 为特征矩阵、`y` 为标签）。

本项目实现一条完整的「联邦学习 → 知识蒸馏」流水线，并可选择性地在联邦训练阶段接入差分隐私（DP）。

整体思路：

1. **联邦学习**：多客户端（模拟多个 RSU/路侧单元）协作训练一个较大的教师模型（SE-ResNet）。
2. **知识蒸馏**：将训练好的教师模型蒸馏到一个轻量级学生模型（LightCNN），在保持性能的同时大幅降低模型体积与计算量。
3. **差分隐私（可选）**：在联邦训练阶段对客户端上传的梯度加噪，满足严格的 `(ε, δ)-DP` 上界。

---

## 1. 环境依赖

Python 3，主要依赖见 `requirements.txt`：

```
torch==2.7.1
numpy==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.6
```

安装：

```bash
pip install -r requirements.txt
```

> 说明：若开启差分隐私（`--use-dp`），还需要 `scipy`（用于精确 RDP 隐私预算核算）：

```bash
pip install scipy
```

---

## 2. 数据说明

> 说明：本仓库附带的 `.npz` 文件为 Car-Hacking 数据集的部分样本（受上传大小限制，非完整数据集），方便读者直接运行代码。

数据集样本为若干维特征 + 类别标签，特征数（序列长度）与类别数由 `Federated_Learning/py_func/hyperparams.py` 中的 `seq_len` 与 `num_classes` 配置。

`.npz` 文件格式统一为：

- `x`：形状 `[N, seq_len]` 的浮点特征矩阵
- `y`：形状 `[N]` 的整型标签

模型内部会将输入 reshape 为 `[N, 1, seq_len]`（1 通道 × 序列长度 seq_len）用于 1D 卷积。

各 `.npz` 文件默认路径（可在命令行覆盖）：

- 联邦训练集：`Federated_Learning/data_train.npz`
- 联邦测试集：`Federated_Learning/data_test.npz`
- 联邦公共集：`Federated_Learning/data_public.npz`
- 蒸馏训练集：`Knowledge_Distillation/data_public_kd.npz`
- 蒸馏测试集：`Knowledge_Distillation/data_kd_test.npz`

---

## 3. 快速开始

推荐按以下顺序执行，即可跑通完整流水线（无 DP）。提供两种运行方式：

### 方式一：全量运行

使用全部训练样本，默认 100 轮联邦训练：

```bash
cd <项目根目录>

# 第一步：联邦学习训练教师模型（无 DP，全量 100 轮）
python3 Federated_Learning/train_federated.py

# 第二步：知识蒸馏得到学生模型
python3 Knowledge_Distillation/train_distill.py
```

### 方式二：快速运行（无 DP）

联邦学习阶段仅使用约 10 万条训练样本、10 轮，便于快速验证：

```bash
python3 Federated_Learning/train_federated.py --max-samples 100000 --rounds 10

# 之后进行知识蒸馏
python3 Knowledge_Distillation/train_distill.py
```

---

## 4. 联邦学习训练（教师模型）

入口：`Federated_Learning/train_federated.py`

```bash
python3 Federated_Learning/train_federated.py [选项]
```

### 5.1 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--train-npz` | str | `Federated_Learning/data_train.npz` | 训练集路径 |
| `--test-npz` | str | `Federated_Learning/data_test.npz` | 测试集路径 |
| `--public-npz` | str | `Federated_Learning/data_public.npz` | 公共集路径 |
| `--quick` | flag | False | 小规模冒烟测试（5 万样本 / 5 客户端 / 3 轮 / 1 epoch） |
| `--max-samples` | int | None | 从训练集随机抽取的样本数（None = 全量） |
| `--seed` | int | 0 | 随机种子 |
| `--lr` | float | None | 学习率（None = 使用默认值 0.0001） |
| `--rounds` | int | None | 联邦轮数（None = 使用默认值 100） |
| `--local-epochs` | int | None | 每轮客户端本地 epoch 数（None = 2） |
| `--n-clients` | int | None | 客户端数量（None = 30） |
| `--n-clusters` | int | None | 聚类簇数（None = 9） |
| `--optimizer` | str | None | 优化器 `sgd` / `adam`（None = 默认 sgd） |
| `--beta` | float | None | 增益加权聚合参数（None = 0.5） |
| `--use-dp` | flag | False | 开启差分隐私（默认关闭） |

### 5.2 常用示例

```bash
# 全量训练（无 DP，30 客户端 / 9 簇 / 100 轮 / SGD）
python3 Federated_Learning/train_federated.py

# 快速运行（无 DP，约 10 万样本 / 10 轮）
python3 Federated_Learning/train_federated.py --max-samples 100000 --rounds 10

# 小规模快速验证
python3 Federated_Learning/train_federated.py --quick

# 指定轮次与客户端数
python3 Federated_Learning/train_federated.py --rounds 50 --n-clients 10

# 开启差分隐私（优化器会自动绑定为 adam）
python3 Federated_Learning/train_federated.py --use-dp

# 差分隐私快速版（5 万样本 / 10 轮 / 5 客户端）
python3 Federated_Learning/train_federated.py --quick --use-dp --rounds 10
```

### 5.3 训练流程

1. 加载训练集 / 测试集 / 公共集，按 Dirichlet 分布划分到各客户端。
2. 第 0 轮对客户端做层次聚类（基于代表梯度的余弦距离）。
3. 每轮执行增益感知的簇内采样 + 簇间动态调度。
4. 质量感知聚合得到新的全局模型（教师模型）。
5. 训练结束后在测试集上评估，打印总体指标、分类报告，并保存教师模型。

### 5.4 输出产物

| 文件 | 说明 |
|---|---|
| `checkpoints/teacher_seresnet.pt` | 无 DP 训练的教师模型 |
| `checkpoints/teacher_seresnet_dp.pt` | DP 训练的教师模型 |

---

## 5. 知识蒸馏（学生模型）

入口：`Knowledge_Distillation/train_distill.py`

蒸馏使用 `checkpoints/teacher_seresnet.pt` 作为教师模型（需先完成联邦训练）。

```bash
python3 Knowledge_Distillation/train_distill.py [选项]
```

### 6.1 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--optimizer` | str | `sgd` | 蒸馏优化器 `sgd` / `adam` |
| `--lr` | float | 0.01 | 学习率 |
| `--momentum` | float | 0.9 | SGD 动量（仅 `sgd` 生效） |

### 6.2 常用示例

```bash
# 默认：SGD(lr=0.01, momentum=0.9)
python3 Knowledge_Distillation/train_distill.py

# 换回 Adam（建议配合较小学习率）
python3 Knowledge_Distillation/train_distill.py --optimizer adam --lr 0.001
```

### 6.3 蒸馏要点

- 学生模型为 `LightCNN(width=4)`，参数量约 3.6K，远小于教师模型（约 330 万）。
- 蒸馏损失 = 交叉熵 + Hint 损失 + 软标签 KL。
- 温度由可学习参数 `tau` 经梯度反转层做课程化调度。
- 默认训练 **10 个随机种子 × 150 epoch**，并对 10 个模型做软投票 / 权重平均集成。

### 6.4 输出产物

| 文件 | 说明 |
|---|---|
| `checkpoints/student_s4_ens_{0..9}.pt` | 10 个不同种子的学生模型 |
| `checkpoints/student_s4_ensemble.pt` | 权重平均的集成学生模型 |

---

## 6. 差分隐私（可选）

在联邦训练阶段通过 `--use-dp` 开启。开启后：

- 每个客户端本地训练使用 **逐样本梯度裁剪 + 高斯加噪**（per-batch DP-SGD）。
- 噪声乘数 σ 由 **精确 RDP 核算**按当前数据规模反推，满足严格的隐私预算上界。
- 默认隐私预算：`ε = 8.0`、`δ = 1e-5`。
- 开启 DP 时优化器自动绑定为 `adam`（对加噪后的梯度更稳定）。

### 7.1 DP 相关超参数

集中在 `Federated_Learning/py_func/hyperparams.py` 中，可按需修改：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_dp` | False | 是否开启 DP（也可用 `--use-dp` 开启） |
| `dp_epsilon` | 8.0 | 隐私预算 ε 上界 |
| `dp_delta` | 1e-5 | 松弛项 δ |
| `dp_retention_ratio` | 0.6 | 掩码保留比例 r |
| `dp_clip_threshold` | 1.5 | 裁剪阈值 X |
| `dp_gamma1` | 0.9 | 一阶矩衰减系数 |
| `dp_gamma2` | 0.999 | 二阶矩衰减系数 |

### 7.2 示例

```bash
# DP 快速版：5 万样本 / 10 轮 / 5 客户端
python3 Federated_Learning/train_federated.py --quick --use-dp --rounds 10

# DP 全量版
python3 Federated_Learning/train_federated.py --use-dp
```

> 注意：DP 训练需要进行逐样本梯度计算，速度明显慢于普通训练；样本量 / 轮次越多，耗时越长。

---

## 7. 关键超参数说明

默认超参数集中在 `Federated_Learning/py_func/hyperparams.py` 的 `get_hyperparams()` 中。

| 类别 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| 数据 | `seq_len` | 10 | 序列长度（特征数） |
| 数据 | `num_classes` | 5 | 类别数 |
| 数据 | `test_ratio` | 0.2 | 测试集比例 |
| 数据 | `public_size` | 100000 | 公共集大小 |
| 模型 | `teacher_channels` | (371, 414, 442) | 教师模型三阶段通道数 |
| 模型 | `reduction` | 16 | SE 模块压缩比 |
| 联邦 | `n_clients` | 30 | 客户端数量 |
| 联邦 | `n_clusters` | 9 | 聚类簇数 |
| 联邦 | `dirichlet_alpha` | 3 | Dirichlet 划分浓度参数 |
| 联邦 | `rounds` | 100 | 联邦训练轮数 |
| 联邦 | `local_epochs` | 2 | 每轮本地 epoch 数 |
| 联邦 | `batch_size` | 256 | 批量大小 |
| 联邦 | `lr` | 0.0001 | 学习率 |
| 联邦 | `optimizer` | sgd | 优化器（DP 时自动切 adam） |
| 联邦 | `beta` | 0.5 | 增益加权聚合参数 |
| 联邦 | `use_class_weights` | True | 是否使用类别加权损失 |
| 联邦 | `class_weight_mult` | [1,1,2,1,1] | 类别权重倍率 |

`--quick` 模式会自动覆盖为：`max_samples=50000`、`n_clients=5`、`n_clusters=3`、`rounds=3`、`local_epochs=1`。

---

## 8. 常见问题

**Q1：运行联邦训练报找不到 `.npz` 文件？**

确认 `Federated_Learning/` 目录下存在 `data_train.npz`、`data_test.npz`、`data_public.npz`，或通过 `--train-npz` / `--test-npz` / `--public-npz` 指定正确路径。

**Q2：知识蒸馏报找不到教师模型？**

蒸馏依赖 `checkpoints/teacher_seresnet.pt`，需先完成联邦训练。若用 DP 训练的教师模型，需手动指定为 `checkpoints/teacher_seresnet_dp.pt`。

**Q3：DP 训练为什么很慢？**

DP 需要对每个 batch 内的每个样本单独计算梯度（逐样本裁剪 + 加噪），计算量远大于普通反向传播，且与总样本量、轮次、本地 epoch 成正比。

**Q4：如何修改 DP 隐私预算？**

在 `Federated_Learning/py_func/hyperparams.py` 中修改 `dp_epsilon` 与 `dp_delta`，噪声乘数会自动重新反推。

---

## 注意事项

由于 GitHub 上传文件大小存在限制，本仓库仅上传了部分数据集，方便读者直接运行代码。同时，由于深度学习模型的训练受设备与数据的影响，最终得到的结果可能与论文中的结果（即使用完整数据集）存在一定的误差。
