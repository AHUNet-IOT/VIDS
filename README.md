# VIDS Code Documentation

English | [中文](README.zh-CN.md)

## Contact

If you run into any issues while using this project, feel free to reach out via email: ahufcy123@163.com.

## Dataset Download

This project applies to various in-vehicle / V2X intrusion detection datasets (e.g., CIC-EVSE, Car-Hacking, VeReMi). Please download the required dataset yourself and complete data cleaning and feature selection:

- **Car-Hacking**: <http://ocslab.hksecurity.net/Datasets/car-hacking-dataset>
- **VeReMi**: <https://veremi-dataset.github.io/veremi>
- **CIC-EVSE**: <https://www.unb.ca/cic/datasets/evse-dataset-2024.html>

> Note: Raw datasets usually contain timestamps, IDs, hexadecimal fields, missing values, class imbalance, etc. You must first perform data cleaning (deduplication, missing-value handling, class merging, normalization) and feature selection, then convert them into this project's `.npz` format (`x` is the feature matrix, `y` is the labels).

This project implements a federated learning pipeline for in-vehicle intrusion detection, and optionally integrates differential privacy (DP) during the federated training stage.

Overall workflow:

1. **Federated learning**: Multiple clients (simulating several RSUs / roadside units) collaboratively train a teacher model (SE-ResNet).
2. **Differential privacy (optional)**: Add noise to client-uploaded gradients during federated training to satisfy a strict `(ε, δ)-DP` upper bound.

---

## 1. Environment & Dependencies

Python 3. Main dependencies are listed in `requirements.txt`:

```
torch==2.7.1
numpy==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.6
```

Install:

```bash
pip install -r requirements.txt
```

> Note: If you enable differential privacy (`--use-dp`), `scipy` is also required (for exact RDP privacy budget accounting):

```bash
pip install scipy
```

---

## 2. Data Description

> Note: This repository does not include any dataset files. Please download the full dataset yourself, perform data preprocessing, and place the converted `.npz` files in the expected locations before training.

Each sample consists of a feature vector plus a class label. The number of features (sequence length) and the number of classes are configured via `seq_len` and `num_classes` in `Federated_Learning/py_func/hyperparams.py`.

The `.npz` file format is unified as:

- `x`: floating-point feature matrix of shape `[N, seq_len]`
- `y`: integer labels of shape `[N]`

Internally, the model reshapes the input to `[N, 1, seq_len]` (1 channel × sequence length `seq_len`) for 1D convolution.

Default paths of each `.npz` file (overridable via command line):

- Federated training set: `Federated_Learning/data_train.npz`
- Federated test set: `Federated_Learning/data_test.npz`
- Federated public set: `Federated_Learning/data_public.npz`

---

## 3. Quick Start

Run the federated learning stage to train the teacher model (without DP). Two run modes are provided:

### Mode 1: Full Run

Uses all training samples, with 100 federated rounds by default:

```bash
cd <project root>

# Federated learning to train the teacher model (no DP, full 100 rounds)
python3 Federated_Learning/train_federated.py
```

### Mode 2: Quick Run (without DP)

The federated learning stage uses only about 100,000 training samples over 10 rounds, for quick verification:

```bash
python3 Federated_Learning/train_federated.py --max-samples 100000 --rounds 10
```

---

## 4. Federated Learning Training (Teacher Model)

Entry point: `Federated_Learning/train_federated.py`

```bash
python3 Federated_Learning/train_federated.py [options]
```

### 4.1 Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train-npz` | str | `Federated_Learning/data_train.npz` | Training set path |
| `--test-npz` | str | `Federated_Learning/data_test.npz` | Test set path |
| `--public-npz` | str | `Federated_Learning/data_public.npz` | Public set path |
| `--quick` | flag | False | Small-scale smoke test (50k samples / 5 clients / 3 rounds / 1 epoch) |
| `--max-samples` | int | None | Number of samples randomly drawn from the training set (None = full) |
| `--seed` | int | 0 | Random seed |
| `--lr` | float | None | Learning rate (None = default 0.0001) |
| `--rounds` | int | None | Number of federated rounds (None = default 100) |
| `--local-epochs` | int | None | Local epochs per round per client (None = 2) |
| `--n-clients` | int | None | Number of clients (None = 30) |
| `--n-clusters` | int | None | Number of clusters (None = 9) |
| `--optimizer` | str | None | Optimizer `sgd` / `adam` (None = default sgd) |
| `--beta` | float | None | Gain-weighted aggregation parameter (None = 0.5) |
| `--use-dp` | flag | False | Enable differential privacy (disabled by default) |

### 4.2 Common Examples

```bash
# Full training (no DP, 30 clients / 9 clusters / 100 rounds / SGD)
python3 Federated_Learning/train_federated.py

# Quick run (no DP, ~100k samples / 10 rounds)
python3 Federated_Learning/train_federated.py --max-samples 100000 --rounds 10

# Small-scale quick verification
python3 Federated_Learning/train_federated.py --quick

# Specify rounds and number of clients
python3 Federated_Learning/train_federated.py --rounds 50 --n-clients 10

# Enable differential privacy (optimizer is automatically switched to adam)
python3 Federated_Learning/train_federated.py --use-dp

# Differential privacy quick version (50k samples / 10 rounds / 5 clients)
python3 Federated_Learning/train_federated.py --quick --use-dp --rounds 10
```

### 4.3 Training Process

1. Load the training / test / public sets and partition them among the clients via a Dirichlet distribution.
2. In round 0, perform hierarchical clustering on the clients (based on the cosine distance of representative gradients).
3. Each round performs gain-aware intra-cluster sampling + inter-cluster dynamic scheduling.
4. Quality-aware aggregation produces the new global model (teacher model).
5. After training, evaluate on the test set, print overall metrics and the classification report, and save the teacher model.

### 4.4 Output Artifacts

| File | Description |
|---|---|
| `checkpoints/teacher_seresnet.pt` | Teacher model trained without DP |
| `checkpoints/teacher_seresnet_dp.pt` | Teacher model trained with DP |

---

## 5. Differential Privacy (Optional)

Enable it during federated training via `--use-dp`. When enabled:

- Each client's local training uses **per-sample gradient clipping + Gaussian noise** (per-batch DP-SGD).
- The noise multiplier σ is back-solved from the current data scale via **exact RDP accounting**, satisfying a strict privacy budget upper bound.
- Default privacy budget: `ε = 8.0`, `δ = 1e-5`.
- When DP is enabled, the optimizer is automatically bound to `adam` (more stable for noisy gradients).

### 5.1 DP-Related Hyperparameters

Centralized in `Federated_Learning/py_func/hyperparams.py`; modify as needed:

| Parameter | Default | Description |
|---|---|---|
| `use_dp` | False | Whether to enable DP (can also be enabled via `--use-dp`) |
| `dp_epsilon` | 8.0 | Privacy budget ε upper bound |
| `dp_delta` | 1e-5 | Relaxation term δ |
| `dp_retention_ratio` | 0.6 | Mask retention ratio r |
| `dp_clip_threshold` | 1.5 | Clipping threshold X |
| `dp_gamma1` | 0.9 | First-moment decay factor |
| `dp_gamma2` | 0.999 | Second-moment decay factor |

### 5.2 Examples

```bash
# DP quick version: 50k samples / 10 rounds / 5 clients
python3 Federated_Learning/train_federated.py --quick --use-dp --rounds 10

# DP full version
python3 Federated_Learning/train_federated.py --use-dp
```

> Note: DP training requires per-sample gradient computation, which is significantly slower than normal training; the more samples / rounds, the longer it takes.

---

## 6. Key Hyperparameters

Default hyperparameters are centralized in `get_hyperparams()` in `Federated_Learning/py_func/hyperparams.py`.

| Category | Parameter | Default | Description |
|---|---|---|---|
| Data | `seq_len` | 10 | Sequence length (number of features) |
| Data | `num_classes` | 5 | Number of classes |
| Data | `test_ratio` | 0.2 | Test set ratio |
| Data | `public_size` | 100000 | Public set size |
| Model | `teacher_channels` | (371, 414, 442) | Channel counts of the teacher model's three stages |
| Model | `reduction` | 16 | SE block squeeze ratio |
| Federated | `n_clients` | 30 | Number of clients |
| Federated | `n_clusters` | 9 | Number of clusters |
| Federated | `dirichlet_alpha` | 3 | Dirichlet partitioning concentration parameter |
| Federated | `rounds` | 100 | Number of federated training rounds |
| Federated | `local_epochs` | 2 | Local epochs per round |
| Federated | `batch_size` | 256 | Batch size |
| Federated | `lr` | 0.0001 | Learning rate |
| Federated | `optimizer` | sgd | Optimizer (auto-switches to adam under DP) |
| Federated | `beta` | 0.5 | Gain-weighted aggregation parameter |
| Federated | `use_class_weights` | True | Whether to use class-weighted loss |
| Federated | `class_weight_mult` | [1,1,2,1,1] | Class weight multiplier |

The `--quick` mode automatically overrides: `max_samples=50000`, `n_clients=5`, `n_clusters=3`, `rounds=3`, `local_epochs=1`.

---

## 7. FAQ

**Q1: Federated training reports that `.npz` files cannot be found?**

Make sure `data_train.npz`, `data_test.npz`, `data_public.npz` exist in the `Federated_Learning/` directory, or specify the correct paths via `--train-npz` / `--test-npz` / `--public-npz`.

**Q2: Why is DP training so slow?**

DP requires computing the gradient for each individual sample within each batch (per-sample clipping + noise), which is far more expensive than ordinary backpropagation and scales with the total sample count, rounds, and local epochs.

**Q3: How do I modify the DP privacy budget?**

Modify `dp_epsilon` and `dp_delta` in `Federated_Learning/py_func/hyperparams.py`; the noise multiplier will be automatically back-solved.

---

## Notes

Due to GitHub's file-size limits, this repository does not include any dataset files. Please download the full dataset yourself and perform data preprocessing before training. In addition, since deep learning model training is affected by hardware and data, the final results may differ slightly from the results reported in the paper (which used the complete dataset).
