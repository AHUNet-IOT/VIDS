import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / 'Knowledge_Distillation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.models import SEResNet, LightCNN
from Knowledge_Distillation.distill_func import soft_ce, GradientReversalFunction, curriculum_decay

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS = 5
SEQ = 10
TCH = (371, 414, 442)
STCH = (8, 16, 32)

N_SEEDS = 10
EPOCHS = 150
BATCH = 256
LR = 0.01
OPTIMIZER = 'sgd'
MOMENTUM = 0.9
LAM_CE = 0.8
LAM_HINT = 0.1
LAM_KL = 0.1
T_START = 1.0
T_END = 20.0
TAU_INIT = -2.0
DECAY_LOOPS = 60
USE_ENSEMBLE = False  # Single model by default; when True, the ensemble model is the main result / 默认单模型；True 时以集成模型为主结果
TRAIN_FRAC = 1.0  # Fraction of the training set used for distillation (1.0=all; 0.1=only 10% of samples) / 蒸馏使用的训练集比例（1.0=全部；0.1=仅10%样本）
LOG_EPOCHS = 10  # Print training progress every this many epochs / 每多少个 epoch 打印一次训练进度

SRC = _ROOT / 'Knowledge_Distillation' / 'CarHacking_public_kd.npz'
TEST = _ROOT / 'Knowledge_Distillation' / 'CarHacking_kd_test.npz'
TEACHER_CKPT = _ROOT / 'checkpoints' / 'teacher_seresnet.pt'
OUT_DIR = _ROOT / 'checkpoints'


def load(p):
    """Load features x and labels y from an npz file (uniformly converted to float32 / int64). / 从 npz 加载特征 x 与标签 y（统一转为 float32 / int64）。"""
    d = np.load(p, allow_pickle=True)
    return d['x'].astype(np.float32), d['y'].astype(np.int64)


def teacher_out(teacher, x):
    """Run the teacher model forward; return logits and three block hidden features (mean-pooled along the time dimension). / 前向教师模型，返回 logits 与三个块隐藏特征（沿时间维 mean 池化）。"""
    teacher.eval()
    Z, H1, H2, H3 = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024]).to(DEV).unsqueeze(1)
            z, h = teacher(xb)
            Z.append(z.cpu().numpy())
            H1.append(h[0].mean(dim=2).cpu().numpy())
            H2.append(h[1].mean(dim=2).cpu().numpy())
            H3.append(h[2].mean(dim=2).cpu().numpy())
    return (np.concatenate(Z, 0).astype(np.float32),
            np.concatenate(H1, 0), np.concatenate(H2, 0), np.concatenate(H3, 0))


def train(x, y, hints, zt, *, seed=0, optimizer=OPTIMIZER, lr=LR, momentum=MOMENTUM,
          lam_ce=LAM_CE, lam_hint=LAM_HINT, lam_kl=LAM_KL):
    """Train a single student model (one seed).

    Loss = λ_ce·cross-entropy + λ_hint·hint loss + λ_kl·soft-label KL;
    the temperature T is curriculum-scheduled from the learnable parameter tau via a gradient reversal layer.

    训练单个学生模型（一个种子）。
    损失 = λ_ce·交叉熵 + λ_hint·Hint 损失 + λ_kl·软标签 KL；
    温度 T 由可学习参数 tau 经梯度反转层做课程化调度。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(x)
    n_batches = (n + BATCH - 1) // BATCH
    student = LightCNN(1, CLASS, SEQ, width=4).to(DEV)
    tau = torch.nn.Parameter(torch.zeros(1, device=DEV) + TAU_INIT)
    params = list(student.parameters()) + [tau]
    if optimizer == 'sgd':
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam(params, lr=lr)
    xt_t = torch.from_numpy(x).to(DEV); yt_t = torch.from_numpy(y).to(DEV)
    zt_t = torch.from_numpy(zt).to(DEV)
    tg1_t = torch.from_numpy(hints[0].astype(np.float32)).to(DEV)
    tg2_t = torch.from_numpy(hints[1].astype(np.float32)).to(DEV)
    tg3_t = torch.from_numpy(hints[2].astype(np.float32)).to(DEV)
    for ep in range(1, EPOCHS + 1):
        decay = curriculum_decay(ep, DECAY_LOOPS)
        perm = np.random.permutation(n)
        student.train()
        ep_loss = 0.0
        t0 = time.time()
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = xt_t[idx].unsqueeze(1); yb = yt_t[idx]
            z, h_s = student(xb)
            T = T_START + T_END * torch.sigmoid(GradientReversalFunction.apply(tau, decay))
            loss = lam_ce * F.cross_entropy(z, yb)
            loss_hint = (F.mse_loss(h_s[0].mean(dim=2), tg1_t[idx])
                         + F.mse_loss(h_s[1].mean(dim=2), tg2_t[idx])
                         + F.mse_loss(h_s[2].mean(dim=2), tg3_t[idx]))
            loss = loss + lam_hint * loss_hint
            loss = loss + lam_kl * soft_ce(z, zt_t[idx], T)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        if ep % LOG_EPOCHS == 0 or ep == EPOCHS:
            print(f"  [KD] seed{seed} ep {ep:3d}/{EPOCHS} loss={ep_loss/n_batches:.4f} "
                  f"T={T.item():.3f} [{time.time()-t0:.1f}s]")
    return student


def recall_from_preds(y, preds):
    """Compute the per-class recall from predicted labels. / 由预测标签计算每个类别的召回率。"""
    _, r, _, _ = precision_recall_fscore_support(y, preds, labels=range(CLASS), average=None, zero_division=0)
    return r


def recall(model, x, y):
    """Run inference on data (x, y) and return the per-class recall. / 在数据 (x, y) 上推理模型，返回每类召回率。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024]).to(DEV).unsqueeze(1)
            preds.append(model(xb)[0].argmax(1).cpu().numpy())
    return recall_from_preds(y, np.concatenate(preds))


def softmax_probs(model, x):
    """Run inference on data x and return softmax probabilities (for ensemble soft voting). / 在数据 x 上推理模型，返回 softmax 概率（用于集成软投票）。"""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024]).to(DEV).unsqueeze(1)
            probs.append(F.softmax(model(xb)[0], dim=1).cpu().numpy())
    return np.concatenate(probs, 0)


def fmt(tag, r):
    """Format output: macro recall + per-class recall (N/DoS/Fuzzy/Gear/RPM). / 格式化输出：宏观召回率 + 各类别召回率（N/DoS/Fuzzy/Gear/RPM）。"""
    return (f"{tag:28s} macroR={r.mean():.4f} | "
            f"N={r[0]:.4f} DoS={r[1]:.4f} Fuzzy={r[2]:.4f} Gear={r[3]:.4f} RPM={r[4]:.4f}")


def main():
    """Main distillation flow: load the teacher model -> extract hints (PCA) and logits -> train multi-seed student models and ensemble them. / 蒸馏主流程：加载教师模型 -> 提取 Hint(PCA) 与 logits -> 训练多种子学生模型并集成。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default=OPTIMIZER, choices=['sgd', 'adam'],
                        help='蒸馏优化器（默认 sgd）')
    parser.add_argument('--lr', type=float, default=LR, help='学习率')
    parser.add_argument('--momentum', type=float, default=MOMENTUM,
                        help='SGD 动量（仅 sgd 生效）')
    parser.add_argument('--loss-ratio', type=str, default=f'{LAM_CE}:{LAM_HINT}:{LAM_KL}',
                        help='损失权重比例 λ_ce:λ_hint:λ_kl')
    args = parser.parse_args()

    lam_ce, lam_hint, lam_kl = [float(v) for v in args.loss_ratio.split(':')]
    print(f"损失权重比例 λ_ce:λ_hint:λ_kl = {lam_ce}:{lam_hint}:{lam_kl}")

    torch.manual_seed(0)
    np.random.seed(0)
    print(f"device: {DEV}")
    src = load(SRC)
    xt, yt = load(TEST)

    teacher = SEResNet(1, CLASS, SEQ, channels=TCH).to(DEV)
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEV))
    teacher.eval()

    x, y = src[0], src[1]

    if TRAIN_FRAC < 1.0:
        n_use = max(1, int(len(x) * TRAIN_FRAC))
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(x))[:n_use]
        x, y = x[idx], y[idx]
        print(f"subsample train: {n_use}/{len(src[0])} ({TRAIN_FRAC:.0%})")

    # The 10 CarHacking features are already normalized to [0,1] (no missing/derived features), so the raw features are used directly / CarHacking 的 10 个特征已归一化到 [0,1]（无缺失/派生特征），直接使用原始特征
    Xa = x
    Xt_a = xt

    Zt, H1, H2, H3 = teacher_out(teacher, x)
    pca1 = PCA(n_components=STCH[0]).fit(H1)
    pca2 = PCA(n_components=STCH[1]).fit(H2)
    pca3 = PCA(n_components=STCH[2]).fit(H3)
    print(f"hint PCA ev: h1->{STCH[0]}={pca1.explained_variance_ratio_.sum():.4f} "
          f"h2->{STCH[1]}={pca2.explained_variance_ratio_.sum():.4f} "
          f"h3->{STCH[2]}={pca3.explained_variance_ratio_.sum():.4f}")
    T1 = pca1.transform(H1).astype(np.float32)
    T2 = pca2.transform(H2).astype(np.float32)
    T3 = pca3.transform(H3).astype(np.float32)
    hints = (T1, T2, T3)

    models = []
    r_seeds = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_train = time.time()
    for seed in range(N_SEEDS):
        t0 = time.time()
        m = train(Xa, y, hints, Zt, seed=seed,
                  optimizer=args.optimizer, lr=args.lr, momentum=args.momentum,
                  lam_ce=lam_ce, lam_hint=lam_hint, lam_kl=lam_kl)
        ckpt = OUT_DIR / f'student_s4_ens_{seed}.pt'
        torch.save(m.state_dict(), ckpt)
        models.append(m)
        r = recall(m, Xt_a, yt)
        r_seeds.append(r)
        print(fmt(f"seed{seed}", r), f" [{time.time()-t0:.0f}s]  saved={ckpt.name}")
    train_secs = time.time() - t_train
    print(f"TOTAL distillation time: {train_secs:.1f}s = {train_secs/60:.2f} min  (N_SEEDS={N_SEEDS})")

    r_stack = np.stack(r_seeds, 0)
    macro_per_seed = r_stack.mean(1)
    print(fmt("MEAN", r_stack.mean(0)) + f" | macroR std={macro_per_seed.std():.4f}")

    probs = np.stack([softmax_probs(m, Xt_a) for m in models], 0).mean(0)
    ens_preds = probs.argmax(1)
    r_ens = recall_from_preds(yt, ens_preds)
    print(fmt(f"ENSEMBLE({N_SEEDS})", r_ens))

    # Save the weight-averaged ensemble model (a single state_dict that can be loaded directly) / 保存权重平均的集成模型（单一 state_dict，可直接加载）
    avg_state = {}
    for k in models[0].state_dict().keys():
        avg_state[k] = sum(m.state_dict()[k].float() for m in models) / N_SEEDS
    ens_ckpt = OUT_DIR / 'student_s4_ensemble.pt'
    torch.save(avg_state, ens_ckpt)

    ens_model = LightCNN(1, CLASS, SEQ, width=4).to(DEV)
    ens_model.load_state_dict(avg_state)
    r_wavg = recall(ens_model, Xt_a, yt)
    print(fmt("ENSEMBLE(wavg)", r_wavg) + f"  saved={ens_ckpt.name}")

    # Main result: single model (MEAN) by default; use the ensemble (weight averaging) when USE_ENSEMBLE=True / 主结果：默认单模型（MEAN），USE_ENSEMBLE=True 时用集成（权重平均）
    if USE_ENSEMBLE:
        print(fmt("MAIN(ensemble-wavg)", r_wavg))
    else:
        print(fmt("MAIN(single, mean)", r_stack.mean(0)))


if __name__ == '__main__':
    main()
