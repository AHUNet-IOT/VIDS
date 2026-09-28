import argparse
import sys
import time
from pathlib import Path

# Add the project root (VeReMi) and the current directory (Federated_Learning) to sys.path,
# so that the common and py_func packages can be imported respectively.
# 将项目根目录（VeReMi）与当前目录（Federated_Learning）加入 sys.path，
# 以便分别导入 common 与 py_func 包。
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _p in (str(_ROOT), _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from py_func.hyperparams import get_hyperparams, get_file_name
from py_func.read_db import data_get
from py_func.create_model import load_model, model_num_parameters
from py_func.federated import run_federated_training


def compute_metrics(model, x, y, device, batch_size=1024):
    """Compute macro precision/recall/f1, accuracy, and the confusion matrix on the test set. / 在测试集上计算宏观 precision/recall/f1、accuracy 与混淆矩阵。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(device).unsqueeze(1)
            logits, _ = model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    from sklearn.metrics import (
        precision_recall_fscore_support, accuracy_score, confusion_matrix)
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average='macro', zero_division=0)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    return {'precision': p, 'recall': r, 'f1': f1, 'accuracy': acc,
            'confusion': cm, 'preds': preds, 'y': y}


def main():
    """Federated training main flow: parse arguments -> load data -> train the teacher model -> evaluate and save. / 联邦训练主流程：解析参数 -> 加载数据 -> 训练教师模型 -> 评估并保存。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-npz', type=str, default=str(_ROOT / 'Federated_Learning' / 'CarHacking_train.npz'),
                        help='训练集 npz')
    parser.add_argument('--test-npz', type=str, default=str(_ROOT / 'Federated_Learning' / 'CarHacking_test.npz'),
                        help='测试集 npz')
    parser.add_argument('--public-npz', type=str, default=str(_ROOT / 'Federated_Learning' / 'CarHacking_public.npz'),
                        help='公共集 npz（学习质量评估用，与训练集不重叠）')
    parser.add_argument('--quick', action='store_true', help='快速冒烟测试（少量样本/轮次）')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='从训练集中随机抽取的样本数（用于快速实验，默认不限制）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--local-epochs', type=int, default=None)
    parser.add_argument('--n-clients', type=int, default=None)
    parser.add_argument('--n-clusters', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--use-dp', action='store_true',
                        help='开启差分隐私（默认关闭）')
    args = parser.parse_args()

    # ===== Parameter configuration (centralized in py_func/hyperparams.py) / 参数配置（集中在 py_func/hyperparams.py）=====
    hp = get_hyperparams(quick=args.quick, max_samples=args.max_samples)
    # use_dp is a boolean switch, off by default; enabled only when --use-dp is explicitly passed. / use_dp 是布尔开关，默认关闭；仅当显式传入 --use-dp 时开启。
    # DP training uses Adam by default (per-parameter adaptive, more stable for noised gradients). / DP 训练默认使用 Adam（逐参数自适应，对加噪后的梯度更稳定）。
    if args.use_dp:
        hp['use_dp'] = True
        hp['optimizer'] = 'adam'
    for key in ('lr', 'rounds', 'local_epochs', 'n_clients', 'n_clusters',
                'optimizer', 'beta'):
        val = getattr(args, key)
        if val is not None:
            hp[key] = val
    print("联邦学习超参数：")
    for k, v in hp.items():
        print(f"  {k} = {v}")
    print(f"实验标识: {get_file_name(hp)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ===== Data loading and partitioning (py_func/read_db.py) / 数据加载与划分（py_func/read_db.py）=====
    data = data_get(hp, seed=args.seed,
                    train_npz=args.train_npz, test_npz=args.test_npz,
                    public_npz=args.public_npz)

    # ===== Model creation (py_func/create_model.py) / 模型创建（py_func/create_model.py）=====
    teacher_proto = load_model(num_classes=hp['num_classes'],
                               channels=hp['teacher_channels'],
                               reduction=hp['reduction'],
                               seq_len=hp['seq_len'])
    print(f"\n复杂模型参数量: {model_num_parameters(teacher_proto)}")

    # ===== Federated learning training (py_func/federated.py) / 联邦学习训练（py_func/federated.py）=====
    print("\n" + "=" * 70)
    print("联邦学习训练复杂模型 (SE-ResNet)")
    t0 = time.time()
    teacher, _ = run_federated_training(
        data['x_train'], data['y_train'], data['client_indices'],
        data['x_public'], data['y_public'],
        num_classes=hp['num_classes'], seq_len=hp['seq_len'],
        n_clusters=hp['n_clusters'],
        rounds=hp['rounds'], local_epochs=hp['local_epochs'],
        batch_size=hp['batch_size'], lr=hp['lr'], beta=hp['beta'],
        device=device, channels=hp['teacher_channels'], reduction=hp['reduction'],
        optimizer=hp['optimizer'], samples_per_cluster=hp['samples_per_cluster'],
        use_class_weights=hp['use_class_weights'],
        public_eval_size=hp['public_eval_size'],
        loss_eval_size=hp['loss_eval_size'],
        class_weight_mult=hp['class_weight_mult'],
        use_focal_loss=hp.get('use_focal_loss', False),
        focal_gamma=hp.get('focal_gamma', 2.0),
        use_dp=hp['use_dp'],
        dp_epsilon=hp['dp_epsilon'],
        dp_delta=hp['dp_delta'],
        dp_retention_ratio=hp['dp_retention_ratio'],
        dp_clip_threshold=hp['dp_clip_threshold'],
        dp_gamma1=hp['dp_gamma1'],
        dp_gamma2=hp['dp_gamma2'])
    print(f"联邦学习耗时: {time.time() - t0:.1f}s")

    teacher.eval()
    t_metrics = compute_metrics(teacher, data['x_test'], data['y_test'], device)
    print(f"\n[复杂模型] 测试集 accuracy={t_metrics['accuracy']:.4f}  "
          f"precision={t_metrics['precision']:.4f}  "
          f"recall={t_metrics['recall']:.4f}  f1={t_metrics['f1']:.4f}")

    # Save the teacher model for loading in the second-stage knowledge distillation; DP and non-DP use different file names / 保存教师模型，供第二阶段知识蒸馏加载；DP 与非 DP 使用不同文件名
    ckpt_dir = _ROOT / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = 'teacher_seresnet_dp.pt' if hp['use_dp'] else 'teacher_seresnet.pt'
    ckpt_path = ckpt_dir / ckpt_name
    torch.save(teacher.state_dict(), ckpt_path)
    print(f"\n教师模型已保存: {ckpt_path}")

    # Print the classification report and confusion matrix after training / 训练结束后打印分类报告和混淆矩阵
    from sklearn.metrics import classification_report
    print("\n" + "=" * 70)
    print("分类报告 (classification report):")
    print(classification_report(t_metrics['y'], t_metrics['preds'],
                                digits=4, zero_division=0))


if __name__ == '__main__':
    main()
