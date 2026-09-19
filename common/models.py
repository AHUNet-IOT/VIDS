import torch
import torch.nn as nn


class SELayer(nn.Module):
    """SE attention module: applies global average pooling to each channel, then learns channel weights through two fully-connected layers and reweights the input. / SE 注意力模块：对每个通道做全局平均池化后，经两层全连接学习通道权重并加权。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = x.mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SEBasicBlock(nn.Module):
    """SE residual block: two 1D convolution layers + batch normalization + SE attention; uses a 1x1 convolution shortcut when the input and output channels differ. / SE 残差块：两层 1D 卷积 + BN + SE 注意力；输入输出通道不一致时用 1x1 卷积做捷径。"""

    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SELayer(out_channels, reduction)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return out


class SEResNet(nn.Module):
    """Teacher model: stem + three SE residual blocks + global average pooling + a fully-connected classification head.

    forward returns (logits, (h1, h2, h3)), where h1/h2/h3 are the hidden features of the three blocks, used for hidden-layer alignment during the knowledge distillation stage.

    教师模型：stem + 三个 SE 残差块 + 全局平均池化 + 全连接分类头。

    forward 返回 (logits, (h1, h2, h3))，其中 h1/h2/h3 为三个块的隐藏特征，供知识蒸馏阶段做隐藏层对齐使用。
    """

    def __init__(self, in_channels=1, num_classes=6, seq_len=17,
                 channels=(256, 512, 512), reduction=16):
        super().__init__()
        self.seq_len = seq_len
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.block1 = SEBasicBlock(channels[0], channels[0], reduction)
        self.block2 = SEBasicBlock(channels[0], channels[1], reduction)
        self.block3 = SEBasicBlock(channels[1], channels[2], reduction)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[2], num_classes)

    def forward(self, x):
        x = self.stem(x)
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        logits = self.fc(self.gap(h3).flatten(1))
        return logits, (h1, h2, h3)


class LightCNN(nn.Module):
    """Student model: three 1D convolution layers + a fully-connected classification head, with the width controlled by width (far fewer parameters than the teacher model). / 学生模型：三层 1D 卷积 + 全连接分类头，宽度由 width 控制（参数远少于教师模型）。"""

    def __init__(self, in_channels=1, num_classes=6, seq_len=17, width=4):
        super().__init__()
        c1, c2, c3 = 2 * width, 4 * width, 8 * width
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels, c1, 3, 1, 1)
        self.conv2 = nn.Conv1d(c1, c2, 3, 1, 1)
        self.conv3 = nn.Conv1d(c2, c3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(c3 * seq_len, num_classes)

    def forward(self, x):
        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2(h1))
        h3 = self.relu(self.conv3(h2))
        logits = self.fc(h3.flatten(1))
        return logits, (h1, h2, h3)


def count_parameters(model):
    """Return the total number of trainable parameters in the model. / 返回模型可训练参数总量。"""
    return sum(p.numel() for p in model.parameters())
