#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# ============================================================
# 1. DataLoader
# ============================================================

def load_mnist_train_data(train_batch_size: int = 128,
                          root: str = "./data/mnist") -> DataLoader:
    """基于 torchvision 的简单 MNIST 训练集 loader.
    如果你已经在别处写了自己的 train_loader（比如和 ADBA 一致），
    直接把这个函数内容换成你的实现就行。
    """
    transform = transforms.ToTensor()
    train_dataset = dsets.MNIST(
        root=root,
        train=True,
        download=True,      # 本地已有数据时不会重复下载
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return train_loader


def load_mnist_test_data(test_batch_size: int = 256,
                         root: str = "./data/mnist") -> DataLoader:
    """你给出的 test_loader 思路，这里做了点小扩展。"""
    transform = transforms.ToTensor()
    test_dataset = dsets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader


# ============================================================
# 2. WideResNet spectral normalization
# ============================================================

class BasicBlockSN(nn.Module):
    """带 spectral norm 的 WideResNet 基本 residual block."""
    def __init__(self, in_planes, out_planes, stride=1, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = spectral_norm(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False)
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = spectral_norm(
            nn.Conv2d(out_planes, out_planes, kernel_size=3,
                      stride=1, padding=1, bias=False)
        )
        self.drop_rate = drop_rate

        if in_planes != out_planes or stride != 1:
            self.shortcut = spectral_norm(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=stride, padding=0, bias=False)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        residual_source = out if self.shortcut is not None else x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual_source)
        else:
            residual = x

        return out + residual


class NetworkBlockSN(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes,
                 block, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            s = stride if i == 0 else 1
            layers.append(block(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride=s,
                drop_rate=drop_rate
            ))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetSN(nn.Module):
    def __init__(self,
                 depth: int = 16,
                 widen_factor: int = 4,
                 num_classes: int = 10,
                 input_channels: int = 1,
                 drop_rate: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "(depth-4) % 6 == 0"
        n = (depth - 4) // 6
        k = widen_factor

        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = spectral_norm(
            nn.Conv2d(input_channels, n_stages[0],
                      kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.block1 = NetworkBlockSN(n, n_stages[0], n_stages[1],
                                     BasicBlockSN, stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlockSN(n, n_stages[1], n_stages[2],
                                     BasicBlockSN, stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlockSN(n, n_stages[2], n_stages[3],
                                     BasicBlockSN, stride=2, drop_rate=drop_rate)

        self.bn = nn.BatchNorm2d(n_stages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = spectral_norm(nn.Linear(n_stages[3], num_classes))

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================
# 3. Lipschitz
# ============================================================

def lipschitz_penalty(model: nn.Module,
                      inputs: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
    x = inputs.detach().clone()
    x.requires_grad_(True)

    logits = model(x)                      # [B, C]
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze()  # [B]

    grad = torch.autograd.grad(
        true_logits.sum(),
        x,
        create_graph=True
    )[0]                                    # [B, 1, 28, 28]

    grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)      # [B]
    return grad_norm.mean()


# ============================================================
# 4. train
# ============================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    lambda_lip: float) -> Tuple[float, float]:
    model.train()
    ce_meter, acc_meter = 0.0, 0.0
    n_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)

        lip_loss = lipschitz_penalty(model, x, y)
        loss = ce_loss + lambda_lip * lip_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == y).sum().item()
            bs = x.size(0)
            n_samples += bs
            ce_meter += ce_loss.item() * bs
            acc_meter += correct

    return ce_meter / n_samples, acc_meter / n_samples


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce_meter, acc_meter = 0.0, 0.0
    n_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=1)
        correct = (pred == y).sum().item()
        bs = x.size(0)
        n_samples += bs
        ce_meter += ce_loss.item() * bs
        acc_meter += correct

    return ce_meter / n_samples, acc_meter / n_samples


# ============================================================
# 5. main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 超参数（你可以根据需要改）
    num_epochs = 20
    batch_size = 128
    test_batch_size = 256
    lr = 0.001
    weight_decay = 5e-4
    lambda_lip = 0.1

    train_loader = load_mnist_train_data(batch_size, root="./data/mnist")
    test_loader = load_mnist_test_data(test_batch_size, root="./data/mnist")

    model = WideResNetSN(depth=16, widen_factor=4,
                         num_classes=10, input_channels=1,
                         drop_rate=0.0).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_ce, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, lambda_lip
        )
        test_ce, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"time {dt:.1f}s | "
            f"train CE {train_ce:.4f} acc {train_acc*100:.2f}% | "
            f"test CE {test_ce:.4f} acc {test_acc*100:.2f}%"
        )

        best_test_acc = max(best_test_acc, test_acc)

    print(f"Best test accuracy: {best_test_acc*100:.2f}%")

    torch.save(model.state_dict(), "wrn_mnist_lipschitz.pth")
    print("Model saved to wrn_mnist_lipschitz.pth")


if __name__ == "__main__":
    main()
