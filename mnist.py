import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets  # type:ignore
from torchvision import transforms  # type: ignore

from model import Resnet18
from train import train
from transform import SIZE

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning-rate", type=float, default=0.001)
    parser.add_argument("-t", "--total-epochs", type=int, default=10)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-s", "--saving-dir", type=str, default="result_mnist")

    args = parser.parse_args()

    # データセットの入手
    train_dataset = datasets.MNIST(
        "./mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor()]),
    )

    test_dataset = datasets.MNIST(
        "./mnist",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor()]),
    )

    # cpuでも現実的な時間で実行できるよう、データ数を1/30にしている
    train_dataloader = DataLoader(
        Subset(train_dataset, list(range(0, len(train_dataset), 30))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        Subset(test_dataset, list(range(0, len(test_dataset), 30))),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # ネットワークの定義
    model = Resnet18(en_rgb=False, num_classes=10)

    if args.model is not None:
        model.load_state_dict(torch.load(args.model))

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        classes=train_dataset.classes,
        saving_dir=Path(args.saving_dir),
        lr=args.learning_rate,
        total_epochs=args.total_epochs,
    )
