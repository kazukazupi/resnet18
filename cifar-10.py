import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets  # type:ignore

from model import Resnet18
from train import train
from transform import ImageTransform

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning-rate", type=float, default=0.001)
    parser.add_argument("-t", "--total-epochs", type=int, default=60)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-s", "--saving-dir", type=str, default="result_cifar10")

    args = parser.parse_args()

    image_transform = ImageTransform()

    # データセットの入手
    train_dataset = datasets.CIFAR10(
        "./cifar-10",
        train=True,
        download=True,
        transform=image_transform("train"),
    )

    test_dataset = datasets.CIFAR10(
        "./cifar-10",
        train=False,
        download=True,
        transform=image_transform("test"),
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # ネットワークの定義
    model = Resnet18(num_classes=10)

    if args.model is not None:
        model.load_state_dict(torch.load(args.model))

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        classes=test_dataset.classes,
        saving_dir=Path(args.saving_dir),
        lr=args.learning_rate,
        total_epochs=args.total_epochs,
    )
