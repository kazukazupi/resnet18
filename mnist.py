from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets  # type:ignore
from torchvision import transforms  # type: ignore

from model import Resnet18
from train import train
from transform import SIZE

if __name__ == "__main__":

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
        batch_size=32,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        Subset(test_dataset, list(range(0, len(test_dataset), 30))),
        batch_size=32,
        shuffle=True,
    )

    # ネットワークの定義
    model = Resnet18(en_rgb=False, num_classes=10)
    train(
        model, train_dataloader, test_dataloader, train_dataset.classes, Path("result")
    )
