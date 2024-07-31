from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type:ignore
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix  # type:ignore
from torch.utils.data import DataLoader, Subset
from torchvision import datasets  # type:ignore
from tqdm import tqdm

from model import Resnet18
from transform import ImageTransform


def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    classes: list[str],
    saving_dir: Path,
    epoch: int,
    device: torch.device,
) -> float:
    """
    Args:
        model: テストを行う対象の分類モデル
        test_dataloader: テスト画像のデータローダー
        classes: クラス名のリスト
        saving_dir: 結果を格納するディレクトリ
        epoch: 今何エポック目か
    """

    model.to(device)

    with torch.no_grad():

        y_pred_all = np.array([])
        y_true_all = np.array([])

        for images, labels in tqdm(test_dataloader, desc="testing"):

            images = images.to(device)
            labels = labels.to(device)
            out = model(images)

            y_pred = np.argmax(out.cpu().detach().numpy(), axis=1)
            y_true = labels.cpu().detach().numpy()

            y_pred_all = np.concatenate([y_pred_all, y_pred])
            y_true_all = np.concatenate([y_true_all, y_true])

    # 混同行列の描画
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(saving_dir / f"cm_epoch{epoch+1}.png")

    # 正解率
    acc = np.count_nonzero((y_pred_all - y_true_all) == 0) / len(y_true_all)

    return acc


if __name__ == "__main__":

    image_transform = ImageTransform()

    # データセットの入手
    test_dataset = datasets.CIFAR10(
        "./cifar-10",
        train=False,
        download=True,
        transform=image_transform("test"),
    )

    subset_indices = list(range(0, len(test_dataset), 30))
    test_dataloader = DataLoader(
        Subset(test_dataset, subset_indices), batch_size=32, shuffle=True
    )

    # ネットワークの定義
    model = Resnet18(num_classes=10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(model, test_dataloader, test_dataset.classes, Path("result"), 3, device)
