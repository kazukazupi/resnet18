from pathlib import Path
from test import test

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    classes: list[str],
    saving_dir: Path,
    lr: float = 0.001,
    total_epochs: int = 20,
):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    saving_dir.mkdir(exist_ok=True)

    for epoch in range(total_epochs):

        count_correct = 0
        total_count = 0

        for images, labels in tqdm(train_dataloader, desc=f"epoch{epoch+1}"):

            # 計算の準備
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 学習
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # 推論結果の比較
            y_pred = np.argmax(out.cpu().detach().numpy(), axis=1)
            y_true = labels.cpu().detach().numpy()
            count_correct += np.count_nonzero((y_pred - y_true) == 0)
            total_count += len(y_true)

        # テスト
        test_acc = test(model, test_dataloader, classes, saving_dir, epoch, device)
        torch.save(model.state_dict(), saving_dir / f"model_epoch{epoch+1}.pth")

        train_acc = count_correct / total_count

        print(f"train acc: {(train_acc) * 100:04}%")
        print(f"test acc: {test_acc * 100:04}%")
