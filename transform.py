from typing import Literal

from torchvision import transforms  # type: ignore

SIZE = 224  # resnetの画像サイズ
MEAN = (0.485, 0.456, 0.406)  # ImageNetに基づく画素値の平均値
STD = (0.229, 0.224, 0.225)  # ImageNetに基づく画素値の標準偏差


class ImageTransform:
    """
    カラー画像に対して、学習時と推論時それぞれで適切な画像前処理関数を渡すクラス
    """

    def __init__(self):

        self.transform_dict = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=SIZE, scale=(0.5, 1.0)
                    ),  # スケールaugmentation
                    transforms.RandomHorizontalFlip(),  # 水平方向の反転
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ]
            ),
        }

    def __call__(self, mode: Literal["train", "test"]):
        return self.transform_dict[mode]
