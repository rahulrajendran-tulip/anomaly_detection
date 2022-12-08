import os

import torch
import torchvision.transforms as transforms
from PIL import Image


class ImageTransform:
    def __init__(self, is_train: bool, img_size=(224, 224)):
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAutocontrast(),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)
