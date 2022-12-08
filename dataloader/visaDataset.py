import os

import matplotlib.pyplot as plt
from torchvision.io import read_image


class VisaDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image


if __name__ == "__main__":
    root = "/Users/rahul.rajendran/Desktop/Pytorch/anomaly_detection/datasets/visa_finetune/"
    out = VisaDataset(root)
    img = out.__getitem__(5)

    plt.imshow(img)
    plt.show()
