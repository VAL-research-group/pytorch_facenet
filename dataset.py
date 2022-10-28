import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import *
from sklearn.model_selection import train_test_split

def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y


def load_dataset(train_dir, valid_dir=None, bs=128, val_bs=1, test_size=0.2):
    transform = Compose([
        transforms.Resize((160,160)),
        np.float32,
        transforms.ToTensor(),
        prewhiten
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    classes = train_dataset.classes
    if valid_dir != None:
        valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=transform)
        assert classes == valid_dataset.classes
    else:
        # データセットをtrainとvalidationに分割
        train_indices, val_indices = train_test_split(list(range(len(train_dataset.targets))), test_size=test_size, stratify=dataset.targets)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        valid_dataset = torch.utils.data.Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=bs,     shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=val_bs, shuffle=True, num_workers=4, drop_last=True)
    print("Load Data! train:{}, valid:{}, classes:{}".format(str(len(train_dataset)), str(len(valid_dataset)), str(len(classes))))
    return train_loader, valid_loader, classes


if __name__ == "__main__":
    train_loader, valid_loader, classes = load_dataset("data/VGG-Face2/data/train")