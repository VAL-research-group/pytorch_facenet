import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import *

def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y

def load_dataset():
    transform = Compose([
        transforms.Resize((160,160)),
        np.float32,
        transforms.ToTensor(),
        prewhiten
    ])
    data = torchvision.datasets.ImageFolder(root='./datasets', transform=transform)
    train_size = int(len(data)* 0.8)
    valid_size = len(data) - train_size
    train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.val_batch_size, shuffle=True, num_workers=4)
    print("Load Data! train:{}, valid:{}".format(str(len(train_data)), str(len(valid_data))))
    return train_loader, valid_loader