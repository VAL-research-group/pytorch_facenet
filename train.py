
from cmath import inf
import os
import argparse
import torch
import torch.nn as nn
import datetime
from torch.optim import SGD, Adam
from tqdm import tqdm
from pathlib import Path
from model import FaceNet
from dataset import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
time_now   = datetime.datetime.now().strftime('%y%m%d_%H%M')

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean().item()


def train(args):
    output_dir = Path(args.output_dir) / time_now
    train_loader, valid_loader, classes = load_dataset(args.train_dir,
                                                       args.valid_dir,
                                                       bs=args.batch_size,
                                                       val_bs=args.val_batch_size)
    class_size = len(classes)
    model = FaceNet(class_size=class_size).to(device)
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    os.makedirs(str(output_dir), exist_ok=True)
    prev_loss = 999
    for epoch in range(1, args.epoch+1):
        train_loss = epoch_fn(epoch, model, optimizer, criterion, train_loader, is_train=True)
        valid_loss = epoch_fn(epoch, model, optimizer, criterion, valid_loader, is_train=False)
        filename = "%04d.pth"%(epoch)
        output_path = output_dir / filename
        if epoch > 1 and args.save_better_only and prev_loss > valid_loss:
            torch.save({'model': model.state_dict(), 'classes':classes}, str(output_path))
            prev_loss = valid_loss
    print("Finished!")


# 1epochあたりの処理
def epoch_fn(ep, model, optimizer, criterion, data_loader, is_train=True):
    tqdm_bar = tqdm(total = len(data_loader))
    if is_train:
        tqdm_bar.set_description("[Train Epoch %d]" % (ep))
        model.train()
    else:
        tqdm_bar.set_description("[Valid Epoch %d]" % (ep))
        model.eval()

    total_loss = 0.0
    total_acc = 0.0

    for i_batch, data in enumerate(data_loader):
        img, label = data
        img = img.to(device).float()
        label = label.to(device).long()
        pred = model(img)
        loss = criterion(pred, label)
        total_acc += accuracy(pred, label)
        total_loss += loss.item()
        tqdm_bar.set_postfix({
            'loss': total_loss/(i_batch+1),
            'acc': total_acc/(i_batch+1)
        }, refresh=True)
        tqdm_bar.update()
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_loss / data_loader.__len__() * len(data)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of epoch to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='input batch size for validation (default: 8)')
    parser.add_argument("--output_dir", type=str, default='./weights/',
                        help="path to folder where checkpoint of trained weights will be saved")
    parser.add_argument("--train_dir", type=str, default='data/VGG-Face2/data/adv/train',
                        help="path to folder where train dataset.")
    parser.add_argument("--valid_dir", type=str, default='data/VGG-Face2/data/adv/test',
                        help="path to folder where validation dataset.")
    parser.add_argument("--save_better_only", type=bool, default=True,
                        help="save only good weights")
    args = parser.parse_args()
    train(args)