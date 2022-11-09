import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import *
from facenet_pytorch import MTCNN
from torch.optim import SGD, Adam

from sticker import AttackModel
import glob
import os
from PIL import Image
from skimage import io
import copy
import argparse
from pathlib import Path
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

composed = transforms.Compose(
    [transforms.Resize((160, 160)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def main(args):
    topk = args.topk
    img = Image.open(args.input)

    # If required, create a face detection pipeline using MTCNN:
    transformed_img = composed(img).to(device) # 顔部分をクロップ

    # class_names = classname(args.label)

    model = AttackModel(args.ckpt, device=device)
    idx2label = model.classes.tolist()

    label2idx = {}
    for i, l in enumerate(idx2label):
        label2idx[l] = i

    is_targeted = False if args.target == '-1' or 'notarget' else True
    target_label_idx = label2idx[args.target] if args.target in label2idx else int(args.target)
    print('Target  class: %s, idx: %03d'%(idx2label[target_label_idx], target_label_idx))
    # 正解データの準備
    with torch.no_grad():
        print("Prediction result (Before the attack):")
        predict_top_N(model, transformed_img, topk, idx2label, False)
        pred = torch_to_numpy(model.base_model(transformed_img.unsqueeze(0)).squeeze())
    true_label_idx = int(np.argsort(pred)[-1])



    # Train model
    lr = args.lr
    optimizer = Adam(model.parameters())
    with tqdm(range(1, args.epoch+1)) as pbar:
        for epoch in pbar:
            pbar.set_description("[Epoch %d]" % (epoch))
            if (epoch) % args.lr_decay_interval == 0:
                lr *= args.lr_decay_rate

            pred = model(transformed_img.unsqueeze(0))
            loss = compute_loss(pred, true_label_idx, target_label_idx, is_targeted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss':loss.item()})
            for param in model.parameters():
                if param.requires_grad == True:
                    param.data = torch.clamp(param.data - param.grad.data * lr, min=0.0, max=1.0)

            with torch.no_grad():
                half_img = model.image_dot(transformed_img.unsqueeze(0))

            os.makedirs(args.half, exist_ok=True)
            half_dir = Path(args.half)
            half_filename = '%04d.jpg'%(epoch)
            save_img(half_img, str(half_dir/half_filename))

    print("Prediction result (after the attack):")
    predict_top_N(model, transformed_img, topk, idx2label, True)
    save_img(model.image_dot(transformed_img.unsqueeze(0)), args.output)

def compute_loss(pred: torch.Tensor, true_label_idx: int, target_label_idx: int,
                 is_targeted: bool) -> torch.Tensor:
    # Targeted: - loss(true_label) + loss(target_label)
    # Non-targeted: - loss(true_label)
    assert true_label_idx is not None
    true_label_contrib = F.nll_loss(pred, torch.tensor([true_label_idx]).long().to(device))
    if is_targeted:
        target_label_contrib = F.nll_loss(pred, torch.tensor([target_label_idx]).long().to(device))
        return torch.mean(- true_label_contrib + target_label_contrib)  # targeted
    else:
        return torch.mean(- true_label_contrib)  # non-targeted

def torch_to_numpy(x):
    return x.to('cpu').detach().numpy().copy()

def predict_top_N(model: AttackModel, transformed_img: torch.Tensor,
                  N: int, idx2label: list, is_attacked=False) -> None:
    assert len(transformed_img.shape) == 3  # Assume the input is single [C, H, W].
    if is_attacked:
        pred = model(transformed_img.unsqueeze(0))
    else:
        pred = model.base_model(transformed_img.unsqueeze(0))
    pred = nn.Softmax(dim=1)(pred)
    pred = np.squeeze(torch_to_numpy(pred))

    indices = np.argsort(pred)[::-1]
    for rank, elem in enumerate(indices[:N]):
        print(f"({rank+1}) class: {idx2label[elem]}, idx: {elem:03d}, logit: {pred[elem]:.4f}")

def save_img(img: torch.Tensor, out_path: str, is_normalized:bool=True):
    img = torch_to_numpy(img.squeeze()).transpose(1, 2, 0)
    if is_normalized:
        for idx, (m, v) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
            img[:, :, idx] = (img[:, :, idx] * v) + m
    img = Image.fromarray((img * 255).astype(np.uint8))
    out_dir = Path(out_path).parent
    os.makedirs(out_dir, exist_ok=True)
    img.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',    type=str,
                        default='data/VGG-Face2/data/adv/test/n000001/0001_01.jpg')
    parser.add_argument('--output', '-o',    type=str,
                        default='result/n000001/0001_01-result.jpg')
    parser.add_argument('--half',    type=str,
                        default='result/n000001/0001_01-result-half/')
    parser.add_argument('--label', '-l',    type=str,
                        default='data/VGG-Face2/data/adv/test/**')
    parser.add_argument('--ckpt',   type=str,
                        default='weights/221030_0313/0181.pth')
    parser.add_argument('--topk',   type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='Initial learning rate.',
                        default=0.00001, type=float)
    parser.add_argument('--lr_decay_interval',
                        help='Learning rate will be decayed after each this interval.',
                        default=50, type=int)
    parser.add_argument('--lr_decay_rate',
                        help='Decay rate of learning rate.',
                        default=0.005, type=float)
    parser.add_argument('--epoch',
                        help='Number of training epochs.',
                        default=200, type=int)
    parser.add_argument('--target', '-t',
                        type=str,
                        help='target. if you set -1, no target.',
                        default="n009294")
    args = parser.parse_args()
    main(args)