from pyexpat import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from sticker import ImageDot, AttackModel
import glob
import os
from PIL import Image
from skimage import io
import copy
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

composed = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def main(args):
    img_array = io.imread(args.input)
    transformed_img = composed(Image.fromarray(img_array))

    model = AttackModel(args.ckpt, device)
    
    class_names = classname(args.label)

    pred = model.base_model(transformed_img.unsqueeze(0).to(device))
    pred = np.squeeze(torch_to_numpy(pred))
    for elem in np.argsort(pred)[:-10:-1]:
        print(f"class: {class_names[elem]}, idx: {elem}, logit: {pred[elem]:.4f}")
    
    truelabel = int(np.argsort(pred)[-1])

    #ドットを置いた状態の画像
    dotted_img = model.image_dot(transformed_img.unsqueeze(0).to(device))
    img_half = plot_img_from_normalized_img(torch_to_numpy(dotted_img.squeeze()))
    img_half = Image.fromarray((img_half*255).astype(np.uint8))
    img_half.save(args.half)

    lr = 0.008
    losslist = []

    for epoch in range(10):  #200
        if (epoch + 1) % 25 == 0:  #50
            lr *= 0.5
        model.zero_grad()
        pred = model(transformed_img.unsqueeze(0).to(device))
        #loss = compute_loss(pred, truelabel, -1)  # Non-targeted
        loss = compute_loss(pred, truelabel , 2)  # Targeted
        loss.backward(retain_graph=True)

        print(f"epoch: {epoch + 1}, loss: {loss.data:.4f}")
        losslist.append(loss.data)

        for param in model.parameters():
            if param.requires_grad == True:
                param.data = torch.clamp(param.data - param.grad.data * lr, min=0.0001, max=1.0)
                #print(f"param.data : {param.data}")

    #ドットの位置、色を最適化した画像
    doted_img = model.image_dot(transformed_img.unsqueeze(0).to(device))
    img_result = plot_img_from_normalized_img(torch_to_numpy(doted_img.squeeze()))
    img_result = Image.fromarray((img_result*255).astype(np.uint8))
    img_result.save(args.output)
    
    pred = model(transformed_img.unsqueeze(0).to(device))
    pred = np.squeeze(torch_to_numpy(pred))
    for elem in np.argsort(pred)[:-10:-1]:
        print(f"class: {class_names[elem]}, idx: {elem}, logit: {pred[elem]:.4f}")

def plot_img_from_normalized_img(img_array, is_normalized=True):
    img_to_be_plotted = copy.deepcopy(img_array)
    assert len(img_array.shape) == 3
    if img_to_be_plotted.shape[0] == 3:
        img_to_be_plotted = img_to_be_plotted.transpose(1, 2, 0)
    if is_normalized:
        for idx, (m, v) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
            img_to_be_plotted[:, :, idx] = (img_to_be_plotted[:, :, idx] * v) + m
    #plt.figure()
    #plt.imshow(img_to_be_plotted)
    #plt.show()
    return img_to_be_plotted

def compute_loss(pred: torch.Tensor, true_label_idx: int, target_label_idx: int) -> torch.Tensor:
    # Targeted: - loss(true_label) + loss(target_label)
    # Non-targeted: - loss(true_label)
    assert true_label_idx is not None
    true_label_contrib = F.nll_loss(pred, torch.tensor([true_label_idx]).to(device))
    if target_label_idx == -1:
        return torch.mean(- true_label_contrib)  # non-targeted
    else:
        target_label_contrib = F.nll_loss(pred, torch.tensor([target_label_idx]).to(device))
        return torch.mean(- true_label_contrib + target_label_contrib)  # targeted

def classname(label):
    class_names = glob.glob(label)
    namelist = []
    for i in range(len(class_names)):
        basename = os.path.basename(class_names[i])
        namelist.append(basename)

    return namelist

def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img 

def torch_to_numpy(x):
    return x.to('cpu').detach().numpy().copy()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',    type=str,
                        default='data/VGG-Face2/data/adv/test/n000001/0001_01.jpg')
    parser.add_argument('--output', '-o',    type=str,
                        default='result/img/n000001-0001_01-result.jpg')
    parser.add_argument('--half',    type=str,
                        default='result/img/n000001-0001_01-half.jpg')                    
    parser.add_argument('--label', '-l',    type=str,
                        default='data/VGG-Face2/data/adv/test/**')    
    parser.add_argument("--wb_user", type=str, default="",
                        help="User name of Weights&Biases")           
    parser.add_argument('--ckpt',   type=str,
                        default='weights/221030_0313/0181.pth')     
    args = parser.parse_args()
    main(args)