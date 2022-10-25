
import os
import csv
import torch
import torch.nn as nn
from torchvision.transforms import *
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def torch_to_numpy(x):
    return x.to('cpu').detach().numpy().copy()


def load_meta(filename):
    filename = Path(filename)
    with open(filename, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"', skipinitialspace=True)
        data = [row for row in reader]
    header = data[0]
    data = data[1:]
    print(len(data))
    df = pd.DataFrame(data, columns=header)
    return df

def load_train_list(filename):
    filename = Path(filename)
    classes = set([])
    with open(filename, 'r', encoding="utf-8") as f:
        train_list = f.readlines()
    for row in tqdm(train_list):
        classes.add(row.split('/')[0])
    classes = sorted(list(classes))
    return classes, train_list


# https://github.com/timesler/facenet-pytorch
def predict(img:Image):
    img = img.convert('RGB')

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=64,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').to(device)
    resnet.eval()

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=None)

    # VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0).to(device))
    img_probs = nn.Softmax(dim=0)(img_probs.squeeze())
    return torch_to_numpy(img_probs)


def main(args):
    train_classes, train_list = load_train_list(args.train)
    meta_df = load_meta(args.meta)

    img = Image.open(args.input)
    img_probs = predict(img)
    idx = img_probs.argmax()
    prob = img_probs[idx]
    person_key = train_classes[idx]

    meta = meta_df.loc[meta_df['Class_ID']==person_key, :]
    print(meta, prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',    type=str, default='./data/VGG-Face2/data/test/n000001/0001_01.jpg')
    parser.add_argument('--meta', '-m',     type=str, default='data/VGG-Face2/meta/identity_meta.csv')
    parser.add_argument('--train',          type=str, default='data/VGG-Face2/data/train_list.txt')
    args = parser.parse_args()
    main(args)