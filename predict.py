
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
    img_cropped = mtcnn(img, save_path=None) # 顔部分をクロップ

    # VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0).to(device)).squeeze() # 8631クラスに対する確率
    img_probs = nn.Softmax(dim=0)(img_probs) # 0.0~1.0
    return torch_to_numpy(img_probs)


def main(args):
    # 訓練に使用した人物リストを読み込み
    train_classes, train_list = load_train_list(args.train) # 8631クラス

    # 全人物データを読み込み
    meta_df = load_meta(args.meta) # 9131クラス

    img = Image.open(args.input)
    img_probs = predict(img) # 8631クラスそれぞれの確率
    idx = img_probs.argmax() # 最も確率が高い確率のindex
    prob = img_probs[idx] # 最も確率が高い確率
    # 訓練に使用したクラス数と全体のクラス数が異なるので，訓練の方のからidxを参照する必要がある
    person_key = train_classes[idx]

    # metaデータからkeyで検索して人物名等の情報を取得
    meta = meta_df.loc[meta_df['Class_ID']==person_key, :]
    print("[INFO]")
    print(meta)
    print("[Probability] %0.1f %%"%(prob*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',    type=str, default='./data/VGG-Face2/data/test/n000001/0001_01.jpg')
    parser.add_argument('--meta', '-m',     type=str, default='data/VGG-Face2/meta/identity_meta.csv')
    parser.add_argument('--train',          type=str, default='data/VGG-Face2/data/train_list.txt')
    args = parser.parse_args()
    main(args)