import csv
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import *
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
from model import FaceNet


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


# def load_train_list(filename):
#     filename = Path(filename)
#     classes = set([])
#     with open(filename, 'r', encoding="utf-8") as f:
#         train_list = f.readlines()
#     for row in tqdm(train_list):
#         classes.add(row.split('/')[0])
#     classes = sorted(list(classes))
#     return classes, train_list


def main(args):
    img = Image.open(args.input).convert('RGB')

    # load trained model
    checkpoint = torch.load(args.ckpt)
    classes = np.array(checkpoint['classes'])

    # Create an inception resnet (in eval mode):
    model = FaceNet(class_size=len(classes)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 全人物データを読み込み
    meta_df = load_meta(args.meta) # 9131クラス

    topk = args.topk

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=64,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    img = mtcnn(img, save_path=None) # 顔部分をクロップ

    # prediction
    img_probs = model(img.unsqueeze(0).to(device)).squeeze() # 500クラスに対する確率
    img_probs = nn.Softmax(dim=0)(img_probs) # 0.0~1.0
    img_probs = torch_to_numpy(img_probs)

    indices = np.argsort(img_probs)[::-1] # sort
    probs = img_probs[indices[0:topk]]
    person_keys = classes[indices[0:topk]]

    print("[input] %s"%(args.input))
    for i, (prob, key) in enumerate(zip(probs, person_keys)):
        print('Rank %d'%(i+1))
        # metaデータからkeyで検索して人物名等の情報を取得
        meta = meta_df.loc[meta_df['Class_ID']==key, :]
        print("[Predict] %s"%(key)) 
        print("[Name] %s"%(meta['Name'].values[0]))
        print("[Probability] %0.1f %%"%(prob*100))
        print('-'*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',    type=str,
                        default='data/VGG-Face2/data/adv/test/n000001/0001_01.jpg')
    parser.add_argument('--meta', '-m',     type=str,
                        default='data/VGG-Face2/meta/identity_meta.csv')
    parser.add_argument('--ckpt',   type=str,
                        default='weights/221029_0135/0003.pth')
    parser.add_argument('--topk',   type=int,
                        default=3)
    args = parser.parse_args()
    main(args)