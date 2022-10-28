# pytorch_facenet
## 準備
### 環境
- Windows 11
- cuda 11.8
- Python 3.10
- Docker image: [haruka0000/vggface:1.0](https://hub.docker.com/layers/haruka0000/vggface/1.0/images/sha256-c7c0207b62c812df96b3c3202ba9e7ace167c4f76be6257c52c16f9d40dd85cf?context=repo)

### PyTorchのインストール
https://pytorch.org/get-started/locally/
```
torch==1.12.1+cu116
torchaudio==0.12.1+cu116
torchvision==0.13.1+cu116
```

### Weights & Biases
[https://scrapbox.io/VAL/Weights & Biasesの使い方](https://scrapbox.io/VAL/Weights_&_Biases%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9)

### その他必要なライブラリのインストール
```
pip install -r Env/requirements.txt
```

### データの配置
```
mkdir data
```
VGG-Face2を`data`の下に配置．
```
./data/VGG-Face2/data/train/n000012/0001_01.jpg
./data/VGG-Face2/data/train_list.txt
data/VGG-Face2/meta/identity_meta.csv
```

## Train
```
python train.py --wb_user haruka0000
```

## 実行
```
python predict.py
```
or
```
python predict.py -i ./data/VGG-Face2/data/adv/test/n000001/0001_01.jpg
```

## 結果
```
[input] data/VGG-Face2/data/adv/test/n000001/0001_01.jpg
Rank 1
[Predict] n000001
[Name] 14th_Dalai_Lama
[Probability] 39.6 %
--------------------------------------------------
Rank 2
[Predict] n001830
[Name] Cristóbal_Montoro
[Probability] 18.0 %
--------------------------------------------------
Rank 3
[Predict] n002329
[Name] Dési_Bouterse
[Probability] 4.8 %
--------------------------------------------------
```
