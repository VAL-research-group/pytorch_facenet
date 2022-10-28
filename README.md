# pytorch_facenet
## 準備
### 環境
- Windows 11
- cuda 11.8
- Python 3.10
- Docker image: [haruka0000/vggface:01](https://hub.docker.com/layers/haruka0000/vggface/01/images/sha256-e3afe0af072c50d5e95be82e88e4c761753cba6be7ec87d4f66da1ea0e9643ae?context=repo)

### PyTorchのインストール
https://pytorch.org/get-started/locally/
```
torch==1.12.1+cu116
torchaudio==0.12.1+cu116
torchvision==0.13.1+cu116
```

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

## 実行
```
python predict.py
```
or
```
python predict.py -i ./data/VGG-Face2/data/train/n000012/0001_01.jpg
```

## 結果
```
[INFO]
   Class_ID           Name Sample_Num Flag Gender
11  n000012  Aaron_Ashmore        382    1      m
[Probability] 69.1 %
```
