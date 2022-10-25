# pytorch_facenet
## 準備
### 環境
- Windows 11
- cuda 11.8
- Python 3.10

### PyTorchのインストール
https://pytorch.org/get-started/locally/
```
torch==1.12.1+cu116
torchaudio==0.12.1+cu116
torchvision==0.13.1+cu116
```

### その他必要なライブラリのインストール
```
pip install -r requirements.txt
```

### データの配置
```
./data/VGG-Face2/data/
./data/VGG-Face2/meta/
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