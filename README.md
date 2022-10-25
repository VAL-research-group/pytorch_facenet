# pytorch_facenet
## 準備
### PyTorchのインストール
https://pytorch.org/get-started/locally/

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
python predict.py -i ./data/VGG-Face2/data/test/n000001/0001_01.jpg
```

## 結果
```
[INFO]
     Class_ID          Name Sample_Num Flag Gender
2990  n003027  George_Takei        476    1      m
[Probability] 22.7 %
```