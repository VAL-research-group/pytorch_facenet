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

### Weights & Biasesの準備
以下のリンクを参考にWeights & Biasesのアカウントを作成．
- [https://scrapbox.io/VAL/Weights & Biasesの使い方](https://scrapbox.io/VAL/Weights_&_Biases%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9)

GitHubアカウントに紐付けできるので簡単に作成できる．

アカウントが作成できたら`vggface`という名前のプロジェクトを作成．

`pip install wandb`を完了した状態で以下のコマンドを実行し，キーを貼り付けてログイン．
```
wandb login
```
WSで実行している場合はpodを作るたびにログインが必要．

### その他必要なライブラリのインストール
```
pip install -r Env/requirements.txt
```
### 学習済みモデル
[Google Drive](https://drive.google.com/drive/folders/1-RLHrneywDEXiDUi1EidlQFSpfwFX9KA?usp=sharing)

### データの配置
```
mkdir data
```
VGG-Face2を`data`の下に配置．
```
./data/VGG-Face2/data/adv/train
./data/VGG-Face2/data/adv/test
./data/VGG-Face2/meta/identity_meta.csv
# 学習済みモデルを使う場合
./weights/221030_0313/0181.pth
```
生成画像を入れておくフォルダ`result/img`を作っておく
```
.result/img/n000001-0001_01-result.jpg
```

## 学習
```
python train.py --wb_user haruka0000
```
Weights & Biasesのプロジェクト`vggface`に学習の進捗が更新される．

## 実行
```
python predict.py
```
or
```
python predict.py --ckpt ./weights/221030_0313/0181.pth -i ./data/VGG-Face2/data/adv/test/n000001/0001_01.jpg
```

## 結果
```
[input] data/VGG-Face2/data/adv/test/n000001/0001_01.jpg
Rank 1
[Predict] n000001
[Name] 14th_Dalai_Lama
[Probability] 93.1 %
--------------------------------------------------
Rank 2
[Predict] n009288
[Name] Song_Dandan
[Probability] 2.2 %
--------------------------------------------------
Rank 3
[Predict] n000958
[Name] Benigno_Noynoy_Aquino_III
[Probability] 1.0 %
--------------------------------------------------
```

## 攻撃
```
python advsticker.py
```
