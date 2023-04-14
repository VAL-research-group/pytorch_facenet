import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceNet(nn.Module):
    def __init__(self, class_size:int, pretrained:str='vggface2'):
        super(FaceNet, self).__init__()
        # ファインチューニング：訓練済みの公開モデルを使って再学習
        self.model = InceptionResnetV1(classify=True, pretrained=pretrained)
        for name, module in self.model._modules.items():
            # 全ての層を凍結(ある程度学習できている部分は変更させない)
            module.requires_grad = False

        # 以下の部分だけ学習される
        self.model.last_linear = nn.Linear(1792, 512, bias=False)
        self.model.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        self.model.logits = nn.Linear(512, class_size) # 今回はnクラスに分類

    def forward(self, x):
        output = self.model(x)
        return output