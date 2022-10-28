import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceNet(nn.Module):
    def __init__(self, class_size:int, pretrained:str='vggface2'):
        super(FaceNet, self).__init__()
        self.model = InceptionResnetV1(classify=True, pretrained=pretrained)
        for name, module in self.model._modules.items():
            module.requires_grad = False # 全ての層を凍結
        self.model.last_linear = nn.Linear(1792, 512, bias=False)
        self.model.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        self.model.logits = nn.Linear(512, class_size) # 今回はnクラスに分類

    def forward(self, x):
        output = self.model(x)
        return output