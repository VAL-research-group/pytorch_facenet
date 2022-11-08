import torchvision.models as models
#from facenet.src import facenet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import FaceNet

class ImageDot(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self, device):
        super(ImageDot, self).__init__()
        self.device = device
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.alpha = 0.9
        self.radius = 25.0
        self.beta = 20.0  #2.0    #数字が大きいほどぼやけがなくなる
        self.dot_num = 4.0
        self.dn = self.dot_num+1.0
         #3*3= 9
        self.center = nn.Parameter(torch.tensor([
            [0.25, 0.25], [0.25, 0.5], [0.25, 0.75],
            [0.5, 0.25], [0.5, 0.5], [0.5, 0.75],
            [0.75, 0.25], [0.75, 0.5], [0.75, 0.75]]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            requires_grad=True)

        self.center.to(self.device)
        self.color.to(self.device)
        '''
        # 4*4 = 16
        self.center = nn.Parameter(torch.tensor([
            [1/(self.dn), 1/(self.dn)], [1/(self.dn), 2/(self.dn)], [1/(self.dn), 3/(self.dn)],[1/(self.dn), 4/(self.dn)],
            [2/(self.dn), 1/(self.dn)], [2/(self.dn), 2/(self.dn)], [2/(self.dn), 3/(self.dn)],[2/(self.dn), 4/(self.dn)],
            [3/(self.dn), 1/(self.dn)], [3/(self.dn), 2/(self.dn)], [3/(self.dn), 3/(self.dn)],[3/(self.dn), 4/(self.dn)],
            [4/(self.dn), 1/(self.dn)], [4/(self.dn), 2/(self.dn)], [4/(self.dn), 3/(self.dn)],[4/(self.dn), 4/(self.dn)]]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            requires_grad=True)
        '''

    def forward(self, x):
        _, _, height, width = x.shape
        blended = x
        for idx in range(self.center.shape[0]):
            mask = self._create_circle_mask(height, width,
                                            self.center[idx] * 255.0, self.beta).to(self.device)
            normalized_color = self._normalize_color(self.color[idx],
                                                     self.means, self.stds)
            blended = self._create_blended_img(blended, mask, normalized_color)
        return blended

    def _normalize_color(self, color, means, stds):
        return list(map(lambda x, m, s: (x - m) / s, color, means, stds))

    def _create_circle_mask(self, height, width, center, beta):
        hv, wv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        # torch.arange(0, height) -> tensor([0,1,2,...,(height-1)])  0～height-1 の　height個のリスト
        # torch.arange(0, wedth) -> tensor([0,1,2,...,(width-1)])
        # hv = tensor([[0,0,...,0],[1,1,...,1],......])
        # hw = tensor([[0,1,...,(width-1)],[0,1,...,(width-1)],...])
        hv, wv = hv.type(torch.FloatTensor), wv.type(torch.FloatTensor)
        d = ((hv - center[0]) ** 2 + (wv - center[1]) ** 2) / self.radius ** 2   #ここで形が決まる
        #print(d.shape) #[256,256]
        #print(d)
        #torch.set_printoptions(edgeitems=100)   #表示限界を決める文
        return torch.exp(- d ** beta + 1e-10)

    def _create_blended_img(self, base, mask, color):
        alpha_tile = self.alpha * mask.expand(3, mask.shape[0], mask.shape[1])
        color_tile = torch.zeros_like(base)
        for c in range(3):
            color_tile[:, c, :, :] = color[c]
        return (1. - alpha_tile) * base + alpha_tile * color_tile

class AttackModel(nn.Module):
    """
    Class to create an adversarial example.
    forward method returns the prediction result of the perturbated image.
    """
    def __init__(self, ckpt, device):
        super(AttackModel, self).__init__()
        self.image_dot = ImageDot(device)
        # load trained model
        #checkpoint = torch.load(ckpt)
        self.checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        self.classes = np.array(self.checkpoint['classes'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create an inception resnet (in eval mode):
        self.base_model = FaceNet(class_size=len(self.classes)).to(device)
        self._freeze_pretrained_model()
        
    def _freeze_pretrained_model(self):
        self.base_model.load_state_dict(self.checkpoint['model'])
        self.base_model.eval()
        for name, module in self.base_model._modules.items():
            module.requires_grad = False # 全ての層を凍結

    def forward(self, x):
        x = self.image_dot(x)
        return self.base_model(x)