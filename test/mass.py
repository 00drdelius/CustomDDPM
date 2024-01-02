import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from torchvision.transforms import (
    Resize,CenterCrop,ToTensor,ToPILImage,InterpolationMode,Compose,Lambda
)
from PIL import Image
import torch
from einops.layers.torch import Rearrange
from torch import nn

in_channels=3
out_channels=256

def img2tensor_module():
    size=256
    return Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Lambda(lambda t:(t*2)-1)
    ])

conv=nn.Conv2d(
    in_channels,out_channels,
    kernel_size=3,stride=2,padding=1
)
maxPool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
norm=nn.GroupNorm(num_groups=3,num_channels=3,eps=1e-4)

img=Image.open("images/ling.jpg")
toTensor=img2tensor_module()
ten=toTensor(img)
print(ten.shape)

def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        # b: batch; c: channel; h,w:height,width;
        # element: 先无视维度，将整个矩阵在h维度上拼接，则有矩阵(h*b*c,w), rearrange后的矩阵(h/p1,w/p2)
        # 则矩阵(h/p1,w/p2)上每个元素(m,n) = 原矩阵(h*b*c,w)上的元素(p1*m,p2*n)
        # 单个矩阵的元素确定后，添加维度：dimension: origin: (b,c,h,w) -> h/p1,w/p2,c*p1*p2 -> (b,c*p1*p2,h/p1,w/p2)
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2,p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )

in_out=[(512,512),(512,1024),(1024,2048)]