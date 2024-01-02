import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from torchvision.transforms import (
    Resize,CenterCrop,ToTensor,ToPILImage,InterpolationMode,Compose,Lambda
)
from PIL import Image
import torch
from torch import nn

in_channels=3
out_channels=256

def img2tensor_module():
    size=512
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
