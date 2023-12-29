from inspect import isfunction
import torch
from torchvision.transforms import (
    Compose,ToTensor,Lambda,ToPILImage,CenterCrop,Resize
)
from torchvision.transforms import InterpolationMode
bilinear=InterpolationMode.BILINEAR
bicubic=InterpolationMode.BICUBIC
import numpy as np

Tensor=torch.Tensor

def exists(x):
    """
    判断数值是否为空

    Parameters
    ---
    x: 输入数据

    Return
    ---
    return: 如果不为空则True 反之则返回False
    """
    return x is not None

def default(val, d):
    """
    该函数的目的是提供一个简单的机制来获取给定变量的默认值。
    如果 val 存在，则返回该值。如果不存在，则使用 d 函数提供的默认值，
    或者如果 d 不是一个函数，则返回 d。

    Parameters
    ---
    val:需要判断的变量
    d:提供默认值的变量或函数
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num,divisor):
    """
    该函数的目的是将一个数字分成若干组，每组的大小都为 divisor，并返回一个列表，
    其中包含所有这些组的大小。如果 num 不能完全被 divisor 整除，则最后一组的大小将小于 divisor。

    Parameters
    ---
    num:int
    divisor:int

    Return
    ---
    arr: list
    """
    groups=num // divisor
    remainder = num % divisor
    arr:list = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def extract(a:Tensor, t:Tensor, x_shape):
    "extract the appropriate t index for a batch of indices."
    batch_size = t.shape[0]
    # Tensor.gather:
    # rearrange the indices of the element in the tensor in the specific dim, by indices t.
    # indices t: Tensor, elements in t must not exceed the max index in the dim 
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def img2tensor_module():
    """
    Example
    ---
    preprocess=img2tensor()

    img: PIL.Image, shape:(H,W,C)\n
    totensor=img2tensor_module()\n
    img=totensor(img)\n
    img:shape(C,H,W)
    """
    img_size=512
    transform=Compose([
        Resize(img_size,interpolation=bicubic),
        CenterCrop(img_size),
        ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
        Lambda(lambda t:(t*2)-1)
    ])
    # output should: output.unsqueeze(0)
    return transform

def tensor2img():
    reverse_tranform=Compose([
        Lambda(lambda t: (t+1)/2),
        Lambda(lambda t: t.permute(1,2,0)), # CHW to HWC
        Lambda(lambda t: t*255), # range[0,1]->range[0,255]
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    return reverse_tranform

