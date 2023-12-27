import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from functools import partial
import math

from utils import *


class Residual(nn.Module):
    def __init__(self, fn):
        """
        Parameters
        ---
        fn: activation function
        """
        super().__init__()
        self.fn=fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        # b: batch; c: channel; h,w:height,width;
        # element: 先无视维度，将整个矩阵在h维度上拼接，则有矩阵(h*b*c,w), rearrange后的矩阵(h/p1,w/p2)
        # 则矩阵(h/p1,w/p2)上每个元素(m,n) = 原矩阵(h*b*c,w)上的元素(p1*m,p2*n)
        # 单个矩阵的元素确定后，添加维度：dimension: origin: (b,c,h,w) -> h/p1,w/p2,c*p1*p2 -> (b,c*p1*p2,h/p1,w/p2)
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2,p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim
    
    def forward(self, time:torch.Tensor):
        """
        Parameter
        ---
        time: (batch_size, 1)

        Return
        ---
        embeddings: (batch_size, dim),
        with dim being the dimensionality of the position embeddings.

        This is then added to each residual block.
        """
        device=time.device
        half_dim=self.dim // 2
        embeddings = math.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim,device=device)*-embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(),embeddings.cos()), dim=-1)
        return embeddings
    
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520

    weight standardization purportedly(据称) works synergistically(协同地) with group normalization
    """
    def forward(self,x:torch.Tensor):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight=self.weight
        # o ... -> o 1 1 1 || o ...: emit other dimensions; o 1 1 1: keepdim
        mean=reduce(weight, "o ... -> o 1 1 1", "mean")
        var=reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight-mean) / (var + eps).rsqrt()
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8) -> None:
        super().__init__()
        self.proj=WeightStandardizedConv2d(
            dim, dim_out, kernel_size=3,padding=1
        )
        self.norm=nn.GroupNorm(groups,dim_out)
        self.act=nn.SiLU()
    
    def forward(self,x,scale_shift=None):
        x=self.proj(x)
        x=self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x=x*(scale+1)+shift
        
        x=self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8) -> None:
        super().__init__()
        self.mlp=(
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out*2))
            if exists(time_emb_dim)
            else None
        )
        self.block1=Block(dim,dim_out,groups=groups)
        self.block2=Block(dim,dim_out,groups=groups)
        self.res_conv=nn.Conv2d(dim, dim_out, 1) if dim !=dim_out else nn.Identity()

    def forward(self, x, time_emb:torch.Tensor=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb=self.mlp(time_emb)
            time_emb=rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        
        h=self.block1(x,scale_shift=scale_shift)
        h=self.block2(h)
        return h+self.res_conv(x)

class Attention(nn.Module):
    "implement multi-head attention mechanism"
    def __init__(self,dim,heads=4,dim_head=32):
        super().__init__()
        self.scale=dim_head**-0.5
        self.heads=heads
        hidden_dim = dim_head*heads
        self.to_qkv=nn.Conv2d(dim, hidden_dim*3,kernel_size=1,bias=False)
        self.to_out=nn.Conv2d(hidden_dim,dim,1)

    def forward(self,x:torch.Tensor):
        b,c,h,w=x.shape
        qkv=self.to_qkv(x).chunk(3,dim=1)
        q,k,v=map(
            lambda t:rearrange(t, "b (h c) x y-> b h c (x y)",h=self.heads),
            qkv
        )
        q=q*self.scale
        sim=einsum("b h d i, b h d j->b h i j", q,k)
        # amax:Returns the maximum value of each slice of the input tensor in the given dimension(s) dim
        # prevent precision overflow
        sim=sim-sim.amax(dim=-1, keepdim=True).detach()
        attn=sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """
    linear attention variant
    https://github.com/lucidrains/linear-attention-transformer
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

