from torch import nn
from einops import rearrange
from rich.console import Console
import torch
from xformers.ops import memory_efficient_attention as xf_attn

console=Console(style="#ffef4d")
print=console.print
torch.set_default_device("cuda:0")

class Attention(nn.Module):
    """
    linear attention variant\n
    https://github.com/lucidrains/linear-attention-transformer\n
    """
    def __init__(self, dim=512, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x, xformer=False):
        print("input shape:"+str(x.shape))
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        if not xformer:
            print("attention normal mode")
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)

            q = q * self.scale
            context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

            out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
            out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        else:
            print("attention xformers mode")
            out = xf_attn(q,k,v,scale=self.scale)
        print("output shape:"+str(out.shape))
        print("memory allocated:"+str(torch.cuda.memory_allocated("cuda:0")//1024**3)+"GB")
        return self.to_out(out)

ten=torch.randn(2,512,512,512)
#q,k,v .shape: (2,4,32,512*512)
#out.shape: (2,128,512,512)
attention=Attention()
attention(ten,xformer=False)
