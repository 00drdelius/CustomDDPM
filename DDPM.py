import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from functools import partial

# %matplotlib inline
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from utils import extract,default
from unet_components import *
from dif_components import *


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        # dim_mults increases the output_dim,
        # for instance, set init_dim=128, dim=256, then dims=[256, 256, 512, 1024, 2048]
        # then in_out = [(128, 256), (256, 512), (512, 1024), (1024, 2048)]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class Diffusion:
    def __init__(self,timesteps,device) -> None:
        self.timesteps=timesteps
        self.device=device
        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
        # F.pad, accept a tensor for padded and pad_size=(1,0) and pad_value=1.0,
        # pad_size=(1,0) will pads the last dimension of the input tensor.
        # for instance, input:(1,2,2,5), pad_size=(1,0), pad_value=1.0; output:(1,2,2,6) with value all are 1.0.
        self.alphas_cumprod_prev=F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0)
        self.sqrt_recip_alphas=torch.sqrt(1.0/self.alphas)

        # calculations for diffusion q(x_t|x_{t-1}) and others
        # forward diffusion
        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance=self.betas*(1.-self.alphas_cumprod_prev)/(1.-self.alphas_cumprod)

    @torch.no_grad()
    def q_sample(self,x_start,t,noise=None):
        """
        forward diffusion (using the nice property)\n
        Parameters
        ---
        x_start:tensor, x_0\n
        t:tensor,timestep tensor\n
        noise: None is ok\n
        Return
        ---
        q(x_t|x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start,device=self.device) # random sample from StandNormalDist
        sqrt_alphas_cumprod_t=extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape
        ).to(self.device)
        sqrt_one_minus_alphas_cumprod_t=extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape
        ).to(self.device)
        return (
            sqrt_alphas_cumprod_t*x_start +  
            sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(
        self,
        denoise_model,
        x_start,
        t,
        noise=None,
        loss_type="l2"
    ):
        """
        sample,forward,predict and calculate losses
        """
        if not noise:
            noise = torch.randn_like(x_start,device=self.device)
        
        x_noisy=self.q_sample(x_start,t,noise)
        predict_noise = denoise_model(x_noisy,t)

        if loss_type=='l1':
            loss=F.l1_loss(noise,predict_noise)
        elif loss_type=='l2':
            loss=F.mse_loss(noise,predict_noise)
        elif loss_type=='huber':
            loss=F.smooth_l1_loss(noise,predict_noise)
        else:
            raise NotImplementedError

        return loss

    @torch.no_grad()
    def p_sample(self,model,x,t,t_index):
        "reverse process,denoising,single image"
        betas_t=extract(self.betas,t,x.shape)
        sqrt_one_minus_alphas_cumprod_t=extract(
            self.sqrt_one_minus_alphas_cumprod,t,x.shape
        )
        sqrt_recip_alphas_t=extract(
            self.sqrt_recip_alphas, t, x.shape
        )
        # Use our model (noise predictor) to predict the mean
        model_mean=sqrt_recip_alphas_t * (
            x-betas_t*model(x,t)/sqrt_one_minus_alphas_cumprod_t
        )
        if t_index==0:
            return model_mean
        else:
            posterior_variance_t=extract(
                self.posterior_variance,t,x.shape
            )
            noise=torch.randn_like(x)
            return model_mean+torch.sqrt(posterior_variance_t)*noise

    @torch.no_grad()
    def p_sample_loop(self,model,shape):
        device=next(model.parameters()).device
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img=torch.randn(shape,device=device)
        imgs=[]
        for i in tqdm(
            reversed(range(0,self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps
        ):
            img=self.p_sample(
                model,
                img,
                torch.full((b,),i,device=device,dtype=torch.long),
                i
            )
            imgs.append(img.cpu().numpy())
        return imgs

    def sample(
        self,
        model,
        image_size,
        batch_size=16,
        channels=3
    ):
        return self.p_sample_loop(
            model,
            shape=(batch_size,channels,image_size,image_size)
        )

    def get_noisy_image(self,x_start,t):
        "test q_sample function"
        # interrupt image by noise
        x_noisy = self.q_sample(x_start,t)
        #turn back into PIL image
        noise_image = tensor2img(x_noisy.squeeze())

        return noise_image




