import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
import torch

def linear_beta_schedule(timesteps):
    beta_start=1e-4
    beta_end=2e-2
    return torch.linspace(beta_start,beta_end,timesteps)

def cosine_beta_schedule(timesteps,s=0.008):
    """
    cosine schedule
    https://arxiv.org/abs/2102.09672
    """
    steps=timesteps+1
    x=torch.linspace(0,timesteps,steps)
    #cumprod:累积
    alphas_cumprod=torch.cos(
        ((x/timesteps)+s)/(1+s)*torch.pi*0.5
    )**2
    alphas_cumprod=alphas_cumprod / alphas_cumprod[0]
    betas=1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)

