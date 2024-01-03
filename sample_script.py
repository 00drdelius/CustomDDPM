import torch
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
from utils import num_to_groups
from DDPM import Diffusion

batch_size=1
timesteps=2000
channels=3
image_size=256
device="cuda" if torch.cuda.is_available() else "cpu"
results_dir=Path(__file__).parent.joinpath("results")
if not results_dir.exists():
    results_dir.mkdir(parents=True)

torch.set_default_device(device)
torch.set_default_dtype(torch.float16)

diffusion=Diffusion(timesteps=timesteps,device=device)
model=torch.load("checkpoints/3_218.bin",map_location="cuda")

# save generated images
batches = num_to_groups(1, batch_size)
all_images_list = list(map(lambda n: diffusion.sample(model,image_size=image_size,batch_size=n, channels=channels), batches))
cnt=1
torch.save(all_images_list[0],f"temp.pt")
for all_images in all_images_list[0]:
    all_images = [(image + 1) * 0.5 for image in all_images]
    save_image(all_images, str(results_dir / f'sample-{cnt}.jpg'), nrow = 6)
    cnt+=1
