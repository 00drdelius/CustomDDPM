import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
import torch
from torchvision.utils import save_image
from torch.optim import Adam
from rich.console import Console

from DDPM import Unet,Diffusion
from utils import num_to_groups
from dataset import createLoader

console=Console(style="#fff385")
print=console.print

timesteps=300
image_size=512
channels=3
epochs=2

save_and_sample_every=1000
results_folder=Path("results").__str__()

arknightsDataLoader=createLoader("images",False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:",device)
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1,2,4,)
)
model.to(device)
optimizer = Adam(model.parameters(),lr=1e-3)

for epoch in range(epochs):
    for step,batch in enumerate(arknightsDataLoader):
        optimizer.zero_grad()
        if isinstance(batch,list):
            batch_size=len(batch)
            batch = [i['image'] for i in batch]
        else:
            batch_size=1
            batch=[batch['image']]
        batch=torch.cat(batch,dim=0).to(device)
        print("batch size:",batch.shape)
        #sample t uniformally for every example in the batch
        t = torch.randint(0,timesteps,(batch_size,),device=device).long()
        diffusion=Diffusion(timesteps,device)
        loss=diffusion.p_losses(model,batch,t,loss_type="l2")

        if step % 2==0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        # save generated images
        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: diffusion.sample(model, batch_size=n, channels=channels), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

