from pathlib import Path
import torch
from torchvision.utils import save_image
from torch.optim import Adam

from DDPM import Unet,Diffusion
from utils import num_to_groups
from dataset import createLoader

timesteps=300
image_size=None
channels=3
epochs=0

save_and_sample_every=1000
results_folder=Path("results").__str__()

arknightsDataLoader=createLoader("images",False)
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        batch_size=len(batch)
        batch = [i for i in batch]

        #sample t uniformally for every example in the batch
        t = torch.randint(0,timesteps,(batch_size,),device=device).long()
        diffusion=Diffusion(timesteps)
        loss=diffusion.p_losses(model,batch,t,loss_type="l2")

        if step % 100==0:
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

