import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
import torch
from torchvision.utils import save_image
from torch.optim import Adam
from rich.console import Console
import json

from DDPM import Unet,Diffusion
from custom_lomo import CustomLOMO
from utils import num_to_groups
from dataset import createLoader

console=Console(style="#fff385")
print=console.print

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:",device)
torch.set_default_device(device)
torch.set_default_dtype(torch.float16)
timesteps=300
image_size=256
channels=3
epochs=2
learning_rate=1e-4

save_and_sample_every=1000
results_folder=Path("results").__str__()

arknightsDataLoader=createLoader("images",refresh=False)

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1,2,4,)
)
if not Path(__file__).parent.joinpath("params.json").exists():
    #records the parameters name of model
    params=[name for name,p in model.named_parameters() if p.requires_grad]
    with open("params.json",'w',encoding='utf8') as jsn:
        json.dump(params,jsn,ensure_ascii=False,indent=4)
# optimizer = Adam(model.parameters(),lr=1e-3)
optimizer = CustomLOMO(model,lr=learning_rate)

for epoch in range(epochs):
    for step,batch in enumerate(arknightsDataLoader):
        optimizer.zero_grad()
        if isinstance(batch,list):
            batch_size=len(batch)
            batch = [i['image'] for i in batch]
        else:
            batch_size=1
            batch=[batch['image']]
        batch=torch.cat(batch,dim=0).cuda().to(torch.float16)
        print("batch size:",batch.shape)
        #sample t uniformally for every example in the batch
        t = torch.randint(0,timesteps,(batch_size,),device=device).long()
        diffusion=Diffusion(timesteps,device)
        loss=diffusion.p_losses(model,batch,t,loss_type="l2")

        if step:
            print("Loss:", loss.item())

        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        # 实测最后一层bias需要第二次才更新
        optimizer.backword_hook(0)

        # optimizer.step() already update parameters in hook function by SGD

        # save generated images
        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: diffusion.sample(model, batch_size=n, channels=channels), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

