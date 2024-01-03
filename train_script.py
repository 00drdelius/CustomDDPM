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
from tqdm import tqdm

console=Console(style="#fff385")
print=console.print

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:",device)
torch.set_default_device(device)
torch.set_default_dtype(torch.float16)
data_dir="arknights_imgs"
timesteps=400
image_size=256
channels=3
epochs=4
learning_rate=1e-3
batch_size=4

save_and_sample_every=100
save_loss_lt=0.2
cp_dir=Path(__file__).parent.joinpath("checkpoints")
if not cp_dir.exists():
    cp_dir.mkdir(parents=True)
max_cps=4
results_folder=Path(__file__).joinpath("results").__str__()


arknightsDataLoader=createLoader(data_dir,batch_size=batch_size,image_size=image_size,refresh=True)

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
    for step,batch in tqdm(
        enumerate(arknightsDataLoader),desc=f"epoch_{epoch}",total=len(arknightsDataLoader)
    ):
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
        if epoch==epochs-1:
            if loss.item()<=save_loss_lt:
                if len(list(cp_dir.iterdir()))<max_cps:
                    pass
                else:
                    first=list(cp_dir.iterdir())[-1]
                    first.unlink(missing_ok=True)
                # torch.save(model.state_dict(),str(cp_dir.joinpath(f"{epoch}_{step}.bin")))
                torch.save(model,str(cp_dir.joinpath(f"{epoch}_{step}.bin")))


