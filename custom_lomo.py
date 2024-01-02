from torch.optim import Optimizer
import torch
from DDPM import Unet


class CustomLOMO(Optimizer):
    def __init__(
            self,
            model:Unet,
            lr=1e-3,
            clip_grad_norm=None
    ) -> None:
        self.model=model
        self.lr=lr
        for _,p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.backword_hook)
        defaults=dict(lr=lr,clip_grad_norm=clip_grad_norm,)
        super(CustomLOMO,self).__init__(self.model.parameters(), defaults)
    
    def backword_hook(self,grad):
        "param:grad is necessary placeholder for registering hook function"
        with torch.no_grad():
            for name,p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    print("grad not None parameters:",name)
                    if (
                        torch.isnan(p.grad).any().item() or 
                        torch.isinf(p.grad).any().item()
                    ):
                        print(f"Detecting overflow in layer:{name}. Auto breaking..")
                        break
                    #SGD
                    grad_fp32=p.grad.to(torch.float32)
                    p.grad=None
                    p_fp32=p.data.to(torch.float32)
                    p_fp32.add_(grad_fp32,alpha=-self.lr)
                    p.data.copy_(p_fp32)
                else:
                    # print("grad None parameters:",name)
                    pass
