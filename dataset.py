import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from typing import Any,Dict, Iterable, List, Optional,Union
from rich import print
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle as pkl
from utils import img2tensor_module

class ArknightsDataset(Dataset):
    def __init__(
            self,
            data_dir:Union[Path,str],
            refresh:bool=False
    ) -> None:
        super().__init__()
        if isinstance(data_dir,str):
            data_dir=Path(data_dir)
            assert data_dir.exists(), "data_dir not exists in the path you gave"
        cached_path=data_dir.joinpath("cached.pkl")
        if refresh or not cached_path.exists():
            self.datas=[]
            for img_path in data_dir.glob("*.jpg"):
                character=img_path.name.split(".")[0]
                img=Image.open(str(img_path))
                totensor=img2tensor_module()
                self.datas.append({
                    "character":character,
                    "image":totensor(img)
                })
            with cached_path.open('wb') as p:
                pkl.dump(self.datas,p)
        else:
            with cached_path.open('rb') as p:
                self.datas=pkl.load(p)
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index) -> Dict:
        return self.datas[index]

def createLoader(data_dir:Union[Path,str],refresh:bool):
    arknightsDataset=ArknightsDataset(data_dir=data_dir,refresh=refresh)
    print("length of dataset:",len(arknightsDataset))
    ArknightsDataLoader=DataLoader(
        dataset=arknightsDataset,
        batch_size=2,
        shuffle=True
    )
    return ArknightsDataLoader

if __name__ == '__main__':
    arknightsDataLoader=createLoader(data_dir="images",refresh=False)
    for i in arknightsDataLoader:
        print(i)

