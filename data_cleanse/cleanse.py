from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

arknights_path=Path(__file__).parent.parent.joinpath("arknights_imgs")
test_img_path=Path(__file__).parent.parent.joinpath("images")
new_imgs_path=Path(__file__).parent.parent.joinpath("new_arknights")

def png2jpg(img:Path):
    image=Image.open(str(img))
    if image.format.lower()=='png':
        # PNG is RGBA
        image=image.convert("RGB")
    return image

def pngs2jpgs():
    datas={}
    total=len(os.listdir(arknights_path))
    for img in tqdm(arknights_path.glob("*.jpg"),total=total,desc="progress"):
        datas.update({img.name:png2jpg(img)})
    for name,img in tqdm(datas.items(),total=len(datas),desc="save progress"):
        path=new_imgs_path.joinpath(name)
        img.save(str(path))


if __name__ =='__main__':
    pngs2jpgs()
        
    
