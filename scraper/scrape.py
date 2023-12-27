import aiohttp
from aiofiles import open as aopen
import asyncio
from tqdm.asyncio import tqdm_asyncio
from bs4 import BeautifulSoup as BSoup
from pathlib import Path
import json

class AsyncScraper:
    charac_base_url="https://wiki.biligame.com/arknights"
    headers={
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    def __init__(self) -> None:
        self.names:list
        names_path=Path(__file__).parent.joinpath("names.json")
        with names_path.open('r',encoding='utf8') as jsn:
            self.names=json.load(jsn)
        self.img_datas=[]
        self.errors=[]

    async def single_scrape(self,name:str):
        charac_url=self.charac_base_url+"/"+name
        etree=None
        async with aiohttp.request("GET",url=charac_url,headers=self.headers,timeout=5.0) as resp:
            if resp.status == '200':
                data=await resp.text(encoding='utf8')
                etree=BSoup(data,'lxml')
            else:
                self.errors.append(name)
        if etree:
            imgs=etree.select(".resp-tab-content")
            # var x = document.querySelectorAll(".resp-tab-content");
            # x[1].querySelector("img").src; -> https://...
            # x[4].querySelector("img"); -> null