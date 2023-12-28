import aiohttp
from urllib.parse import quote
from aiofiles import open as aopen
import asyncio
from tqdm.asyncio import tqdm_asyncio
from bs4 import BeautifulSoup as BSoup
from pathlib import Path
import json

class AsyncScraper:
    charac_base_url="https://wiki.biligame.com"
    headers={
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    def __init__(self) -> None:
        self.names:list
        names_path=Path(__file__).parent.joinpath("names.json")
        with names_path.open('r',encoding='utf8') as jsn:
            self.names=json.load(jsn)
        self.connector = aiohttp.TCPConnector(force_close=True,limit=50)  # 禁用 HTTP keep-alive
        self.img_urls={k:[] for k in self.names}
        self.urls_path=Path(__file__).parent.joinpath("urls.json")
        with self.urls_path.open('r',encoding='utf8') as jsn:
            self.img_urls=json.load(jsn)
        self.img_datas={k:[] for k in self.names}
        self.errors=[]

    async def single_scrape(self,name:str):
        charac_url=self.charac_base_url+"/arknights"+f"/{name}"
        etree=None
        # async with aiohttp.ClientSession(
        #     base_url=self.charac_base_url,
        #     headers=self.headers,
        #     connector=self.connector,
        # ) as client:

        # charac_url=quote(self.charac_base_url+f"/arknights/{name}")
        async with aiohttp.request(
            "get",url=charac_url,
            headers=self.headers,connector=self.connector
        ) as resp:
            if resp.status == 200:
                data=await resp.text(encoding='utf8')
                etree=BSoup(data,'lxml')
            else:
                print("error:",name)
                self.errors.append(name)
        if etree:
            imgs=etree.select(".resp-tab-content",limit=10)
            for div_tag in imgs:
                img_tag=div_tag.select("img",limit=1)
                if any(img_tag):
                     self.img_urls[name].append(img_tag[0].get("src"))
    
    async def scrape_img_urls(self):
        if self.urls_path.exists():
            return
        for task in await tqdm_asyncio.gather(*[
            self.single_scrape(name) for name in self.names
        ],total=len(self.names),desc="scraping img_urls"):
            pass

    async def scrape_img(self,img_name,img_url):
        async with aiohttp.request(
            "get",url=img_url,headers=self.headers,
            connector=self.connector
        ) as resp:
            if resp.status==200:
                img_raw=await resp.read()
            else:
                if not img_name in self.errors:
                    print("Error:",img_name)
                    self.errors.append(img_name)
                    return 0
        if img_raw:
            async with aopen(
                Path(__file__).parent.parent.joinpath("imgs",img_name+".jpg"),
                'wb'
            ) as img:
                await img.write(img_raw)
        return 1

    async def scrape_imgs(self,loop):
        tasks=list()
        for name,urls_list in self.img_urls.items():
            for idx,url in enumerate(urls_list):
                speical_name=name+"_"+str(idx)
                # print(speical_name)
                tasks.append(self.scrape_img(speical_name,url))

        for task in tqdm_asyncio.as_completed(tasks,total=len(tasks),desc="scraping image",loop=loop):
            await task

if __name__=='__main__':
    scraper=AsyncScraper()
    loop=asyncio.get_event_loop()
    # loop.run_until_complete(scraper.scrape_img_urls())
    # with Path(__file__).parent.joinpath("urls.json").open('w',encoding='utf8') as jsn:
    #     json.dump(scraper.img_urls,jsn,ensure_ascii=False,indent=4)
    loop.run_until_complete(scraper.scrape_imgs(loop))
    loop.close()
    
