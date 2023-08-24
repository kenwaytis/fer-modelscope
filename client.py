import time
import aiohttp
import asyncio

async def send_request(session, url, image_data, semaphore):
    async with semaphore: # 控制并发数量
        async with session.post(url, json={'image': image_data}) as response:
            return await response.json()

async def worker(url, image_data, semaphore):
    async with aiohttp.ClientSession() as session:
        for _ in range(10): # 每个worker发送10个请求
            response = await send_request(session, url, image_data, semaphore)
            print(response)

async def main():
    start_time = time.time()
    url = "http://localhost:9565/fer" # 请更改为您的实际URL
    image_data = "http://edu-evaluation.oss-cn-shanghai.aliyuncs.com/uploads/face_match/202305/22/61216fc9eb46679896a9927c0bef56ac/0180.jpg" # 请更改为您的实际图像URL或base64编码
    semaphore = asyncio.Semaphore(10) # 设置10个并发请求
    tasks = [worker(url, image_data, semaphore) for _ in range(10)] # 创建10个worker

    await asyncio.gather(*tasks)
    print(time.time()-start_time)

if __name__ == '__main__':
    asyncio.run(main())

