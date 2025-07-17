from operator import mul
from benchmark_cornserve import cornserve_invoke
import asyncio
from tqdm import tqdm
from benchmark_backend import RequestInput, RequestOutput

async def main():
    pbar = tqdm(total=10)

    request_inputs =[] 
    for i in range(10):
        request_inputs.append(
            RequestInput(
                url="http://localhost:30080/app/invoke/app-af828b42a72442b59127a82657ad1bfd",
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                prompt="What is the color of the sky?",
                prompt_len=10,
                output_len=20,
                multi_modal_data=[{
                    "type": "image_url",
                    "image_url": {"url": "https://picsum.photos/id/237/200/300"},
                }],
            )
        )
        
    coros = [cornserve_invoke(request_input, pbar) for request_input in request_inputs]
    await asyncio.gather(*coros)
    for output in coros:
        print(output)


if __name__ == "__main__":
    asyncio.run(main())
