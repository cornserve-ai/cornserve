from benchmark_cornserve import cornserve_invoke
import asyncio
from tqdm import tqdm
from benchmark_backend import RequestInput

async def main():
    n = 10
    pbar = tqdm(total=10)

    request_inputs =[] 
    for _ in range(10):
        request_inputs.append(
            RequestInput(
                url="http://localhost:30080/app/invoke/app-3c148691df554d1c90e48ff9aa45e338",
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
        
    coros = [asyncio.create_task(cornserve_invoke(request_input, pbar)) for request_input in request_inputs]
    await asyncio.gather(*coros)
    for output in coros:
        print(output.result())


if __name__ == "__main__":
    asyncio.run(main())
