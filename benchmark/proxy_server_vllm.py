#!/usr/bin/env python
import argparse, asyncio, sys
from itertools import cycle

import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import uvicorn

# --------------------------------------------------------------------------- #
TIMEOUT_SECONDS      = 6 * 60 * 60        # 6 h – matches vLLM default
HEALTHCHECK_TIMEOUT  = 5 * 60             # max wait for /health OK
BACKEND_PING_TIMEOUT = 10                 # per-/health ping
# --------------------------------------------------------------------------- #

async def wait_for_backends(addrs: list[str]) -> None:
    async def healthy(addr: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=BACKEND_PING_TIMEOUT) as c:
                response = await c.get(f"http://{addr}/health")
                if response.status_code == 200:
                    print(f"Back-end {addr} is online.")
                    return True
                else:
                    print(f"Back-end {addr} returned {response.status_code}.")
                    return False
        except httpx.HTTPError:
            return False

    try:
        async with asyncio.timeout(HEALTHCHECK_TIMEOUT):
            ok = await asyncio.gather(*(healthy(a) for a in addrs))
    except asyncio.TimeoutError:
        print("Timed out waiting for back-ends.", file=sys.stderr); sys.exit(1)

    if not all(ok):
        bad = [a for a, is_ok in zip(addrs, ok) if not is_ok]
        print("Back-ends not ready:", ", ".join(bad), file=sys.stderr); sys.exit(1)

def create_app(addrs: list[str]) -> FastAPI:
    backends = cycle(addrs)               # round-robin iterator
    lock     = asyncio.Lock()             # serialize next(backends)
    client   = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

    app = FastAPI()

    @app.get("/health")
    async def health():
        return Response(status_code=200, content="OK")

    @app.post("/v1/chat/completions")
    async def completions(req: Request):
        payload = await req.json()
        async with lock:
            backend = next(backends)

        url = f"http://{backend}/v1/chat/completions"

        # ---- one request, streamed ----------------------------------------
        req_obj = client.build_request("POST", url, json=payload)
        resp    = await client.send(req_obj, stream=True)

        if resp.status_code != 200:
            body = await resp.aread()
            await resp.aclose()
            raise HTTPException(resp.status_code, body.decode())

        async def body_iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()        # make sure connection is freed
        # -------------------------------------------------------------------

        return StreamingResponse(body_iter(), status_code=resp.status_code)

    return app

async def start_proxy_server(proxy_addrs: list[str], port: int = 8000) -> None:
    print(f"Starting proxy for {len(proxy_addrs)} back-ends")
    print("Waiting for back-ends to be ready...")
    await wait_for_backends(proxy_addrs)
    try:
        uvicorn.run(create_app(proxy_addrs),
                    host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("Shutting down gracefully...")

# --------------------------- CLI / entry-point ----------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser("minimal vLLM round-robin proxy (stream-safe)")
    p.add_argument("--proxy-addrs", nargs="+", required=True, help="host:port …")
    p.add_argument("--port", type=int, default=8000, help="listen port (8000)")
    args = p.parse_args()

    asyncio.run(start_proxy_server(args.proxy_addrs, args.port))
