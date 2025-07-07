import json
from typing import Any

import httpx

def parse_chat_sse_response(resp: httpx.Response) -> dict[str, Any]:
    """
    Collapse an OpenAI-style SSE chat stream (already buffered in `resp.text`)
    into a single chat-completion JSON object and *also* pick up the running
    token-usage counters that some models emit in every chunk.
    """
    content_parts: list[str] = []
    tool_calls:  list[Any]  = []
    finish_reason: str | None = None
    meta: dict[str, Any] | None = None

    # will be overwritten each time we see a newer counter
    usage_stats: dict[str, int] = {}

    for raw in resp.text.splitlines():
        raw = raw.strip()
        if not raw.startswith("data:"):
            continue                         # skip keep-alive / comment lines

        payload = raw[5:].strip()
        if payload == "[DONE]":
            break

        chunk = json.loads(payload)

        # grab basic metadata only once
        if meta is None:
            meta = {k: chunk[k] for k in ("id", "object", "created", "model")}

        # keep the most recent usage snapshot
        if "usage" in chunk:
            usage_stats.update(chunk["usage"])

        # empty-choices chunks carry only usage counters – nothing to merge
        if not chunk.get("choices"):
            continue

        choice = chunk["choices"][0]
        delta  = choice.get("delta", {})

        if (part := delta.get("content")):
            content_parts.append(part)

        if delta.get("tool_calls"):
            tool_calls.extend(delta["tool_calls"])

        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

    if meta is None:
        raise ValueError("No SSE chunks found in response")

    # final assembly ----------------------------------------------------------
    content = "".join(content_parts)

    # fall back to our own count if the server never sent usage numbers
    completion_tokens = usage_stats.get("completion_tokens", len(content_parts))

    completion = {
        **meta,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": None,
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
                "stop_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens":     usage_stats.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "total_tokens":      usage_stats.get("total_tokens"),
            "prompt_tokens_details": None,
        },
        "prompt_logprobs": None,
        "kv_transfer_params": None,
    }

    return {"output": completion}
