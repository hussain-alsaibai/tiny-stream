"""Usage examples for tiny-stream.

Run individual sections by uncommenting the relevant call at the bottom.

The examples use a tiny fake response so they run offline; for real usage,
swap the ``FakeResponse`` instances for real OpenAI / Anthropic / requests /
httpx / aiohttp responses.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Iterator, List

from tiny_stream import LLMStream, StreamFormat, StreamParser


# ---------------------------------------------------------------------------
# Fake stream generators (replace with real HTTP responses in production)
# ---------------------------------------------------------------------------


def fake_openai_response() -> "FakeResponse":
    """Simulate an OpenAI chat.completion.chunk stream."""

    chunks = [
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":", "}}]}\n\n',
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"world"}}]}\n\n',
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"!"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    return FakeResponse(chunks)


def fake_anthropic_response() -> "FakeResponse":
    """Simulate an Anthropic messages stream."""

    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{}}\n\n',
        b'event: content_block_start\ndata: {"type":"content_block_start"}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"Hi"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":" there"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"!"}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    return FakeResponse(chunks)


def fake_local_model_response() -> Iterator[bytes]:
    """Simulate a local LLM server emitting JSON-lines."""

    return iter(
        [
            b'{"text":"Once "}\n',
            b'{"text":"upon "}\n',
            b'{"text":"a "}\n',
            b'{"text":"time"}\n',
        ]
    )


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, chunks: List[bytes], content_type: str = "text/event-stream") -> None:
        self._chunks = chunks
        self.headers = {"content-type": content_type}

    def iter_content(self, chunk_size=None):  # noqa: ARG002
        for c in self._chunks:
            yield c

    def iter_lines(self, decode_unicode=True):  # noqa: ARG002
        joined = b"".join(self._chunks).decode("utf-8")
        for line in joined.split("\n"):
            yield line


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def example_openai_streaming() -> None:
    """Stream tokens from an OpenAI-compatible response."""

    print("=== OpenAI streaming ===")
    response = fake_openai_response()
    stream = LLMStream(response)
    for token in stream:
        print(token, end="", flush=True)
    print("\n")


def example_anthropic_streaming() -> None:
    """Stream tokens from an Anthropic response."""

    print("=== Anthropic streaming ===")
    response = fake_anthropic_response()
    stream = LLMStream(response)
    for token in stream:
        print(token, end="", flush=True)
    print("\n")


def example_local_model_jsonl() -> None:
    """Stream tokens from a local model emitting JSON-lines."""

    print("=== Local model (JSON-lines) ===")
    stream = LLMStream(
        fake_local_model_response(),
        format=StreamFormat.JSONL,
    )
    for token in stream:
        print(token, end="", flush=True)
    print("\n")


def example_low_level_sse() -> None:
    """Use ``StreamParser`` directly for full control."""

    print("=== Low-level SSE ===")
    parser = StreamParser()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"low"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" level"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    for token in parser.parse_sse(chunks):
        # Inspect the raw payload instead of just extracting tokens.
        print(f"event={token.event!r} data={json.dumps(token.data)}")
    print()


def example_extract_tokens_disabled() -> None:
    """Yield raw ``ParsedToken`` objects instead of plain strings."""

    print("=== Raw ParsedToken objects ===")
    response = fake_openai_response()
    stream = LLMStream(response, extract_tokens=False)
    for token in stream:
        print(f"index={token.index} data={token.data}")
    print()


async def example_async_streaming() -> None:
    """Async iteration over a streaming response."""

    print("=== Async streaming ===")

    async def async_openai() -> AsyncIterator[bytes]:
        response = fake_openai_response()
        for chunk in response._chunks:
            yield chunk

    stream = LLMStream(async_openai())
    async for token in stream.aiter_tokens():
        print(token, end="", flush=True)
    print("\n")


def example_gzip_decompression() -> None:
    """Transparent gzip decompression."""

    import gzip
    import io

    print("=== Gzip decompression ===")
    body = b"".join(
        [
            b'data: {"choices":[{"delta":{"content":"compressed"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
    )
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(body)
    response = FakeResponse(
        [buf.getvalue()],
        content_type="text/event-stream",
    )
    response.headers["content-encoding"] = "gzip"
    stream = LLMStream(response)
    for token in stream:
        print(token, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    example_openai_streaming()
    example_anthropic_streaming()
    example_local_model_jsonl()
    example_low_level_sse()
    example_extract_tokens_disabled()
    asyncio.run(example_async_streaming())
    example_gzip_decompression()


if __name__ == "__main__":
    main()