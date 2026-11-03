"""Tests for tiny-stream.

Run with:
    python -m pytest test_tiny_stream.py -v
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import zlib
from typing import AsyncIterator, Iterator, List

import pytest

from tiny_stream import (
    LLMStream,
    ParsedToken,
    SSEEvent,
    StreamFormat,
    StreamParser,
    __version__,
)


# ---------------------------------------------------------------------------
# Tiny stand-ins for popular HTTP clients.
# ---------------------------------------------------------------------------


class FakeResponse:
    """Mimic enough of ``requests.Response`` for LLMStream."""

    def __init__(
        self,
        chunks: List[bytes],
        content_type: str = "text/event-stream",
        content_encoding: str = "",
        iter_lines_decoded: bool = True,
    ) -> None:
        self._chunks = chunks
        self.headers = {
            "content-type": content_type,
            "content-encoding": content_encoding,
        }
        self._iter_lines_decoded = iter_lines_decoded

    def iter_content(self, chunk_size=None):  # noqa: ARG002 - requests-compatible
        for c in self._chunks:
            yield c

    def iter_lines(self, decode_unicode=True):  # noqa: ARG002 - requests-compatible
        # Reconstruct lines from chunks so SSE multi-line data works.
        joined = b"".join(self._chunks).decode("utf-8")
        for line in joined.split("\n"):
            yield line


class FakeHttpxResponse:
    """Mimic ``httpx.Response`` for LLMStream."""

    def __init__(self, chunks: List[bytes], content_type: str = "text/event-stream") -> None:
        self._chunks = chunks
        self.headers = {"content-type": content_type}

    def iter_bytes(self) -> Iterator[bytes]:
        for c in self._chunks:
            yield c

    def iter_lines(self) -> Iterator[str]:
        joined = b"".join(self._chunks).decode("utf-8")
        for line in joined.split("\n"):
            yield line


class FakeAiohttpResponse:
    """Mimic ``aiohttp.ClientResponse`` (async)."""

    def __init__(self, chunks: List[bytes], content_type: str = "text/event-stream") -> None:
        self._chunks = chunks
        self.headers = {"content-type": content_type}

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            yield c

    @property
    def content(self):  # pragma: no cover - async property
        return self


class FakeAiohttpContent:
    """Async content iterable used by aiohttp.ClientResponse."""

    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            yield c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


OPENAI_CHUNKS = [
    b'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":", "}}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":"world"}}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":"!"}}]}\n\n',
    b"data: [DONE]\n\n",
]

ANTHROPIC_CHUNKS = [
    b'event: message_start\ndata: {"type":"message_start"}\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"Hi"}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":" there"}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]

JSONL_CHUNKS = [
    b'{"token":"foo"}\n',
    b'{"token":"bar"}\n',
    b'{"token":"baz"}\n',
]

CHUNKED_BYTES = [
    b'{"a":1}{"b":2}{"c":3}',
]


# ---------------------------------------------------------------------------
# Test 1: version is exposed
# ---------------------------------------------------------------------------


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__ == "0.2.0"


# ---------------------------------------------------------------------------
# SSE: basic parsing
# ---------------------------------------------------------------------------


def test_parse_sse_basic_openai():
    parser = StreamParser()
    tokens = list(parser.parse_sse(OPENAI_CHUNKS))
    assert len(tokens) == 4
    assert [t.data["choices"][0]["delta"]["content"] for t in tokens] == [
        "Hello",
        ", ",
        "world",
        "!",
    ]
    assert [t.index for t in tokens] == [0, 1, 2, 3]


def test_parse_sse_stops_on_done_sentinel():
    parser = StreamParser()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
        b'data: [DONE]\n\n',
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',  # must be ignored
    ]
    tokens = list(parser.parse_sse(chunks))
    assert len(tokens) == 1
    assert tokens[0].data["choices"][0]["delta"]["content"] == "a"


def test_parse_sse_keeps_done_token_when_not_skipping():
    parser = StreamParser()
    chunks = [b'data: [DONE]\n\n']
    tokens = list(parser.parse_sse(chunks, skip_done=False))
    assert len(tokens) == 1
    assert tokens[0].done is True
    assert tokens[0].data == "[DONE]"


def test_parse_sse_handles_multiline_data():
    parser = StreamParser()
    chunks = [
        b"data: line1\n",
        b"data: line2\n",
        b"\n",
    ]
    tokens = list(parser.parse_sse(chunks))
    assert len(tokens) == 1
    # Multi-line data is joined by '\n'.
    assert tokens[0].raw == "line1\nline2"


def test_parse_sse_handles_crlf_line_endings():
    parser = StreamParser()
    chunks = [b'data: {"a":1}\r\n\r\n', b'data: {"a":2}\r\n\r\n']
    tokens = list(parser.parse_sse(chunks))
    assert [t.data for t in tokens] == [{"a": 1}, {"a": 2}]


def test_parse_sse_handles_comments_and_event_id():
    parser = StreamParser()
    chunks = [
        b": this is a comment\n",
        b"id: 42\n",
        b'event: ping\ndata: {"hello":"world"}\n',
        b"\n",
    ]
    tokens = list(parser.parse_sse(chunks))
    assert len(tokens) == 1
    assert tokens[0].event == "ping"
    assert tokens[0].event_id == "42"
    assert tokens[0].data == {"hello": "world"}


def test_parse_sse_handles_retry_field():
    parser = StreamParser()
    chunks = [b"retry: 5000\ndata: ok\n\n"]
    tokens = list(parser.parse_sse(chunks))
    assert tokens[0].raw == "ok"


def test_parse_sse_partial_chunks_across_boundaries():
    parser = StreamParser()
    # Split a single SSE event across three chunks.
    src = [
        b'data: {"chunks',
        b'":"split"}',
        b"\n\n",
    ]
    tokens = list(parser.parse_sse(src))
    assert tokens[0].data == {"chunks": "split"}


def test_parse_sse_byte_by_byte():
    parser = StreamParser()
    payload = 'data: {"k":1}\n\n'
    # Pass an iterable of bytes (one chunk) rather than iterating over the bytes.
    tokens = list(parser.parse_sse([payload.encode("utf-8")]))
    assert tokens[0].data == {"k": 1}


def test_parse_sse_anthropic_event_types():
    parser = StreamParser()
    tokens = list(parser.parse_sse(ANTHROPIC_CHUNKS))
    events = [t.event for t in tokens]
    assert events == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "message_stop",
    ]
    assert tokens[2].data["delta"]["text"] == "Hi"


def test_parse_sse_non_json_data_passes_through_as_string():
    parser = StreamParser()
    tokens = list(parser.parse_sse([b"data: not json\n\n"]))
    assert tokens[0].data == "not json"


def test_parse_sse_keepalive_comments_only_blocks():
    parser = StreamParser()
    # Two blank lines = one empty block = should yield nothing.
    tokens = list(parser.parse_sse([b": keepalive\n\n"]))
    assert tokens == []


def test_sse_event_is_done():
    event = SSEEvent(data="[DONE]")
    assert event.is_done() is True
    assert SSEEvent(data="  [DONE]  ").is_done() is True
    assert SSEEvent(data="other").is_done() is False


def test_sse_event_to_dict():
    event = SSEEvent(data='{"k":1}')
    assert event.to_dict() == {"k": 1}
    event = SSEEvent(data="raw text")
    assert event.to_dict() == {"raw": "raw text"}


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def test_parse_jsonl_basic():
    parser = StreamParser()
    tokens = list(parser.parse_jsonl(JSONL_CHUNKS))
    assert [t.data for t in tokens] == [{"token": "foo"}, {"token": "bar"}, {"token": "baz"}]


def test_parse_jsonl_stops_on_done():
    parser = StreamParser()
    chunks = [b'{"a":1}\n', b"[DONE]\n", b'{"a":2}\n']
    tokens = list(parser.parse_jsonl(chunks))
    assert [t.data for t in tokens] == [{"a": 1}]


def test_parse_jsonl_skips_empty_lines():
    parser = StreamParser()
    chunks = [b'{"a":1}\n', b"\n", b"   \n", b'{"a":2}\n']
    tokens = list(parser.parse_jsonl(chunks))
    assert [t.data for t in tokens] == [{"a": 1}, {"a": 2}]


def test_parse_jsonl_handles_non_json_gracefully():
    parser = StreamParser()
    chunks = [b"not json\n", b'{"k":1}\n']
    tokens = list(parser.parse_jsonl(chunks))
    assert tokens[0].data == "not json"
    assert tokens[1].data == {"k": 1}


def test_parse_jsonl_partial_chunks_across_boundaries():
    parser = StreamParser()
    chunks = [b'{"foo":', b'"bar"}\n']
    tokens = list(parser.parse_jsonl(chunks))
    assert tokens[0].data == {"foo": "bar"}


# ---------------------------------------------------------------------------
# Chunked
# ---------------------------------------------------------------------------


def test_parse_chunked_json_objects():
    parser = StreamParser()
    tokens = list(parser.parse_chunked(CHUNKED_BYTES, chunk_size=8))
    payloads = [t.data for t in tokens]
    assert {"a": 1} in payloads
    assert {"b": 2} in payloads
    assert {"c": 3} in payloads


def test_parse_chunked_byte_chunks():
    parser = StreamParser()
    tokens = list(parser.parse_chunked([b"abcdef"], chunk_size=2))
    assert [t.data for t in tokens] == ["ab", "cd", "ef"]


def test_parse_chunked_raises_on_invalid_size():
    parser = StreamParser()
    with pytest.raises(ValueError):
        list(parser.parse_chunked([b"x"], chunk_size=0))


def test_parse_chunked_handles_partial_json_then_bytes():
    parser = StreamParser()
    # First emit a complete JSON, then leave trailing bytes.
    chunks = [b'{"k":1}TRAILINGBYTES']
    tokens = list(parser.parse_chunked(chunks, chunk_size=4))
    assert tokens[0].data == {"k": 1}
    # Subsequent tokens must cover the remaining bytes.
    reconstructed = b"".join(
        t.data.encode("utf-8") if isinstance(t.data, str) else b""
        for t in tokens[1:]
    )
    assert reconstructed == b"TRAILINGBYTES"


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------


def test_parse_auto_detects_sse():
    parser = StreamParser()
    tokens = list(parser.parse(OPENAI_CHUNKS))
    assert len(tokens) == 4
    assert tokens[0].data["choices"][0]["delta"]["content"] == "Hello"


def test_parse_auto_detects_jsonl():
    parser = StreamParser()
    tokens = list(parser.parse(JSONL_CHUNKS))
    assert [t.data for t in tokens] == [{"token": "foo"}, {"token": "bar"}, {"token": "baz"}]


def test_parse_raises_on_unknown_format():
    parser = StreamParser()
    with pytest.raises(ValueError):
        list(parser.parse([b"x"], fmt="xml"))


# ---------------------------------------------------------------------------
# LLMStream — fake requests.Response
# ---------------------------------------------------------------------------


def test_llm_stream_with_fake_requests_response():
    response = FakeResponse(OPENAI_CHUNKS)
    stream = LLMStream(response)
    tokens = list(stream.iter_tokens())
    assert tokens == ["Hello", ", ", "world", "!"]


def test_llm_stream_with_httpx_response():
    response = FakeHttpxResponse(OPENAI_CHUNKS)
    stream = LLMStream(response)
    tokens = list(stream.iter_tokens())
    assert tokens == ["Hello", ", ", "world", "!"]


def test_llm_stream_returns_parsed_tokens_when_extract_disabled():
    response = FakeResponse(OPENAI_CHUNKS)
    stream = LLMStream(response, extract_tokens=False)
    tokens = list(stream.iter_tokens())
    assert all(isinstance(t, ParsedToken) for t in tokens)
    assert tokens[0].data["choices"][0]["delta"]["content"] == "Hello"


def test_llm_stream_explicit_format_jsonl():
    response = FakeResponse(JSONL_CHUNKS, content_type="application/x-ndjson")
    stream = LLMStream(response, format=StreamFormat.JSONL)
    # JSONL payloads with a top-level ``token`` key yield just that string.
    tokens = list(stream.iter_tokens())
    assert tokens == ["foo", "bar", "baz"]


def test_llm_stream_jsonl_extract_disabled_yields_parsed_tokens():
    response = FakeResponse(JSONL_CHUNKS, content_type="application/x-ndjson")
    stream = LLMStream(response, format=StreamFormat.JSONL, extract_tokens=False)
    tokens = list(stream.iter_tokens())
    assert [t.data for t in tokens] == [
        {"token": "foo"},
        {"token": "bar"},
        {"token": "baz"},
    ]


def test_llm_stream_gzip_decompression():
    body = b"".join(OPENAI_CHUNKS)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(body)
    response = FakeResponse(
        [buf.getvalue()],
        content_type="text/event-stream",
        content_encoding="gzip",
    )
    # requests-like iter_content emits a single gzip-encoded blob.
    stream = LLMStream(response)
    tokens = list(stream)
    assert tokens == ["Hello", ", ", "world", "!"]


def test_llm_stream_deflate_decompression():
    body = b"".join(OPENAI_CHUNKS)
    compressed = zlib.compress(body)
    response = FakeResponse(
        [compressed],
        content_type="text/event-stream",
        content_encoding="deflate",
    )
    stream = LLMStream(response)
    tokens = list(stream)
    assert tokens == ["Hello", ", ", "world", "!"]


def test_llm_stream_plain_iterable():
    def gen():
        yield from OPENAI_CHUNKS
    stream = LLMStream(gen())
    assert list(stream) == ["Hello", ", ", "world", "!"]


def test_llm_stream_async_iteration():
    async def agen() -> AsyncIterator[bytes]:
        for chunk in OPENAI_CHUNKS:
            yield chunk

    async def run() -> List[str]:
        stream = LLMStream(agen())
        out: List[str] = []
        async for token in stream.aiter_tokens():
            out.append(token)
        return out

    assert asyncio.run(run()) == ["Hello", ", ", "world", "!"]


def test_llm_stream_async_with_fake_aiohttp_response():
    class Resp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        @property
        def content(self):
            return FakeAiohttpContent(OPENAI_CHUNKS)

    stream = LLMStream(Resp())

    async def run() -> List[str]:
        out: List[str] = []
        async for token in stream.aiter_tokens():
            out.append(token)
        return out

    assert asyncio.run(run()) == ["Hello", ", ", "world", "!"]


def test_llm_stream_async_raises_when_called_sync_on_async_response():
    class Resp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        async def __aiter__(self):
            for chunk in OPENAI_CHUNKS:
                yield chunk

    stream = LLMStream(Resp())
    with pytest.raises(TypeError):
        list(stream.iter_tokens())


def test_llm_stream_anthropic_via_format():
    response = FakeResponse(ANTHROPIC_CHUNKS)
    stream = LLMStream(response)
    tokens = list(stream)
    assert tokens == ["Hi", " there"]


def test_llm_stream_chat_completion_legacy_text_field():
    chunks = [
        b'data: {"choices":[{"text":"once "}]}\n\n',
        b'data: {"choices":[{"text":"upon "}]}\n\n',
        b'data: {"choices":[{"text":"a "}]}\n\n',
        b'data: {"choices":[{"text":"time"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    response = FakeResponse(chunks)
    tokens = list(LLMStream(response))
    assert tokens == ["once ", "upon ", "a ", "time"]


def test_llm_stream_cohere_style_text():
    chunks = [
        b'data: {"text":"hello"}\n\n',
        b'data: {"text":" world"}\n\n',
        b"data: [DONE]\n\n",
    ]
    response = FakeResponse(chunks)
    tokens = list(LLMStream(response))
    assert tokens == ["hello", " world"]


def test_llm_stream_chunked_format():
    response = FakeResponse(CHUNKED_BYTES, content_type="application/octet-stream")
    stream = LLMStream(response, format=StreamFormat.CHUNKED, extract_tokens=False)
    tokens = list(stream)
    # chunked format yields ParsedToken objects.
    payloads = [t.data for t in tokens]
    assert {"a": 1} in payloads
    assert {"b": 2} in payloads


def test_llm_stream_auto_format_from_content_type():
    response = FakeResponse(JSONL_CHUNKS, content_type="application/x-ndjson")
    stream = LLMStream(response)
    tokens = list(stream)
    # JSONL ``token`` field comes through.
    assert tokens == ["foo", "bar", "baz"]


def test_llm_stream_iter_returns_iter_tokens():
    response = FakeResponse(OPENAI_CHUNKS)
    stream = LLMStream(response)
    # iter() must produce a generator that yields the same tokens.
    assert list(iter(stream)) == list(stream.iter_tokens())


def test_llm_stream_done_short_circuits():
    """After [DONE] no further tokens are emitted."""

    response = FakeResponse(OPENAI_CHUNKS)
    stream = LLMStream(response, extract_tokens=False)
    tokens = list(stream)
    assert all(not t.done for t in tokens[:-1]) or len(tokens) == 4


# ---------------------------------------------------------------------------
# Edge cases / robustness
# ---------------------------------------------------------------------------


def test_empty_source_yields_nothing():
    parser = StreamParser()
    assert list(parser.parse_sse([])) == []
    assert list(parser.parse_jsonl([])) == []
    assert list(parser.parse_chunked([])) == []


def test_mixed_crlf_and_lf_in_sse():
    parser = StreamParser()
    chunks = [
        b'data: {"a":1}\r\n\r\n',
        b'data: {"a":2}\n\n',
    ]
    tokens = list(parser.parse_sse(chunks))
    assert [t.data for t in tokens] == [{"a": 1}, {"a": 2}]


def test_string_source_is_supported():
    parser = StreamParser()
    payload = 'data: {"a":1}\n\ndata: {"a":2}\n\n'
    tokens = list(parser.parse_sse(iter([payload])))
    assert [t.data for t in tokens] == [{"a": 1}, {"a": 2}]


def test_parser_reuse_is_safe():
    """Reusing a parser for two streams should not leak buffers."""

    parser = StreamParser()
    list(parser.parse_sse([b'data: {"a":1}\n\n']))
    tokens = list(parser.parse_sse([b'data: {"b":2}\n\n']))
    assert tokens[0].data == {"b": 2}


def test_extract_tokens_handles_non_dict():
    from tiny_stream import _extract_tokens

    assert _extract_tokens("raw") == ["raw"]
    assert _extract_tokens(None) == []
    assert _extract_tokens([{"text": "x"}, {"text": "y"}]) == ["x", "y"]


def test_extract_tokens_anthropic_payload():
    from tiny_stream import _extract_tokens

    payload = {"type": "content_block_delta", "delta": {"text": "hello"}}
    assert _extract_tokens(payload) == ["hello"]


def test_extract_tokens_delta_text_payload():
    from tiny_stream import _extract_tokens

    payload = {"delta": {"text": "world"}}
    assert _extract_tokens(payload) == ["world"]


def test_extract_tokens_token_field():
    from tiny_stream import _extract_tokens

    payload = {"token": "!"}
    assert _extract_tokens(payload) == ["!"]


def test_parsed_token_done_flag_propagates():
    parser = StreamParser()
    chunks = [b'data: [DONE]\n\n']
    tokens = list(parser.parse_sse(chunks, skip_done=False))
    assert tokens[0].done is True
    assert tokens[0].data == "[DONE]"


def test_parse_sse_string_with_no_event_field():
    parser = StreamParser()
    tokens = list(parser.parse_sse([b'data: {"a":1}\n\n']))
    assert tokens[0].event is None
    assert tokens[0].event_id is None


# ---------------------------------------------------------------------------
# Final tally
# ---------------------------------------------------------------------------


def test_total_tests_at_least_25():
    """Sanity check: the test module should contain >= 25 tests."""

    import test_tiny_stream as t  # noqa: PLC0415 - self import for inspection

    count = sum(
        1
        for name, obj in vars(t).items()
        if name.startswith("test_") and callable(obj)
    )
    assert count >= 25, f"only {count} tests defined"