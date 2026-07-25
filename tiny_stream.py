"""tiny-stream: streaming response parser for LLM APIs.

Parse Server-Sent Events (SSE), JSON-lines, and chunked transfer encoding
from LLM streaming responses (OpenAI, Anthropic, local models, etc.) with a
tiny, dependency-free API.

Example
-------
>>> from tiny_stream import StreamParser
>>> parser = StreamParser()
>>> for token in parser.parse_sse(openai_chunk_stream):
...     print(token, end="", flush=True)
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

__version__ = "0.1.0"

__all__ = [
    "StreamParser",
    "LLMStream",
    "SSEEvent",
    "ParsedToken",
    "StreamFormat",
    "__version__",
]


# Type alias: any byte/string iterator that we can read line by line.
ByteStream = Iterable[Union[bytes, str]]
StreamSource = Union[ByteStream, Iterable[str], "LLMStream"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SSEEvent:
    """A single Server-Sent Event parsed from a stream."""

    data: str = ""
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None
    # The raw multiline data field (lines joined by '\n'). Useful when callers
    # need to forward the exact bytes the server sent.
    raw: str = ""

    def is_done(self) -> bool:
        """Return True if this event represents an end-of-stream sentinel."""

        return self.data.strip() == "[DONE]"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-decoded representation when possible."""

        try:
            return json.loads(self.data)
        except (ValueError, TypeError):
            return {"raw": self.data}


@dataclass
class ParsedToken:
    """A single parsed token/object produced by one of the parsers."""

    # The decoded JSON payload, or the raw string when JSON parsing fails.
    data: Any
    # The raw line/text that produced this token.
    raw: str = ""
    # Event name (SSE only).
    event: Optional[str] = None
    # Event id (SSE only).
    event_id: Optional[str] = None
    # 1-based index within the stream. Useful for benchmarks and debugging.
    index: int = 0
    # Set to True when this token represents an end-of-stream sentinel.
    done: bool = False


class StreamFormat:
    """Format constants used by ``LLMStream`` to pick a parser automatically."""

    SSE = "sse"
    JSONL = "jsonl"
    CHUNKED = "chunked"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import-only branch
    import requests  # type: ignore
except Exception:  # pragma: no cover - import-only branch
    requests = None

try:  # pragma: no cover - import-only branch
    import httpx  # type: ignore
except Exception:  # pragma: no cover - import-only branch
    httpx = None

try:  # pragma: no cover - import-only branch
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover - import-only branch
    aiohttp = None


# ---------------------------------------------------------------------------
# StreamParser
# ---------------------------------------------------------------------------


class StreamParser:
    """Parse streaming LLM responses in multiple wire formats.

    The parser is stateless across calls — every method is a generator that
    accepts any iterable of bytes/strings and yields :class:`ParsedToken`
    instances.  Buffers are kept per-call so the same parser instance can be
    reused safely across multiple streams.
    """

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------

    def parse_sse(
        self,
        source: ByteStream,
        skip_done: bool = True,
    ) -> Iterator[ParsedToken]:
        """Parse Server-Sent Events from ``source``.

        Handles standard SSE framing (``data:``, ``event:``, ``id:``,
        ``retry:``), comments, multi-line ``data`` fields, CRLF/LF line
        endings, and the ``data: [DONE]`` sentinel used by OpenAI.
        """
        buffer = ""
        index = 0
        for chunk in _normalize_source(source):
            buffer += chunk
            while True:
                # SSE events are separated by blank lines (CRLF or LF).
                sep_index, sep_len = _find_sse_separator(buffer)
                if sep_index == -1:
                    break
                raw_block = buffer[:sep_index]
                buffer = buffer[sep_index + sep_len :]
                event = _parse_sse_block(raw_block)
                if event is None:
                    continue
                if skip_done and event.is_done():
                    return
                token = _event_to_token(event, index)
                index += 1
                yield token

    # ------------------------------------------------------------------
    # JSON-lines (ndjson)
    # ------------------------------------------------------------------

    def parse_jsonl(
        self,
        source: ByteStream,
        skip_done: bool = True,
        skip_empty: bool = True,
    ) -> Iterator[ParsedToken]:
        """Parse newline-delimited JSON objects.

        Each non-empty line is parsed as JSON.  Lines equal to ``[DONE]``
        terminate the stream by default.
        """
        buffer = ""
        index = 0
        for chunk in _normalize_source(source):
            buffer += chunk
            while True:
                newline = buffer.find("\n")
                if newline == -1:
                    break
                line = buffer[:newline]
                buffer = buffer[newline + 1 :]
                line = line.rstrip("\r")
                if skip_empty and not line.strip():
                    continue
                if skip_done and line.strip() == "[DONE]":
                    return
                token = _jsonl_line_to_token(line, index)
                index += 1
                yield token

    # ------------------------------------------------------------------
    # Chunked transfer encoding
    # ------------------------------------------------------------------

    def parse_chunked(
        self,
        source: ByteStream,
        chunk_size: int = 64,
    ) -> Iterator[ParsedToken]:
        """Parse raw chunked-transfer streams.

        Splits the incoming byte stream into fixed-size byte chunks and
        reconstructs arbitrary objects by buffering bytes until a full JSON
        document is available.  ``chunk_size`` controls the emit granularity
        for non-JSON streams (default: 64 bytes).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        buffer = ""
        json_index = 0
        bytes_index = 0
        for chunk in _normalize_source(source):
            buffer += chunk

            # First, attempt to peel complete JSON objects from the buffer.
            while buffer:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                obj, consumed = _try_parse_json(stripped)
                if obj is not None:
                    yield ParsedToken(
                        data=obj,
                        raw=stripped[:consumed],
                        index=json_index,
                    )
                    json_index += 1
                    buffer = stripped[consumed:]
                else:
                    break

            # Then emit any leftover bytes as fixed-size chunks.
            while len(buffer) >= chunk_size:
                piece = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                yield ParsedToken(
                    data=piece,
                    raw=piece,
                    index=bytes_index,
                )
                bytes_index += 1

        # Flush remaining bytes at end-of-stream.
        if buffer:
            yield ParsedToken(
                data=buffer,
                raw=buffer,
                index=bytes_index,
            )

    # ------------------------------------------------------------------
    # JSON-lines (ndjson)
    # ------------------------------------------------------------------

    def parse_jsonl(
        self,
        source: ByteStream,
        skip_done: bool = True,
        skip_empty: bool = True,
    ) -> Iterator[ParsedToken]:
        """Parse newline-delimited JSON objects.

        Each non-empty line is parsed as JSON.  Lines equal to ``[DONE]``
        terminate the stream by default.
        """
        buffer = ""
        index = 0
        for chunk in _normalize_source(source):
            buffer += chunk
            # Split on \n first, but also retain a partial trailing fragment.
            while True:
                newline = buffer.find("\n")
                if newline == -1:
                    break
                line = buffer[:newline]
                buffer = buffer[newline + 1 :]
                line = line.rstrip("\r")
                if skip_empty and not line.strip():
                    continue
                if skip_done and line.strip() == "[DONE]":
                    return
                token = _jsonl_line_to_token(line, index)
                index += 1
                yield token

    # ------------------------------------------------------------------
    # Chunked transfer encoding
    # ------------------------------------------------------------------

    def parse_chunked(
        self,
        source: ByteStream,
        chunk_size: int = 64,
    ) -> Iterator[ParsedToken]:
        """Parse raw chunked-transfer streams.

        Splits the incoming byte stream into fixed-size byte chunks and
        reconstructs arbitrary objects by buffering bytes until a full JSON
        document is available.  ``chunk_size`` controls the emit granularity
        for non-JSON streams (default: 64 bytes).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        buffer = ""
        json_index = 0
        bytes_index = 0
        for chunk in _normalize_source(source):
            buffer += chunk

            # First, attempt to peel complete JSON objects from the buffer.
            while buffer:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                obj, consumed = _try_parse_json(stripped)
                if obj is not None:
                    yield ParsedToken(
                        data=obj,
                        raw=stripped[:consumed],
                        index=json_index,
                    )
                    json_index += 1
                    buffer = stripped[consumed:]
                else:
                    break

            # Then emit any leftover bytes as fixed-size chunks.
            while len(buffer) >= chunk_size:
                piece = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                yield ParsedToken(
                    data=piece,
                    raw=piece,
                    index=bytes_index,
                )
                bytes_index += 1

        # Flush remaining bytes at end-of-stream.
        if buffer:
            yield ParsedToken(
                data=buffer,
                raw=buffer,
                index=bytes_index,
            )

    # ------------------------------------------------------------------
    # Convenience auto-dispatch
    # ------------------------------------------------------------------

    def parse(
        self,
        source: ByteStream,
        fmt: str = StreamFormat.AUTO,
        **kwargs: Any,
    ) -> Iterator[ParsedToken]:
        """Dispatch to the right parser based on ``fmt``.

        ``fmt`` may be ``"sse"``, ``"jsonl"``, ``"chunked"`` or
        ``"auto"``.  ``"auto"`` inspects the first non-empty line: SSE blocks
        start with ``data:`` or ``event:``; everything else falls back to
        JSON-lines.
        """
        fmt = (fmt or StreamFormat.AUTO).lower()
        if fmt == StreamFormat.AUTO:
            fmt = _detect_format(source)
        if fmt == StreamFormat.SSE:
            return self.parse_sse(source, **kwargs)
        if fmt == StreamFormat.JSONL:
            return self.parse_jsonl(source, **kwargs)
        if fmt == StreamFormat.CHUNKED:
            return self.parse_chunked(source, **kwargs)
        raise ValueError(f"unknown stream format: {fmt!r}")


# ---------------------------------------------------------------------------
# LLMStream — convenience wrapper
# ---------------------------------------------------------------------------


def _extract_tokens(obj: Any) -> List[str]:
    """Pull token strings out of a decoded SSE/JSONL payload.

    Supports a wide range of vendor payloads: OpenAI chat (``choices[].delta``),
    OpenAI legacy (``choices[].text``), Anthropic (``delta.text`` /
    ``content_block_delta``), and Cohere.
    """
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        out: List[str] = []
        for item in obj:
            out.extend(_extract_tokens(item))
        return out
    if isinstance(obj, Mapping):
        # OpenAI chat.completion.chunk
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, Mapping):
                    content = delta.get("content")
                    if isinstance(content, str):
                        if content:
                            return [content]
                    if content is None and "text" in delta:
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            return [text]
                if "text" in choice:
                    text = choice.get("text")
                    if isinstance(text, str) and text:
                        return [text]
                # Some providers put content directly on the choice.
                if "content" in choice:
                    content = choice.get("content")
                    if isinstance(content, str) and content:
                        return [content]
        # Anthropic messages stream
        if obj.get("type") == "content_block_delta":
            delta = obj.get("delta")
            if isinstance(delta, Mapping):
                text = delta.get("text")
                if isinstance(text, str):
                    return [text]
        if "delta" in obj and isinstance(obj["delta"], Mapping):
            text = obj["delta"].get("text")
            if isinstance(text, str):
                return [text]
        # Cohere / generic
        if "text" in obj and isinstance(obj["text"], str):
            return [obj["text"]]
        if "token" in obj and isinstance(obj["token"], str):
            return [obj["token"]]
    return []


class LLMStream:
    """High-level wrapper around :class:`StreamParser` for real HTTP responses.

    Parameters
    ----------
    response:
        Any object that exposes ``iter_lines()`` / ``iter_content()`` /
        ``content`` and that supports streaming.  Supported types include
        ``requests.Response``, ``httpx.Response`` (sync or async),
        ``aiohttp.ClientResponse``, an async generator from
        ``openai.AsyncStream``, or a plain iterable of strings/bytes.

    format:
        ``"sse"`` (default), ``"jsonl"``, ``"chunked"`` or ``"auto"``.

    decompress:
        When True (default), transparently inflate gzip/deflate payloads.
        Auto-detected from the ``Content-Encoding`` header when present.

    extract_tokens:
        When True (default), yields plain token strings pulled out of the
        decoded payload.  When False, yields :class:`ParsedToken` instances.
    """

    def __init__(
        self,
        response: Any,
        format: str = StreamFormat.AUTO,
        decompress: bool = True,
        extract_tokens: bool = True,
        parser: Optional[StreamParser] = None,
    ) -> None:
        self.response = response
        self.format = (format or StreamFormat.AUTO).lower()
        self.decompress = decompress
        self.extract_tokens = extract_tokens
        self._parser = parser or StreamParser()
        self._decompressor: Optional[Union[gzip.GzipFile, "io.BufferedReader"]] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Synchronous iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Union[str, ParsedToken]]:
        return self.iter_tokens()

    def iter_tokens(self) -> Iterator[Union[str, ParsedToken]]:
        """Yield tokens (strings by default, :class:`ParsedToken` otherwise)."""

        source = self._iter_source()
        fmt = self.format
        if fmt == StreamFormat.AUTO:
            fmt = _detect_format_from_response(self.response)
        stream: Iterator[ParsedToken]
        if fmt == StreamFormat.SSE:
            stream = self._parser.parse_sse(source)
        elif fmt == StreamFormat.JSONL:
            stream = self._parser.parse_jsonl(source)
        elif fmt == StreamFormat.CHUNKED:
            stream = self._parser.parse_chunked(source)
        else:  # pragma: no cover - guarded above
            raise ValueError(f"unknown stream format: {fmt!r}")

        for token in stream:
            if token.done:
                return
            if not self.extract_tokens:
                yield token
                continue
            for piece in _extract_tokens(token.data):
                if piece:
                    yield piece

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    async def aiter_tokens(self) -> "AsyncIterator[Union[str, ParsedToken]]":
        """Async equivalent of :meth:`iter_tokens`.

        The async source is drained into a list of string chunks (typical LLM
        responses are a few MB at most, so buffering is fine) and then handed
        to the sync parser.  This keeps the parser state machines in one
        place without duplicating their buffering logic.
        """
        source = self._aiter_source()
        # Drain the async source to completion.  For typical LLM streams this
        # is a few thousand tokens, well within reasonable memory bounds.
        chunks: List[str] = []
        async for chunk in source:
            chunks.append(chunk)

        fmt = self.format
        if fmt == StreamFormat.AUTO:
            fmt = _detect_format_from_response(self.response)
        stream: Iterator[ParsedToken]
        if fmt == StreamFormat.SSE:
            stream = self._parser.parse_sse(iter(chunks))
        elif fmt == StreamFormat.JSONL:
            stream = self._parser.parse_jsonl(iter(chunks))
        elif fmt == StreamFormat.CHUNKED:
            stream = self._parser.parse_chunked(iter(chunks))
        else:  # pragma: no cover - guarded above
            raise ValueError(f"unknown stream format: {fmt!r}")

        for token in stream:
            if token.done:
                return
            if not self.extract_tokens:
                yield token
                continue
            for piece in _extract_tokens(token.data):
                if piece:
                    yield piece

    # ------------------------------------------------------------------
    # Source adapters
    # ------------------------------------------------------------------

    def _ensure_decompressor(self, encoding: str) -> None:
        if self._decompressor is not None or not self.decompress:
            return
        encoding = (encoding or "").lower().strip()
        if encoding in ("gzip", "x-gzip"):
            # Stream-friendly gzip decoder; works on chunked inputs.
            self._decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)  # type: ignore[assignment]
        elif encoding in ("deflate",):
            self._decompressor = zlib.decompressobj()  # type: ignore[assignment]

    def _iter_source(self) -> Iterator[str]:
        """Return a string iterator over the underlying response body.

        Uses duck typing so we work with ``requests``, ``httpx``, ``aiohttp``,
        ``openai`` streams, and any plain iterable of bytes/strings.
        """

        response = self.response
        encoding = _get_encoding(response)
        self._ensure_decompressor(encoding)
        # Native async iterables must be consumed via aiter_tokens().
        if _is_async_iterable(response):
            raise TypeError(
                "response is async; use LLMStream.aiter_tokens() instead",
            )
        # requests-style: iter_content() + iter_lines()
        iter_content = getattr(response, "iter_content", None)
        iter_lines = getattr(response, "iter_lines", None)
        if self._decompressor is not None and callable(iter_content):
            for chunk in iter_content(chunk_size=None):
                if not chunk:
                    continue
                yield _decode_bytes(self._inflate(_to_bytes(chunk)))
            return
        if callable(iter_lines):
            # ``requests.Response.iter_lines`` strips the terminator from each
            # line.  We re-add it so SSE framing (``\n\n``) is preserved.
            try:
                line_iter = iter_lines(decode_unicode=True)
            except TypeError:
                line_iter = iter_lines()
            for line in line_iter:
                if line is None:
                    continue
                if not isinstance(line, str):
                    line = line.decode("utf-8", "replace")
                if not line.endswith("\n"):
                    line += "\n"
                yield line
            return
        # httpx-style: iter_bytes()
        iter_bytes = getattr(response, "iter_bytes", None)
        if self._decompressor is not None and callable(iter_bytes):
            for chunk in iter_bytes():
                yield _decode_bytes(self._inflate(_to_bytes(chunk)))
            return
        if callable(iter_bytes):
            for chunk in iter_bytes():
                if isinstance(chunk, bytes):
                    yield _decode_bytes(self._inflate(chunk))
                else:
                    yield chunk
            return
        # httpx-style: iter_lines() (sync)
        if callable(iter_lines):
            for line in iter_lines():
                if line is None:
                    continue
                if not isinstance(line, str):
                    line = line.decode("utf-8", "replace")
                if not line.endswith("\n"):
                    line += "\n"
                yield line
            return
        # aiohttp-style: response.content is async-only.
        content = getattr(response, "content", None)
        if content is not None and _is_async_iterable(content):
            raise TypeError(
                "response is async; use LLMStream.aiter_tokens()",
            )
        if content is not None and hasattr(content, "__iter__"):
            for chunk in content:
                yield _decode_bytes(self._inflate(_to_bytes(chunk)))
            return
        # Plain bytes/string iterator.
        for chunk in response:
            if isinstance(chunk, bytes):
                yield _decode_bytes(self._inflate(chunk))
            else:
                yield chunk

    async def _aiter_source(self) -> AsyncIterator[str]:
        """Return an async string iterator over the underlying response body."""

        response = self.response
        encoding = _get_encoding(response)
        self._ensure_decompressor(encoding)
        # aiohttp.ClientResponse: ``response.content`` is async.
        content = getattr(response, "content", None)
        if content is not None and _is_async_iterable(content):
            async for chunk in content:
                yield _decode_bytes(self._inflate(_to_bytes(chunk)))
            return
        # httpx async aiter_bytes().
        aiter_bytes = getattr(response, "aiter_bytes", None)
        if callable(aiter_bytes):
            async for chunk in aiter_bytes():
                if isinstance(chunk, bytes):
                    yield _decode_bytes(self._inflate(chunk))
                else:
                    yield chunk
            return
        # Async generators / async iterables.
        if _is_async_iterable(response):
            async for chunk in response:
                if isinstance(chunk, bytes):
                    yield _decode_bytes(self._inflate(chunk))
                else:
                    yield chunk
            return
        # Fall back to the sync iterator.
        for chunk in self._iter_source():
            yield chunk

    # ------------------------------------------------------------------
    # Decompression helpers
    # ------------------------------------------------------------------

    def _inflate(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        # ``self._decompressor`` is a ``zlib._Decompress`` instance for both
        # gzip (windowBits=MAX_WBITS|16) and deflate (windowBits=MAX_WBITS).
        if hasattr(self._decompressor, "decompress"):
            data = self._decompressor.decompress(chunk)
            data += self._decompressor.flush()
            return data
        return chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_source(source: ByteStream) -> Iterator[str]:
    """Yield string fragments from an arbitrary byte/string iterable."""

    for chunk in source:
        if chunk is None:
            continue
        if isinstance(chunk, bytes):
            yield chunk.decode("utf-8", "replace")
        else:
            yield chunk


def _decode_bytes(data: bytes) -> str:
    if not data:
        return ""
    return data.decode("utf-8", "replace")


def _to_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, str):
        return chunk.encode("utf-8")
    return bytes(chunk)


def _find_sse_separator(buffer: str) -> "tuple[int, int]":
    """Find the next blank-line separator in an SSE buffer.

    Returns ``(index, length)`` where ``length`` is the byte length of the
    separator itself (``2`` for ``\\n\\n``, ``4`` for ``\\r\\n\\r\\n``),
    or ``(-1, 0)`` when no separator is present yet.
    """
    crlf_index = buffer.find("\r\n\r\n")
    lf_index = buffer.find("\n\n")
    if crlf_index == -1 and lf_index == -1:
        return -1, 0
    if crlf_index == -1:
        return lf_index, 2
    if lf_index == -1:
        return crlf_index, 4
    # Pick whichever comes first; they don't overlap.
    if crlf_index <= lf_index:
        return crlf_index, 4
    return lf_index, 2


def _parse_sse_block(block: str) -> Optional[SSEEvent]:
    """Parse a single SSE event block (one blank-line-delimited chunk)."""

    if not block:
        return None
    event = SSEEvent()
    data_lines: List[str] = []
    for line in block.split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        if line.startswith(":"):
            # SSE comment line.
            continue
        field_name, _, value = line.partition(":")
        # Spec: a single space after the colon is stripped.
        if value.startswith(" "):
            value = value[1:]
        if field_name == "data":
            data_lines.append(value)
        elif field_name == "event":
            event.event = value
        elif field_name == "id":
            event.id = value
        elif field_name == "retry":
            try:
                event.retry = int(value)
            except ValueError:
                pass
    event.data = "\n".join(data_lines)
    event.raw = event.data
    # A block with no data field is a comment/keep-alive.
    if not data_lines and event.event is None and event.id is None and event.retry is None:
        return None
    return event


def _event_to_token(event: SSEEvent, index: int) -> ParsedToken:
    payload: Any = event.data
    try:
        payload = json.loads(event.data) if event.data else None
    except (ValueError, TypeError):
        payload = event.data
    return ParsedToken(
        data=payload,
        raw=event.data,
        event=event.event,
        event_id=event.id,
        index=index,
        done=event.is_done(),
    )


def _jsonl_line_to_token(line: str, index: int) -> ParsedToken:
    payload: Any = line
    try:
        payload = json.loads(line)
    except (ValueError, TypeError):
        payload = line
    return ParsedToken(data=payload, raw=line, index=index)


_JSON_START = re.compile(r"[\{\[]")


def _try_parse_json(buffer: str) -> "tuple[Any, int]":
    """Try to parse a complete JSON value at the start of ``buffer``.

    Returns ``(value, consumed_chars)`` on success, or ``(None, 0)`` when
    the buffer contains an incomplete JSON value.
    """
    stripped = buffer.lstrip()
    leading_ws = len(buffer) - len(stripped)
    if not stripped:
        return None, 0
    # Quick reject for non-JSON starts (saves a lot of work on raw bytes).
    if not _JSON_START.match(stripped):
        return None, 0
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None, 0
    return obj, leading_ws + end


def _detect_format(source: ByteStream) -> str:
    """Peek at the first non-empty line to guess the wire format."""

    for chunk in _normalize_source(source):
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:") or line.startswith("event:"):
                return StreamFormat.SSE
            if line.startswith("{") or line.startswith("["):
                return StreamFormat.JSONL
            return StreamFormat.CHUNKED
    return StreamFormat.JSONL


def _detect_format_from_response(response: Any) -> str:
    """Use response headers / type to choose a sensible default format."""

    content_type = _get_header(response, "content-type", "")
    if "text/event-stream" in content_type:
        return StreamFormat.SSE
    if "application/x-ndjson" in content_type or "application/jsonl" in content_type:
        return StreamFormat.JSONL
    if "json" in content_type:
        return StreamFormat.JSONL
    return StreamFormat.SSE


def _get_encoding(response: Any) -> str:
    return _get_header(response, "content-encoding", "") or ""


def _get_header(response: Any, name: str, default: str = "") -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return default
    try:
        value = headers.get(name)  # requests, httpx case-insensitive
    except Exception:
        value = None
    if value is None and isinstance(headers, Mapping):
        # Case-insensitive lookup for plain dicts.
        lower = name.lower()
        for k, v in headers.items():
            if isinstance(k, str) and k.lower() == lower:
                return str(v)
        return default
    return value if value is not None else default


def _is_async_iterable(obj: Any) -> bool:
    return hasattr(obj, "__aiter__") or hasattr(obj, "__anext__")