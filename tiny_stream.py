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

import asyncio
import itertools
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

__version__ = "0.2.0"

__all__ = [
    "StreamParser",
    "LLMStream",
    "SSEEvent",
    "ParsedToken",
    "StreamFormat",
    "SSERelay",
    "TokenCounter",
    "StreamMerger",
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

    # ------------------------------------------------------------------
    # JSONPath filter
    # ------------------------------------------------------------------

    def filter(
        self,
        source: Union[ByteStream, Iterator[ParsedToken], AsyncIterator[ParsedToken]],
        path: str,
        fmt: str = StreamFormat.AUTO,
        **kwargs: Any,
    ) -> Iterator[ParsedToken]:
        """Yield only tokens matching a tiny JSONPath expression.

        Parameters
        ----------
        source:
            A raw byte/string stream (parsed via ``fmt``) **or** an
            iterator/async-iterator of :class:`ParsedToken` objects
            (e.g. the output of another parser or :class:`SSERelay`).

        path:
            A JSONPath expression.  Supported subset:

            * ``"$"`` — the root value (matches every token)
            * ``"$.field"`` — object field access
            * ``"$.a.b.c"`` — nested field access
            * ``"$[0]"`` — array index
            * ``"$.items[2]"`` — field then index
            * ``"$.items[*]"`` — wildcard for arrays
            * ``"$.items[*].name"`` — wildcard then field
            * ``"$[*]"`` — root array wildcard

        fmt:
            When ``source`` is a raw byte stream, the format hint passed
            to :meth:`parse` (default: ``"auto"``).  Ignored when
            ``source`` is already a :class:`ParsedToken` iterator.

        Yields
        ------
        ParsedToken
            Tokens whose decoded ``data`` resolves to a non-empty list of
            matches under ``path``.  Empty matches are skipped silently.

        Notes
        -----
        This is a deliberately small subset of JSONPath — no filters,
        recursive descent, or script expressions.  It exists to pull
        token strings out of vendor payloads without pulling in a
        third-party dependency.

        Examples
        --------
        >>> parser = StreamParser()
        >>> for tok in parser.filter(sse_source, "$.choices[*].delta.content"):
        ...     print(tok.data)
        """
        if not path or not path.startswith("$"):
            raise ValueError("path must start with '$'")

        compiled = _compile_jsonpath(path)

        # If source is already an iterator of ParsedToken, consume it directly.
        if _looks_like_token_iter(source):
            for token in source:
                matches = _apply_jsonpath(compiled, token.data)
                if matches:
                    yield token
            return

        # Otherwise, run a parser first and then filter.
        for token in self.parse(source, fmt=fmt, **kwargs):
            matches = _apply_jsonpath(compiled, token.data)
            if matches:
                yield token


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
# SSERelay — SSE → WebSocket bridge for browser frontends
# ---------------------------------------------------------------------------


class SSERelay:
    """Forward Server-Sent Events to multiple WebSocket clients concurrently.

    The typical workflow is:

        1. An upstream LLM streams SSE chunks (e.g. from a worker process).
        2. A web frontend opens a WebSocket and expects the same content
           in the wire format most browsers can handle (``text/event-stream``
           is one-way SSE, while WebSockets are bidirectional).
        3. :class:`SSERelay` consumes the SSE stream once and fans it out
           to every connected WebSocket client.

    This class is **transport-agnostic** and uses ``asyncio`` only.  Plug
    in any object that exposes ``send()`` / ``send_bytes()`` coroutines
    (the shape used by the ``websockets`` and ``aiohttp`` WebSocket
    libraries): register it with :meth:`attach`, push SSE events with
    :meth:`push` (or :meth:`feed`), and unregister with :meth:`detach`.

    Lifecycle
    ---------
    * :meth:`attach` returns a unique ``client_id``; the caller is
      responsible for calling :meth:`detach` when the client disconnects.
    * :meth:`close` detaches every client and waits for in-flight
      broadcasts to complete.
    * :meth:`wait_closed` returns ``True`` when the relay has no more
      clients and is idle.

    Concurrency
    -----------
    * Broadcasting is fanned out with :func:`asyncio.gather`; one slow
      client will not block the others.
    * Per-client send call timeouts are supported via ``send_timeout``
      (defaults to 10 seconds) — a client that misses the timeout is
      removed automatically.

    Example
    -------
    >>> relay = SSERelay()
    >>> async def on_connect(ws):
    ...     return relay.attach(ws)
    >>> async def on_disconnect(client_id):
    ...     relay.detach(client_id)
    >>> async def pump_sse():
    ...     async for line in sse_response.aiter_lines():
    ...         await relay.feed(line)
    """

    # WebSocket library adapters to detect.  These module names are only
    # checked at attach() time using duck-typing; we never import them.
    _WS_LIKE_ATTRS = ("send", "send_str", "send_bytes", "send_text")

    def __init__(self, send_timeout: float = 10.0) -> None:
        """Create a new SSERelay.

        Parameters
        ----------
        send_timeout:
            Maximum number of seconds to wait for any single client to
            accept a message.  Clients that exceed this are detached
            automatically.  Set to ``None`` to disable.
        """
        self._clients: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._closed = False
        self._send_timeout = send_timeout

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    async def attach(self, ws: Any) -> str:
        """Register a WebSocket-like client and return its new ``client_id``.

        The client must expose an async ``send()`` method (or
        ``send_str()`` / ``send_text()`` / ``send_bytes()``).
        """
        if self._closed:
            raise RuntimeError("SSERelay is closed")
        if not _is_ws_like(ws):
            raise TypeError(
                "ws must expose an async send() method (websockets/aiohttp style)",
            )
        async with self._lock:
            self._next_id += 1
            client_id = f"client-{self._next_id}"
            self._clients[client_id] = ws
        return client_id

    async def detach(self, client_id: str) -> None:
        """Remove a client from the relay.  No-op if not registered."""

        async with self._lock:
            self._clients.pop(client_id, None)

    async def close(self) -> None:
        """Detach all clients and mark the relay as closed."""

        async with self._lock:
            self._closed = True
            self._clients.clear()

    def client_count(self) -> int:
        """Return the current number of connected clients."""

        return len(self._clients)

    def is_closed(self) -> bool:
        """Return ``True`` if the relay has been closed."""

        return self._closed

    async def wait_closed(self) -> bool:
        """Return ``True`` when the relay has zero clients and is closed.

        Useful as an ``asyncio.Event.wait`` equivalent for graceful shutdown.
        """
        while self._closed is False or len(self._clients) > 0:
            await asyncio.sleep(0.01)
        return True

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def push(self, event: Union[str, bytes, SSEEvent]) -> None:
        """Broadcast a single SSE event to every connected client.

        ``event`` may be:

        * a ``str`` — sent verbatim (typically a single ``data: ...\\n\\n``
          block);
        * a ``bytes`` — encoded UTF-8 and sent verbatim;
        * an :class:`SSEEvent` — serialized to standard SSE framing.

        A ``"[DONE]"`` SSE event closes the relay (clients are notified
        with the event first, then the relay is marked closed).
        """
        if self._closed:
            return
        payload = _sse_event_to_wire(event)
        if payload is None:
            return
        # [DONE] implies end-of-stream — broadcast it and then close.
        if _payload_means_done(payload):
            await self._broadcast(payload)
            await self.close()
            return
        await self._broadcast(payload)

    async def feed(self, event: Union[str, bytes, SSEEvent]) -> None:
        """Alias for :meth:`push` (matches the SSERelay examples in the README)."""

        await self.push(event)

    async def feed_sse_token(
        self,
        token: ParsedToken,
        event_name: str = "message",
        wrap_json: bool = True,
    ) -> None:
        """Convenience: take a :class:`ParsedToken` and push it as SSE.

        When ``wrap_json`` is True (default), the token's decoded data
        is JSON-encoded.  When False, the raw ``str(token.data)`` is
        placed in the ``data:`` field.
        """
        if self._closed:
            return
        if token.done:
            await self.push(SSEEvent(data="[DONE]"))
            return
        if wrap_json:
            try:
                data = json.dumps(token.data)
            except (TypeError, ValueError):
                data = str(token.data)
        else:
            data = str(token.data)
        await self.push(SSEEvent(data=data, event=event_name))

    async def _broadcast(self, payload: str) -> None:
        """Send ``payload`` to all clients concurrently."""

        if not self._clients:
            return
        # Snapshot the client list so detach() during send is safe.
        async with self._lock:
            clients = list(self._clients.items())
        if not clients:
            return

        async def _send_one(client_id: str, ws: Any) -> None:
            try:
                coro = _ws_send(ws, payload)
                if self._send_timeout is not None:
                    await asyncio.wait_for(coro, timeout=self._send_timeout)
                else:
                    await coro
            except asyncio.TimeoutError:
                # Slow client — drop them so the relay stays healthy.
                await self.detach(client_id)
            except Exception:
                # Defensive: a faulty client should not bring the relay down.
                await self.detach(client_id)

        await asyncio.gather(
            *(_send_one(cid, ws) for cid, ws in clients),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# TokenCounter — wrap a stream and count tokens for billing/cost tracking
# ---------------------------------------------------------------------------


class TokenCounter:
    """Wrap any string/ParsedToken stream and count tokens via a counter fn.

    Useful for billing dashboards, cost tracking, and rate-limit budget
    enforcement when you're streaming responses from a paid LLM API.

    Parameters
    ----------
    source:
        A synchronous iterable of strings, an async iterable of strings,
        or a :class:`ParsedToken` iterable.  Strings are counted as-is;
        :class:`ParsedToken` objects are counted by their decoded JSON
        (``str(token.data)``) so vendor payload structure doesn't matter.

    counter:
        A callable ``(str) -> int`` used to count tokens.  The default
        ``len()`` counts characters — fine for rough byte budgets and
        deterministic in tests.  Plug in ``tiktoken.encoding_for_model(...)``
        or any tokenizer for accurate counts.

    extract_tokens:
        When True (default) and ``source`` yields :class:`ParsedToken`
        instances, use the vendor-aware :func:`_extract_tokens` to pick
        out the text before counting.  Set to False to count the JSON
        envelope instead.

    Attributes
    ----------
    total_tokens:
        Running total of tokens counted so far.  Updated as the wrapped
        stream is consumed.

    Examples
    --------
    >>> total = 0
    >>> async def run():
    ...     async for token in LLMStream(response).aiter_tokens():
    ...         counted = TokenCounter([token], counter=lambda s: len(s.split()))
    ...         async for word in counted:
    ...             print(word)
    ...         print("words:", counted.total_tokens)
    """

    DEFAULT_COUNTER: Callable[[str], int] = staticmethod(len)  # type: ignore[assignment]

    def __init__(
        self,
        source: Union[
            Iterable[Union[str, ParsedToken]],
            AsyncIterator[Union[str, ParsedToken]],
        ],
        counter: Optional[Callable[[str], int]] = None,
        extract_tokens: bool = True,
    ) -> None:
        self._source = source
        self._counter: Callable[[str], int] = counter or TokenCounter.DEFAULT_COUNTER
        self._extract_tokens = extract_tokens
        self.total_tokens: int = 0

    # ------------------------------------------------------------------
    # Sync iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Union[str, ParsedToken]]:
        return self.iter_tokens()

    def iter_tokens(self) -> Iterator[Union[str, ParsedToken]]:
        """Yield items from the source, counting tokens as we go."""

        if _is_async_iterable(self._source):
            raise TypeError(
                "source is async; use TokenCounter.aiter_tokens() instead",
            )
        for item in self._source:  # type: ignore[union-attr]
            self._count_item(item)
            yield item

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    async def aiter_tokens(self) -> AsyncIterator[Union[str, ParsedToken]]:
        """Async equivalent of :meth:`iter_tokens`."""

        if not _is_async_iterable(self._source):
            # Plain sync iterable — still works.
            for item in self._source:  # type: ignore[union-attr]
                self._count_item(item)
                yield item
            return
        async for item in self._source:
            self._count_item(item)
            yield item

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_item(self, item: Union[str, ParsedToken]) -> None:
        if isinstance(item, ParsedToken):
            if self._extract_tokens:
                for piece in _extract_tokens(item.data):
                    if piece:
                        self.total_tokens += self._counter(piece)
            else:
                try:
                    text = json.dumps(item.data)
                except (TypeError, ValueError):
                    text = str(item.data)
                self.total_tokens += self._counter(text)
        elif isinstance(item, str):
            self.total_tokens += self._counter(item)
        else:
            self.total_tokens += self._counter(str(item))


# ---------------------------------------------------------------------------
# StreamMerger — combine multiple async streams into one ordered stream
# ---------------------------------------------------------------------------


class StreamMerger:
    """Merge multiple async streams into a single async stream.

    Useful for multi-model fan-out: query several LLMs in parallel and
    present a unified stream to the caller.  Two ordering modes are
    supported:

    * ``order="interleaved"`` (default) — yield items as soon as any
      source produces one, regardless of the source index.
    * ``order="round_robin"`` — interleave one item from each source
      in turn, blocking on a slow source until it produces (or until
      the merger is cancelled).

    Both modes preserve the **per-stream** ordering of items (i.e. you
    never see item N+1 from source A before item N).

    Parameters
    ----------
    streams:
        List of async iterables (one per source).

    order:
        ``"interleaved"`` or ``"round_robin"``.

    stop_on_error:
        If True (default), the first exception from any source cancels
        the merger and propagates.  If False, exceptions are silenced
        and the affected source is dropped.

    Examples
    --------
    >>> async def chat(model):
    ...     async for tok in LLMStream(model).aiter_tokens():
    ...         yield tok
    >>> merger = StreamMerger([chat("gpt-4"), chat("claude-3")])
    >>> async for token in merger.aiter_tokens():
    ...     print(token)
    """

    def __init__(
        self,
        streams: Sequence[AsyncIterator[Any]],
        order: str = "interleaved",
        stop_on_error: bool = True,
    ) -> None:
        if order not in ("interleaved", "round_robin"):
            raise ValueError(
                f"order must be 'interleaved' or 'round_robin', got {order!r}",
            )
        for stream in streams:
            if not _is_async_iterable(stream):
                raise TypeError(
                    "all streams must be async iterables; "
                    "wrap sync iterables with an async generator",
                )
        self._streams: List[AsyncIterator[Any]] = list(streams)
        self._order = order
        self._stop_on_error = stop_on_error

    async def aiter_tokens(self) -> AsyncIterator[Any]:
        """Yield items from all streams, merged according to ``order``."""

        if self._order == "interleaved":
            async for item in self._iter_interleaved():
                yield item
        else:
            async for item in self._iter_round_robin():
                yield item

    # ------------------------------------------------------------------
    # Interleaved mode
    # ------------------------------------------------------------------

    async def _iter_interleaved(self) -> AsyncIterator[Any]:
        """Yield items as soon as any source produces one."""

        queues: List[asyncio.Queue] = [
            asyncio.Queue(maxsize=1) for _ in self._streams
        ]
        active: Set[int] = set(range(len(self._streams)))
        first_error: Optional[BaseException] = None

        async def pump(idx: int, src: AsyncIterator[Any]) -> None:
            nonlocal first_error
            try:
                async for item in src:
                    await queues[idx].put(item)
            except Exception as exc:  # noqa: BLE001 - propagated by main loop
                if first_error is None:
                    first_error = exc
                # Sentinel so the loop can wake up.
                await queues[idx].put(_DONE)
            else:
                await queues[idx].put(_DONE)

        tasks = [
            asyncio.create_task(pump(i, src)) for i, src in enumerate(self._streams)
        ]
        try:
            while active:
                # Wait for any queue to produce an item.
                get_coros = [asyncio.ensure_future(q.get()) for i, q in enumerate(queues) if i in active]
                if not get_coros:
                    break
                done, pending = await asyncio.wait(
                    get_coros,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for fut in pending:
                    fut.cancel()
                for fut in done:
                    idx = get_coros.index(fut)
                    item = fut.result()
                    if item is _DONE:
                        active.discard(idx)
                        continue
                    yield item
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if first_error is not None and self._stop_on_error:
            raise first_error

    # ------------------------------------------------------------------
    # Round-robin mode
    # ------------------------------------------------------------------

    async def _iter_round_robin(self) -> AsyncIterator[Any]:
        """Take one item from each source in turn, skipping exhausted ones."""

        iterators: List[AsyncIterator[Any]] = list(self._streams)
        pending_gets: Dict[asyncio.Task, int] = {}
        first_error: Optional[BaseException] = None

        try:
            while iterators:
                next_iterators: List[AsyncIterator[Any]] = []
                for src in iterators:
                    try:
                        item = await src.__anext__()
                        yield item
                        next_iterators.append(src)
                    except StopAsyncIteration:
                        continue
                    except Exception as exc:  # noqa: BLE001
                        if first_error is None:
                            first_error = exc
                        if self._stop_on_error:
                            return
                        continue
                iterators = next_iterators
                if not pending_gets:
                    break
        finally:
            for task in pending_gets:
                task.cancel()
        if first_error is not None and self._stop_on_error:
            raise first_error


# Sentinel used by StreamMerger to signal end-of-stream per source.
_DONE: Any = object()


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

# ---------------------------------------------------------------------------
# JSONPath helpers (tiny dependency-free subset)
# ---------------------------------------------------------------------------


# AST nodes:
#   ("root",)                          — the root value
#   ("field", "name")                   — object field access
#   ("index", 0)                       — array index
#   ("wild",)                          — array wildcard
_JPNode = Tuple[Any, ...]


def _compile_jsonpath(path: str) -> List[_JPNode]:
    """Compile a tiny JSONPath expression into a list of AST nodes.

    Grammar::

        path     ::= "$" segment*
        segment  ::= "." NAME | "[" (INT | "*") "]"
        NAME     ::= [A-Za-z_][A-Za-z0-9_]*
        INT      ::= "0" | "-?[1-9][0-9]*"

    No recursive descent, no filters, no scripts.  Anything more complex
    belongs in a real JSONPath library.
    """
    if not path.startswith("$"):
        raise ValueError(f"path must start with '$', got {path!r}")
    nodes: List[_JPNode] = [("root",)]
    i = 1
    n = len(path)
    while i < n:
        ch = path[i]
        if ch == ".":
            i += 1
            j = i
            while j < n and (path[j].isalnum() or path[j] == "_"):
                j += 1
            if j == i:
                raise ValueError(f"empty field name at offset {i} in {path!r}")
            nodes.append(("field", path[i:j]))
            i = j
        elif ch == "[":
            j = path.find("]", i)
            if j == -1:
                raise ValueError(f"unclosed '[' at offset {i} in {path!r}")
            inner = path[i + 1 : j].strip()
            if inner == "*":
                nodes.append(("wild",))
            else:
                try:
                    idx = int(inner)
                except ValueError:
                    raise ValueError(
                        f"only integer indices or '*' allowed inside '[]', got {inner!r}",
                    ) from None
                nodes.append(("index", idx))
            i = j + 1
        else:
            raise ValueError(
                f"unexpected character {ch!r} at offset {i} in {path!r}",
            )
    return nodes


def _apply_jsonpath(compiled: List[_JPNode], value: Any) -> List[Any]:
    """Apply a compiled JSONPath to ``value`` and return the match list.

    Returns an empty list when nothing matches.  Wildcards flatten arrays
    into intermediate results.
    """
    current: List[Any] = [value]
    for node in compiled[1:]:  # skip the root marker
        if not current:
            return []
        kind = node[0]
        if kind == "field":
            name = node[1]
            next_vals: List[Any] = []
            for v in current:
                if isinstance(v, Mapping) and name in v:
                    next_vals.append(v[name])
            current = next_vals
        elif kind == "index":
            idx = node[1]
            next_vals = []
            for v in current:
                if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
                    if -len(v) <= idx < len(v):
                        next_vals.append(v[idx])
            current = next_vals
        elif kind == "wild":
            next_vals = []
            for v in current:
                if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
                    next_vals.extend(v)
            current = next_vals
        else:
            return []
    # When the path is just "$", the input is the result.
    return current


def _looks_like_token_iter(obj: Any) -> bool:
    """Return True if ``obj`` looks like a ParsedToken iterator.

    Used by :meth:`StreamParser.filter` to decide whether to call the
    parser first or to consume the iterator directly.  Peek-checks the
    first item without disturbing the iterator.
    """
    # Strongest signal: it's a ParsedToken.
    if isinstance(obj, ParsedToken):
        return True
    # Async iterables are handled separately — we treat them as ParsedToken
    # streams and never auto-parse them.
    if _is_async_iterable(obj):
        # We can't peek without consuming an async iterator's first item,
        # so we conservatively assume it is a token stream.  This matches
        # the documented contract of ``filter()`` when called with an
        # async iterator.
        return True
    # Iterator: try to peek at the first item, then re-chain.
    if hasattr(obj, "__iter__"):
        try:
            iter_obj = iter(obj)
            first = next(iter_obj)
        except (StopIteration, TypeError):
            return False
        if isinstance(first, ParsedToken):
            # Re-chain so the caller sees all items, including the first.
            obj = itertools.chain([first], iter_obj)
            return True
        return False
    return False


# ---------------------------------------------------------------------------
# SSERelay helpers
# ---------------------------------------------------------------------------


def _is_ws_like(obj: Any) -> bool:
    """Return True if ``obj`` exposes any websockets-style async send method."""

    for attr in ("send", "send_str", "send_text", "send_bytes"):
        method = getattr(obj, attr, None)
        if method is not None and callable(method):
            return True
    return False


async def _ws_send(ws: Any, payload: str) -> None:
    """Send ``payload`` via whichever API the WebSocket library exposes."""

    # Prefer the most explicit one for the payload type.
    send_str = getattr(ws, "send_str", None)
    if callable(send_str):
        await send_str(payload)
        return
    send_text = getattr(ws, "send_text", None)
    if callable(send_text):
        await send_text(payload)
        return
    send_bytes = getattr(ws, "send_bytes", None)
    if callable(send_bytes):
        await send_bytes(payload.encode("utf-8"))
        return
    send = getattr(ws, "send", None)
    if callable(send):
        # ``websockets`` WebSocket.send() takes a str or bytes.
        await send(payload)
        return
    raise TypeError("ws object has no usable send() method")


def _sse_event_to_wire(event: Union[str, bytes, SSEEvent]) -> Optional[str]:
    """Serialize an event into a wire-format SSE string."""

    if isinstance(event, bytes):
        return event.decode("utf-8", "replace")
    if isinstance(event, str):
        return event
    if not isinstance(event, SSEEvent):
        return None
    out: List[str] = []
    if event.event:
        out.append(f"event: {event.event}")
    if event.id:
        out.append(f"id: {event.id}")
    if event.retry is not None:
        out.append(f"retry: {event.retry}")
    if event.data:
        for line in event.data.split("\n"):
            out.append(f"data: {line}")
    if not out:
        return None
    out.append("")  # blank line terminator
    out.append("")
    return "\n".join(out)


def _payload_means_done(payload: str) -> bool:
    """Return True if ``payload`` contains a ``[DONE]`` SSE sentinel."""

    for line in payload.split("\n"):
        if line.startswith("data: "):
            data = line[len("data: ") :].strip()
            if data == "[DONE]":
                return True
    return False
