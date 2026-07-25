# tiny-stream

**Streaming response parser for LLM APIs.**
Parse Server-Sent Events (SSE), JSON-lines, and chunked transfer encoding from
OpenAI, Anthropic, Cohere, and local models with a tiny, dependency-free API.

[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Version](https://img.shields.io/badge/version-0.1.0-blue)]()

```python
from tiny_stream import LLMStream

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    json={"model": "gpt-4", "stream": True, **payload},
    headers={"Authorization": f"Bearer {api_key}"},
    stream=True,
)

for token in LLMStream(response):
    print(token, end="", flush=True)
```

---

## Why streaming matters for LLM UX

LLMs generate text one token at a time, but a typical chat completion might
emit 200–2,000 tokens. Users perceive latency two ways:

- **Time to first token (TTFT)** — the wait until anything appears on screen.
- **Tokens per second (TPS)** — how fast subsequent tokens arrive.

Streaming collapses the wait. Even a 3-second model that streams at 30 TPS
shows the first word in ~100 ms, then fills the screen character-by-character
instead of freezing for 3 seconds. That single change converts a frustrating
"is this thing broken?" pause into a feeling of responsiveness on par with a
human typing.

To deliver that experience you need:

1. **A streaming HTTP client** (`stream=True` on requests, `stream()` on httpx,
   etc.)
2. **A wire-format parser** that handles SSE framing, partial chunks, the
   `[DONE]` sentinel, and gzip on the wire.
3. **A token extractor** that pulls the right field out of a vendor-specific
   JSON payload.

`tiny-stream` handles (2) and (3) for you in ~500 lines of pure Python.

---

## Features

- ✅ **SSE** — Server-Sent Events (`data:`, `event:`, `id:`, `retry:`,
  multi-line `data`, comments, CRLF/LF, `[DONE]` sentinel)
- ✅ **JSON-lines** — newline-delimited JSON (one object per line)
- ✅ **Chunked** — raw chunked transfer encoding with JSON-aware splitting
- ✅ **Format auto-detection** from response content-type or first bytes
- ✅ **Gzip / deflate** decompression on the wire
- ✅ **Partial-chunk buffering** — chunks split mid-event are reassembled
- ✅ **Vendor-agnostic token extraction** — OpenAI, Anthropic, Cohere,
  legacy completions, generic
- ✅ **Async support** via `aiter_tokens()` for aiohttp / httpx / openai
  async streams
- ✅ **Dependency-free** — pure Python stdlib only
- ✅ **56 tests, ~250k+ SSE tokens/sec, ~500k JSONL tokens/sec** on a modern
  laptop

---

## Installation

```bash
pip install tiny-stream
```

Or from source:

```bash
git clone https://github.com/hussain-alsaibai/tiny-stream.git
cd tiny-stream
pip install -e .
```

---

## Quick start

### OpenAI chat completion

```python
import requests
from tiny_stream import LLMStream

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    json={
        "model": "gpt-4",
        "stream": True,
        "messages": [{"role": "user", "content": "Tell me a joke."}],
    },
    headers={"Authorization": f"Bearer {api_key}"},
    stream=True,
)

for token in LLMStream(response):
    print(token, end="", flush=True)
```

### Anthropic messages

```python
response = requests.post(
    "https://api.anthropic.com/v1/messages",
    json={
        "model": "claude-3-opus-20240229",
        "stream": True,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi"}],
    },
    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    stream=True,
)

for token in LLMStream(response):
    print(token, end="", flush=True)
```

### httpx (async)

```python
import httpx
from tiny_stream import LLMStream

async with httpx.AsyncClient() as client:
    async with client.stream(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={...},
        headers={...},
    ) as response:
        stream = LLMStream(response)
        async for token in stream.aiter_tokens():
            print(token, end="", flush=True)
```

### aiohttp

```python
import aiohttp
from tiny_stream import LLMStream

async with aiohttp.ClientSession() as session:
    async with session.post(url, json=payload, headers=headers) as response:
        stream = LLMStream(response)
        async for token in stream.aiter_tokens():
            print(token, end="", flush=True)
```

### Local model (JSON-lines)

```python
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={"prompt": "Once upon", "stream": True},
    stream=True,
)
for token in LLMStream(response, format="jsonl"):
    print(token, end="", flush=True)
```

---

## Low-level API: `StreamParser`

If you have raw bytes (e.g. from a websocket or a custom transport), use
`StreamParser` directly.

```python
from tiny_stream import StreamParser

parser = StreamParser()

# SSE
for token in parser.parse_sse(sse_byte_stream):
    print(token)

# JSON-lines
for token in parser.parse_jsonl(ndjson_byte_stream):
    print(token)

# Chunked transfer (raw bytes / partial JSON)
for token in parser.parse_chunked(byte_stream, chunk_size=128):
    print(token)

# Auto-detect
for token in parser.parse(byte_stream):
    print(token)
```

Each method returns an iterator of `ParsedToken` objects with these fields:

| Field      | Type             | Description                              |
| ---------- | ---------------- | ---------------------------------------- |
| `data`     | `Any`            | Decoded JSON payload, or raw string      |
| `raw`      | `str`            | The raw text line that produced it       |
| `event`    | `Optional[str]`  | SSE event name                           |
| `event_id` | `Optional[str]`  | SSE event id                             |
| `index`    | `int`            | 0-based position in the stream           |
| `done`     | `bool`           | True for `[DONE]` sentinels              |

---

## Architecture

```
                          ┌─────────────────────┐
                          │   HTTP response     │
                          │ (requests/httpx/    │
                          │  aiohttp/openai)    │
                          └──────────┬──────────┘
                                     │
                          gzip/deflate? ──► zlib.decompressobj
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │     LLMStream       │
                          │  ┌───────────────┐  │
                          │  │ format detect │  │  ← Content-Type / first bytes
                          │  └───────┬───────┘  │
                          └──────────┼──────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
        ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
        │ parse_sse()   │    │ parse_jsonl() │    │parse_chunked()│
        │  SSE state    │    │  line buffer  │    │ JSON detect + │
        │  machine      │    │               │    │ fixed-size    │
        └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                └────────────────────┼────────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │   _extract_tokens() │
                          │ vendor-aware        │
                          │ (OpenAI/Anthropic/  │
                          │  Cohere/generic)    │
                          └──────────┬──────────┘
                                     ▼
                          ┌─────────────────────┐
                          │  str | ParsedToken  │
                          │  (yielded to user)  │
                          └─────────────────────┘
```

The parser is **stateless across calls**: each method maintains its own
internal buffer, so a single `StreamParser` instance can be reused for
multiple streams safely.

---

## Comparison

### vs raw `response.iter_lines()`

```python
# Without tiny-stream — works for OpenAI but breaks on:
#   - partial chunks that split SSE events
#   - Anthropic event types (only emits "data:" lines)
#   - gzip-encoded responses
#   - non-data SSE fields (event:, id:, retry:, comments)
for line in response.iter_lines():
    if line.startswith("data: "):
        chunk = json.loads(line[6:])
        if chunk == "[DONE]":
            break
        print(chunk["choices"][0]["delta"].get("content", ""), end="")
```

```python
# With tiny-stream — works across vendors and edge cases.
for token in LLMStream(response):
    print(token, end="")
```

### vs `openai.StreamCompletion`

| Feature                          | `openai.Stream`  | `tiny-stream`     |
| -------------------------------- | ---------------- | ----------------- |
| Vendor lock-in                   | OpenAI only      | Any HTTP library  |
| Handles partial chunks           | ✅               | ✅                |
| Handles `[DONE]` sentinel        | ✅               | ✅                |
| Handles gzip                     | ✅               | ✅                |
| Custom transports (websockets)   | ❌               | ✅                |
| JSON-lines format                | ❌               | ✅                |
| Raw byte streams                 | ❌               | ✅                |
| Async iteration                  | ✅               | ✅                |
| `ParsedToken` metadata           | Limited          | Full              |
| Dependencies                     | `openai`         | None              |

`tiny-stream` is a great choice when you want **one parser for many LLM
providers** or you're working outside the official SDKs.

---

## Format auto-detection

`LLMStream` picks a parser from the response's `Content-Type` header:

| Content-Type contains | Format |
| --------------------- | ------ |
| `text/event-stream`   | SSE    |
| `application/x-ndjson` or `application/jsonl` | JSONL |
| `application/json`    | JSONL  |
| anything else         | SSE    |

You can override with `format="jsonl"` or `format="chunked"`. Use
`format="auto"` (the default) to fall back to byte-level sniffing — useful
when the server mislabels its content-type.

---

## Vendor token extraction

`_extract_tokens()` recognizes these payload shapes and yields the right
field:

| Vendor             | Payload shape (abbreviated)                                  | Yields   |
| ------------------ | ------------------------------------------------------------ | -------- |
| OpenAI chat        | `{"choices":[{"delta":{"content":"..."}}]}`                  | content  |
| OpenAI legacy      | `{"choices":[{"text":"..."}]}`                               | text     |
| Anthropic          | `{"type":"content_block_delta","delta":{"text":"..."}}`     | text     |
| Anthropic (flat)   | `{"delta":{"text":"..."}}`                                   | text     |
| Cohere / generic   | `{"text":"..."}` or `{"token":"..."}`                       | text     |

Pass `extract_tokens=False` to receive the raw `ParsedToken` instead.

---

## Performance

```
$ python benchmark.py --tokens 50000

tiny-stream benchmark (50,000 tokens per format)

parser               tokens         time             rate
------------------------------------------------------------
sse (ours)         50,000 tokens    200.14 ms/run       249,830 tok/s
jsonl (ours)       50,000 tokens    100.91 ms/run       495,482 tok/s
chunked (ours)     50,000 tokens    870.47 ms/run        57,441 tok/s
```

On a modern laptop tiny-stream sustains **~250k SSE tokens/sec** and **~500k
JSONL tokens/sec** — comfortably above the throughput of every commercial LLM
API (the fastest models top out at ~200 tokens/sec). Parsing is not the
bottleneck; the network is.

---

## Running tests

```bash
python -m pytest test_tiny_stream.py -v
```

All 56 tests run in <1 second and cover:

- SSE parsing (basic, multi-line, CRLF, partial chunks, comments, event ids,
  retry field, byte-level)
- JSON-lines parsing (basic, partial chunks, done sentinel, empty lines,
  non-JSON)
- Chunked transfer (JSON objects, byte chunks, partial-then-bytes, invalid
  size)
- Auto-detection
- `LLMStream` with fake requests/httpx/aiohttp responses
- Gzip and deflate decompression
- Async iteration
- Token extraction across vendor payload shapes

---

## Files

```
tiny_stream.py      Main library (~500 lines, zero deps)
test_tiny_stream.py 56 tests
example.py          OpenAI / Anthropic / local / async / gzip examples
benchmark.py        Throughput benchmark
README.md           This file
pyproject.toml      Package metadata
LICENSE             MIT
```

---

## License

MIT — see [LICENSE](LICENSE).