"""Benchmark parsing throughput for tiny-stream.

Measures tokens/sec for SSE, JSON-lines, and chunked transfer parsing.

Usage:
    python benchmark.py                 # default sizes
    python benchmark.py --tokens 100000 # bigger stream
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Iterable, List

from tiny_stream import StreamParser


# ---------------------------------------------------------------------------
# Stream generators
# ---------------------------------------------------------------------------


def gen_sse(num_tokens: int) -> List[bytes]:
    """Generate ``num_tokens`` SSE chunks in a single list."""

    chunks: List[bytes] = []
    for i in range(num_tokens):
        payload = {
            "id": f"chatcmpl-{i}",
            "choices": [{"delta": {"content": f"tok{i} "}}],
        }
        chunks.append(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def gen_jsonl(num_tokens: int) -> List[bytes]:
    """Generate ``num_tokens`` JSON-lines."""

    chunks: List[bytes] = []
    for i in range(num_tokens):
        chunks.append(json.dumps({"text": f"tok{i} "}).encode("utf-8") + b"\n")
    return chunks


def gen_chunked(num_tokens: int) -> List[bytes]:
    """Generate ``num_tokens`` JSON objects concatenated."""

    body = "".join(
        json.dumps({"text": f"tok{i} "}) for i in range(num_tokens)
    )
    return [body.encode("utf-8")]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(
    label: str,
    source: Iterable[bytes],
    parse_fn: Callable[[Iterable[bytes]], int],
) -> float:
    """Run ``parse_fn`` over ``source`` and report tokens/sec."""

    # Warm-up.
    parse_fn(source)

    start = time.perf_counter()
    iterations = 3
    total_tokens = 0
    for _ in range(iterations):
        total_tokens = parse_fn(source)
    elapsed = time.perf_counter() - start
    per_run = elapsed / iterations
    rate = total_tokens / per_run if per_run > 0 else float("inf")
    print(
        f"{label:<16} {total_tokens:>8,} tokens  "
        f"{per_run * 1000:>8.2f} ms/run  "
        f"{rate:>12,.0f} tok/s"
    )
    return rate


def _consume_sse(source: Iterable[bytes]) -> int:
    parser = StreamParser()
    count = 0
    for _ in parser.parse_sse(source):
        count += 1
    return count


def _consume_jsonl(source: Iterable[bytes]) -> int:
    parser = StreamParser()
    count = 0
    for _ in parser.parse_jsonl(source):
        count += 1
    return count


def _consume_chunked(source: Iterable[bytes]) -> int:
    parser = StreamParser()
    count = 0
    for _ in parser.parse_chunked(source, chunk_size=32):
        count += 1
    return count


# ---------------------------------------------------------------------------
# Comparison vs raw line iteration
# ---------------------------------------------------------------------------


def _consume_sse_naive(source: Iterable[bytes]) -> int:
    """Naive SSE parser for comparison: split on ``\\n\\n``, JSON-decode."""

    import json as _json

    data = b"".join(source).decode("utf-8")
    count = 0
    for block in data.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    return count
                _json.loads(payload)
                count += 1
    return count


def _consume_jsonl_naive(source: Iterable[bytes]) -> int:
    """Naive JSON-lines parser for comparison."""

    import json as _json

    count = 0
    data = b"".join(source).decode("utf-8")
    for line in data.split("\n"):
        if line.strip():
            _json.loads(line)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tiny-stream parsing.")
    parser.add_argument(
        "--tokens",
        type=int,
        default=10_000,
        help="Number of tokens/objects per format (default: 10,000).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the naive baseline comparison.",
    )
    args = parser.parse_args()

    n = args.tokens
    print(f"\ntiny-stream benchmark ({n:,} tokens per format)\n")
    print(
        f"{'parser':<16} {'tokens':>10} {'time':>12} {'rate':>16}"
    )
    print("-" * 60)

    sse = gen_sse(n)
    jsonl = gen_jsonl(n)
    chunked = gen_chunked(n)

    rates = {
        "sse": benchmark("sse (ours)", sse, _consume_sse),
        "jsonl": benchmark("jsonl (ours)", jsonl, _consume_jsonl),
        "chunked": benchmark("chunked (ours)", chunked, _consume_chunked),
    }

    if not args.no_baseline:
        print()
        print("baseline (manual split + json.loads):")
        print("-" * 60)
        benchmark("sse (naive)", sse, _consume_sse_naive)
        benchmark("jsonl (naive)", jsonl, _consume_jsonl_naive)

    print()
    print("summary (tokens/sec, higher is better):")
    for label, rate in rates.items():
        print(f"  {label:<10} {rate:>12,.0f}")


if __name__ == "__main__":
    main()