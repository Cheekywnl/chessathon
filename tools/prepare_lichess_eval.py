"""Streams the Lichess open evaluation database (CC0-licensed, 409M+ positions annotated by
real Stockfish search -- https://database.lichess.org/, "lichess_db_eval") into a CSV this
project's tools/train_nnue.py can read directly, without ever downloading the full multi-GB
file: reads the .zst stream incrementally and stops once --limit positions have been written.

Explicitly the kind of data use the competition's own rules describe as allowed: "training it on
positions an existing engine labelled is allowed" -- this ships nothing, trains nothing by
itself, and produces no network weights on its own. It only prepares a CSV; tools/train_nnue.py
still trains a network from a random initialization on this data (plus, optionally, this
project's own self-play data), never starting from or fine-tuning any published network, which
the rules draw a hard line at.

Run this on whichever machine has the bandwidth and disk for a multi-GB streaming download --
almost certainly not this repo's own dev machine. Needs `zstandard` (pip install zstandard).

Usage:
    pip install zstandard requests
    uv run python -m tools.prepare_lichess_eval \
        --url <the exact eval-file URL from https://database.lichess.org/> \
        --limit 3000000 --min-depth 12 --out data/lichess_eval_sample.csv

--url is required and deliberately not defaulted: verify the exact current download link on
database.lichess.org yourself rather than trust a URL hardcoded here, since this file is
periodically regenerated and the exact path has been inconsistently reported by automated
fetches of that page during this project's own development.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import requests  # type: ignore[import-untyped]
    import zstandard  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "This script needs `requests` and `zstandard` (pip install requests zstandard) -- "
        "meant to run wherever you have bandwidth for a large streaming download, not "
        "necessarily this repo's own dev machine."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-sample the Lichess eval database into a training CSV."
    )
    parser.add_argument(
        "--url", type=str, required=True,
        help="Exact eval-file URL, verified from https://database.lichess.org/ yourself.",
    )
    parser.add_argument("--limit", type=int, default=3_000_000, help="Positions to write.")
    parser.add_argument(
        "--min-depth", type=int, default=12,
        help="Skip evals shallower than this -- shallow search is a noisier label.",
    )
    parser.add_argument("--sample-every", type=int, default=1, help="Keep 1 line in N seen.")
    parser.add_argument("--out", type=Path, default=Path("data/lichess_eval_sample.csv"))
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0

    with requests.get(arguments.url, stream=True) as response:
        response.raise_for_status()
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(response.raw) as reader:
            raw_chunks = iter(lambda: reader.read(65536), b"")
            text_stream = (chunk.decode("utf-8", errors="ignore") for chunk in raw_chunks)
            buffer = ""
            with open(arguments.out, "w") as out_f:
                out_f.write("fen,depth,cp\n")
                for chunk in text_stream:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        seen += 1
                        if seen % arguments.sample_every != 0:
                            continue
                        try:
                            record = json.loads(line)
                            fen = record["fen"]
                            best_eval = max(record["evals"], key=lambda e: e["depth"])
                            if best_eval["depth"] < arguments.min_depth:
                                continue
                            pv = best_eval["pvs"][0]
                            if "cp" not in pv:
                                continue  # skip mate-only lines for this first pass
                            cp = pv["cp"]
                        except (KeyError, IndexError, json.JSONDecodeError):
                            continue
                        out_f.write(f'"{fen}",{best_eval["depth"]},{cp}\n')
                        written += 1
                        if written % 50_000 == 0:
                            print(f"written {written}/{arguments.limit} (seen {seen})", flush=True)
                        if written >= arguments.limit:
                            print(f"\nreached limit: {written} positions -> {arguments.out}")
                            return

    print(f"\nstream ended: {written} positions -> {arguments.out}")


if __name__ == "__main__":
    main()
