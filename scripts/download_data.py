from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


LONGMEMEVAL_BASE = (
    "https://huggingface.co/datasets/xiaowu0162/"
    "longmemeval-cleaned/resolve/main"
)
LONGMEMEVAL_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s_cleaned": "longmemeval_s_cleaned.json",
    "m_cleaned": "longmemeval_m_cleaned.json",
}
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    "main/data/locomo10.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {url}\n       -> {destination}", flush=True)
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        completed = 0
        while block := response.read(1024 * 1024):
            output.write(block)
            completed += len(block)
            if total:
                print(f"\r{completed / total:6.1%}", end="", flush=True)
    if total:
        print()
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("longmemeval", "locomo"), required=True)
    parser.add_argument(
        "--split",
        choices=tuple(LONGMEMEVAL_FILES),
        default="s_cleaned",
        help="LongMemEval release file",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dataset == "locomo":
        destination = Path("data/external/locomo/locomo10.json")
        if destination.exists() and not args.force:
            print(f"Already present: {destination} (sha256={sha256(destination)})")
            return
        download(LOCOMO_URL, destination)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"Unexpected LoCoMo payload in {destination}")
        qa_count = sum(len(sample.get("qa", [])) for sample in payload)
        print(
            f"Ready: {destination} ({len(payload)} conversations, {qa_count} QA "
            f"examples, sha256={sha256(destination)})"
        )
        return
    filename = LONGMEMEVAL_FILES[args.split]
    destination = Path("data/external/longmemeval") / filename
    if destination.exists() and not args.force:
        print(f"Already present: {destination} (sha256={sha256(destination)})")
        return
    download(f"{LONGMEMEVAL_BASE}/{filename}", destination)
    # Parse immediately so an HTML/error payload cannot be mistaken for benchmark data.
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected LongMemEval payload in {destination}")
    print(f"Ready: {destination} ({len(payload)} examples, sha256={sha256(destination)})")


if __name__ == "__main__":
    main()
