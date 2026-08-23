from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def download_range(
    *, url: str, part_path: Path, start: int, end: int, retries: int = 12
) -> tuple[int, int]:
    expected = end - start + 1
    part_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        current = part_path.stat().st_size if part_path.exists() else 0
        if current == expected:
            return start, expected
        if current > expected:
            raise RuntimeError(f"oversized part {part_path}: {current}>{expected}")
        headers = {"Range": f"bytes={start + current}-{end}"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 15)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"server ignored range: HTTP {response.status_code}")
                with part_path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if part_path.stat().st_size == expected:
                return start, expected
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(5, 1 + attempt))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel exact range downloader")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bytes", required=True, type=int)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--part-mib", type=int, default=64)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.is_file():
        digest = sha256_file(output)
        if output.stat().st_size == args.bytes and digest == args.sha256.upper():
            print(f"already complete: {output} {digest}")
            return 0
        raise RuntimeError(f"existing output has unexpected size/hash: {output}")
    parts_root = output.with_suffix(output.suffix + ".parts")
    part_size = args.part_mib * 1024 * 1024
    ranges = [
        (start, min(args.bytes - 1, start + part_size - 1))
        for start in range(0, args.bytes, part_size)
    ]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_range,
                url=args.url,
                part_path=parts_root / f"{index:05d}.part",
                start=start,
                end=end,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            elapsed = max(time.monotonic() - started, 1e-6)
            downloaded = sum(
                path.stat().st_size for path in parts_root.glob("*.part")
            )
            print(
                f"parts={completed}/{len(ranges)} bytes={downloaded}/{args.bytes} "
                f"MiB/s={downloaded / elapsed / 1024**2:.3f}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for index in range(len(ranges)):
            with (parts_root / f"{index:05d}.part").open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(chunk)
    if temporary.stat().st_size != args.bytes:
        raise RuntimeError(f"assembled size mismatch: {temporary.stat().st_size}")
    digest = sha256_file(temporary)
    if digest != args.sha256.upper():
        raise RuntimeError(f"assembled SHA-256 mismatch: {digest}")
    os.replace(temporary, output)
    print(f"complete: {output} bytes={args.bytes} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
