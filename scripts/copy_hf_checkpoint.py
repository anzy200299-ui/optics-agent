#!/usr/bin/env python3
"""Copy an HF checkpoint file-by-file, skipping files that already match in size."""
import shutil
import sys
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    index = src / "model.safetensors.index.json"
    if not index.is_file():
        raise SystemExit(f"missing {index}")
    dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(p for p in src.iterdir() if p.is_file()):
        target = dst / src_file.name
        src_size = src_file.stat().st_size
        if target.is_file() and target.stat().st_size == src_size:
            print(f"keep {target.name} ({src_size} bytes)", flush=True)
            continue
        partial = dst / f"{src_file.name}.partial"
        print(f"copy {src_file.name} ({src_size} bytes)", flush=True)
        with src_file.open("rb") as src_handle, partial.open("wb") as dst_handle:
            shutil.copyfileobj(src_handle, dst_handle, length=8 * 1024 * 1024)
        got = partial.stat().st_size
        if got != src_size:
            raise SystemExit(f"size mismatch for {src_file.name}: {got} != {src_size}")
        partial.replace(target)
    if not (dst / "model.safetensors.index.json").is_file():
        raise SystemExit(f"index missing after copy: {dst}")
    print(f"copied {src} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
