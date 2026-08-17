"""Copy Flask's browser assets into Vercel's CDN-served ``public`` directory."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "webapp" / "static"
TARGET = ROOT / "public" / "static"


def main() -> None:
    if not SOURCE.is_dir():
        raise RuntimeError(f"Missing Flask static directory: {SOURCE}")
    TARGET.mkdir(parents=True, exist_ok=True)
    for source_path in SOURCE.iterdir():
        if source_path.is_file():
            copy2(source_path, TARGET / source_path.name)
    print(f"Copied browser assets to {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
