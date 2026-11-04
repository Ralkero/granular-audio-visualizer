"""Explicit bounded download for a local research benchmark; never called by the app."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"
SHA256 = "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
SIZE = 84025440
URL = f"https://huggingface.co/adefossez/HTDemucs/resolve/{REVISION}/955717e8.safetensors"


def verify(path):
    with Path(path).open("rb") as file:
        if (
            Path(path).stat().st_size != SIZE
            or hashlib.file_digest(file, "sha256").hexdigest() != SHA256
        ):
            raise ValueError("Pinned model size/checksum mismatch")


def main():
    target = Path(".model-cache/955717e8.safetensors")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(".partial")
        try:
            count = 0
            with urlopen(URL, timeout=60) as source, temp.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > SIZE:
                        raise ValueError("Model download exceeded pinned size")
                    output.write(chunk)
            verify(temp)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    verify(target)
    print(f"Verified {SIZE} bytes, SHA256 {SHA256}; research only: weights license unresolved")


if __name__ == "__main__":
    main()
