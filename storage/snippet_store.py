# storage/snippet_store.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

_DATA_SUFFIX = ".sigmf-data"
_META_SUFFIX = ".sigmf-meta"


class SnippetStore(Protocol):
    def adopt(self, staged_data_path: str) -> str:
        """Take ownership of a staged SigMF pair; return its final .sigmf-data path."""
        ...


class LocalSnippetStore:
    """Local flat-file IQ snippet storage (design doc section 7), swappable for
    an S3-compatible store behind the SnippetStore protocol.

    Capture processes write SigMF pairs into a staging directory; ingest --
    the only writer to storage -- adopts them here before persisting the
    record that references them. The staged path comes off the ingest
    socket, so it is treated as untrusted: it must resolve (symlinks
    included) to a .sigmf-data file directly inside the staging directory.
    """

    def __init__(self, staging_dir: Path | str, root_dir: Path | str) -> None:
        self._staging_dir = Path(staging_dir).resolve()
        self._root_dir = Path(root_dir).resolve()

    def adopt(self, staged_data_path: str) -> str:
        data = Path(staged_data_path).resolve()
        if data.parent != self._staging_dir:
            raise ValueError(
                f"Snippet {staged_data_path!r} is not inside the staging "
                f"directory {self._staging_dir}"
            )
        if data.suffix != _DATA_SUFFIX:
            raise ValueError(f"Snippet {staged_data_path!r} is not a {_DATA_SUFFIX} file")
        meta = data.with_suffix(_META_SUFFIX)
        for staged in (data, meta):
            if not staged.is_file():
                raise FileNotFoundError(f"Staged snippet file missing: {staged}")
        final_data = self._root_dir / data.name
        final_meta = self._root_dir / meta.name
        for final in (final_data, final_meta):
            if final.exists():
                raise FileExistsError(f"Snippet already stored: {final}")
        self._root_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(data, final_data)
        # Meta last: a .sigmf-meta in the store marks a complete pair.
        shutil.move(meta, final_meta)
        return str(final_data)
