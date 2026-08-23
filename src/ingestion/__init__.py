from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..swarm.models import Document
from .docx import read_docx
from .pdf import read_pdf
from .xlsx import read_xlsx
from .pptx import read_pptx
from .text import read_text
from .eml import read_eml

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".json",
    ".docx", ".xlsx", ".pptx",
    ".pdf", ".eml",
}

_READERS = {
    ".docx": read_docx,
    ".xlsx": read_xlsx,
    ".pptx": read_pptx,
    ".pdf": read_pdf,
    ".eml": read_eml,
}

LAZY_THRESHOLD = 50


def ingest_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    name = path.name
    size_bytes = path.stat().st_size
    doc_id = f"doc_{hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8]}"

    reader = _READERS.get(suffix, read_text)
    text, structured = reader(path)

    return Document(
        id=doc_id,
        name=name,
        text=text,
        size_bytes=size_bytes,
        metadata={"extension": suffix, "path": str(path)},
        structured=structured,
    )


def _make_lazy_doc(path: Path) -> Document:
    suffix = path.suffix.lower()
    name = path.name
    size_bytes = path.stat().st_size
    doc_id = f"doc_{hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8]}"
    reader = _READERS.get(suffix, read_text)

    def _load():
        return reader(path)

    return Document(
        id=doc_id,
        name=name,
        size_bytes=size_bytes,
        metadata={"extension": suffix, "path": str(path)},
        _loader=_load,
    )


def ingest_directory(directory: Path, *, lazy: bool = False) -> list[Document]:
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not lazy and len(paths) <= LAZY_THRESHOLD:
        return [ingest_file(p) for p in paths]
    return [_make_lazy_doc(p) for p in paths]


def discover_documents(task_dir: Path, docs_path: str | None = None) -> list[Document]:
    if docs_path:
        raw = Path(docs_path)
        # Relative paths are relative to the task directory, not the CWD.
        ext = raw.resolve() if raw.is_absolute() else (task_dir / raw).resolve()
        if ext.is_dir():
            return ingest_directory(ext)
        if ext.is_file():
            return [ingest_file(ext)]
    for subdir_name in ("source_documents", "input_documents", "documents", "docs"):
        subdir = task_dir / subdir_name
        if subdir.is_dir():
            return ingest_directory(subdir)
    return [
        ingest_file(p)
        for p in sorted(task_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.name not in ("task.json", "instruction.md", "scores.json", "status.json")
    ]
