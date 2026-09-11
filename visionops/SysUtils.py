"""Small system and filesystem helpers."""

from __future__ import annotations

import fnmatch
import getpass
import os
from pathlib import Path
from typing import Iterable
import uuid


def get_username() -> str:
    """Return the current user name, including in containers and notebooks."""
    return getpass.getuser()


def get_machine_uid() -> str:
    """Return a stable identifier derived from the machine network address."""
    return str(uuid.UUID(int=uuid.getnode()))


def _as_strings(value: str | os.PathLike[str] | Iterable[str | os.PathLike[str]]) -> list[str]:
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    return [os.fspath(item) for item in value]


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def _matches(path: Path, root: Path, patterns: list[str], case_sensitive: bool) -> bool:
    relative = path.relative_to(root).as_posix()
    name = path.name
    if not case_sensitive:
        relative = relative.lower()
        name = name.lower()
        patterns = [pattern.lower() for pattern in patterns]
    return any(
        fnmatch.fnmatchcase(relative, pattern) or fnmatch.fnmatchcase(name, pattern)
        for pattern in patterns
    )


def search(
    search_in_dirs: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] = ".",
    patterns: str | Iterable[str] = "*",
    recursive: bool = True,
    path_type: str = "abs",
    search_for: str = "files",
    ignore_hidden: bool = False,
    case_sensitive_patterns: bool = False,
) -> list[str]:
    """Find filesystem paths matching one or more glob-style patterns.

    Args:
        search_in_dirs: One directory or an iterable of directories.
        patterns: One pattern or an iterable of patterns, such as ``*.jpg``.
        recursive: Search below each directory when true.
        path_type: Return absolute (``abs``), relative (``rel``), or basename
            (``base``) paths.
        search_for: Select ``files``, ``dirs``, or ``all`` entries.
        ignore_hidden: Exclude paths containing a dot-prefixed component.
        case_sensitive_patterns: Match patterns with exact case when true.

    Returns:
        A sorted list without duplicates. Missing input directories are ignored.
    """
    if path_type not in {"abs", "rel", "base"}:
        raise ValueError("path_type must be one of: 'abs', 'rel', 'base'")
    if search_for not in {"all", "files", "dirs"}:
        raise ValueError("search_for must be one of: 'all', 'files', 'dirs'")

    roots = [Path(path).expanduser() for path in _as_strings(search_in_dirs)]
    pattern_list = _as_strings(patterns)
    matches: set[str] = set()

    for root in roots:
        if not root.is_dir():
            continue
        candidates = root.rglob("*") if recursive else root.glob("*")
        for candidate in candidates:
            if ignore_hidden and _is_hidden(candidate, root):
                continue
            if search_for == "files" and not candidate.is_file():
                continue
            if search_for == "dirs" and not candidate.is_dir():
                continue
            if not _matches(candidate, root, pattern_list, case_sensitive_patterns):
                continue

            if path_type == "abs":
                rendered = str(candidate.resolve())
            elif path_type == "base":
                rendered = candidate.name
            else:
                rendered = os.path.relpath(candidate)
            matches.add(rendered)

    return sorted(matches)


def is_image(path: str | os.PathLike[str]) -> bool:
    """Return whether a path has a conventional raster-image extension."""
    return Path(path).suffix.lower() in {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
    }
