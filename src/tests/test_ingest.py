import os
import builtins
import logging
import io
import sys
from pathlib import Path
import pytest

import src.ingest as ingest


def write_text_file(path: Path, text: str, encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def write_bytes_file(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def test_extension_filtering_and_utf8_read(tmp_path):
    """
    Files with configured extensions are included; others are skipped.
    Also verifies that a normal utf-8 file is read properly.
    """
    repo = tmp_path / "repo1"
    # create files: py (should be included), txt (should be skipped)
    py_file = repo / "a.py"
    txt_file = repo / "note.txt"
    write_text_file(py_file, "print('hello')", encoding="utf-8")
    write_text_file(txt_file, "ignore me", encoding="utf-8")

    docs = ingest.load_codebase(str(repo))
    paths = {d["path"] for d in docs}
    assert any(p.endswith("a.py") for p in paths)
    assert not any(p.endswith("note.txt") for p in paths)


def test_latin1_fallback_emits_warning(tmp_path, caplog):
    """
    Create a file with bytes that will raise UnicodeDecodeError when read as utf-8,
    and ensure the code falls back to latin-1 and logs a warning.
    """
    repo = tmp_path / "repo_latin"
    latin_file = repo / "bad_encoding.py"
    # bytes that are invalid as UTF-8 (0x80 alone will trigger decode error)
    write_bytes_file(latin_file, b"\x80\x81\x82")

    caplog.set_level(logging.WARNING, logger="ingest")
    docs = ingest.load_codebase(str(repo))
    # file should be included and content present
    assert any(d["path"].endswith("bad_encoding.py") for d in docs)
    # should have emitted a latin-1 fallback warning
    found = any("latin-1 fallback" in rec.getMessage() for rec in caplog.records)
    assert found, "Expected latin-1 fallback warning in ingest logs"


def test_permission_and_unexpected_errors_are_logged(tmp_path, monkeypatch, caplog):
    """
    Simulate PermissionError when reading a file and ensure an error is logged and file is skipped.
    """
    repo = tmp_path / "repo_perm"
    target = repo / "noaccess.py"
    write_text_file(target, "print('x')", encoding="utf-8")

    # Wrap builtin open to raise PermissionError only for our target path
    real_open = builtins.open

    def fake_open(path, mode="r", *args, **kwargs):
        # path may be a Path object or str
        pstr = str(path)
        if pstr.endswith("noaccess.py") and "r" in mode:
            raise PermissionError("simulated permission denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    caplog.set_level(logging.ERROR, logger="ingest")
    docs = ingest.load_codebase(str(repo))
    # file should be skipped because of permission error
    assert not any(d["path"].endswith("noaccess.py") for d in docs)
    # ensure an error was logged mentioning the file
    found = any("noaccess.py" in rec.getMessage() and "permission" in rec.getMessage().lower() for rec in caplog.records)
    assert found, "Expected permission error log for noaccess.py"


def test_max_files_limit(tmp_path):
    """
    Create several valid files and ensure load_codebase respects max_files.
    """
    repo = tmp_path / "repo_many"
    # create 5 files with allowed extensions
    for i in range(5):
        p = repo / f"f{i}.py"
        write_text_file(p, f"print({i})", encoding="utf-8")

    docs = ingest.load_codebase(str(repo), max_files=2)
    assert len(docs) == 2


def test_skip_hidden_directories(tmp_path):
    """
    Ensure files under hidden directories (.git) are skipped.
    """
    repo = tmp_path / "repo_hidden"
    # create a normal file and a file under .git
    normal = repo / "ok.py"
    hidden = repo / ".git" / "internal.py"
    write_text_file(normal, "x=1", encoding="utf-8")
    write_text_file(hidden, "secret=1", encoding="utf-8")

    docs = ingest.load_codebase(str(repo))
    paths = [d["path"] for d in docs]
    assert any(p.endswith("ok.py") for p in paths)
    # hidden/internal.py should not be present
    assert not any(".git" in p for p in paths)
