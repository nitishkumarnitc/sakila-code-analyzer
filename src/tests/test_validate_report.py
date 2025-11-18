import json
from pathlib import Path
import os
import importlib
import types

import pytest

# Make sure to import the module from src (use PYTHONPATH=./src when running pytest)
import src.validate_report as vr


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str, encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def test_slice_lines_and_heuristic():
    lines = ["line1", "if cond:", "line3", "for i in range(2):", "end"]
    # slice_lines uses 1-based inclusive indices
    assert vr.slice_lines(lines, 1, 2) == "line1\nif cond:"
    assert vr.slice_lines(lines, 2, 4).startswith("if cond:")
    # heuristic_cyclomatic counts tokens (minimum 1)
    cc = vr.heuristic_cyclomatic("\n if x:\n for y in z:\n")
    assert cc >= 3


def test_validate_report_success(tmp_path, monkeypatch):
    """
    Create a small repo with a Python file containing a 'function' and a simple report
    that references that file and method. After validation, expect evidence_snippet,
    computed_loc, computed_cyclomatic and confidence_adjusted to be present and
    methods_validated == 1.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    # create a simple source file with many lines so slice_lines works
    src_file = repo / "module.py"
    src_content = "\n".join(
        [
            "def helper():",
            "    pass",
            "",
            "def getAllActors():",
            "    # example function",
            "    a = 1",
            "    if a:",
            "        a += 1",
            "    return a",
        ]
    )
    write_text(src_file, src_content)

    # Create a minimal "lead" report JSON referencing that function
    report = {
        "project_name": "TestProject",
        "key_modules": [
            {
                "module_name": "module",
                "paths": [str(src_file.relative_to(repo))],  # relative path
                "path": str(src_file.relative_to(repo)),
                "lines": {"start": 1, "end": 20},
                "key_methods": [
                    {
                        "name": "getAllActors",
                        "signature": "def getAllActors():",
                        "confidence": 0.8,
                        "source_ref": {"path": str(src_file.relative_to(repo)), "start": 4, "end": 9},
                    }
                ],
            }
        ],
    }

    report_path = tmp_path / "report.json"
    write_json(report_path, report)

    out_path = tmp_path / "report_validated.json"

    # Ensure lizard path: simulate lizard available and a known cyclomatic result
    monkeypatch.setattr(vr, "HAS_LIZARD", True)
    # make compute_cyclomatic_with_lizard return a fixed value (simulate lizard)
    monkeypatch.setattr(vr, "compute_cyclomatic_with_lizard", lambda fp, s, e: 5)

    # Run validation
    vr.validate_report(report_path=report_path, repo_root=repo, out_path=out_path)

    # Read output and assert fields
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert "_validation_summary" in out
    summary = out["_validation_summary"]
    assert summary["methods_total"] == 1
    assert summary["methods_validated"] == 1
    # Check the module has example_snippet_actual and method evidence
    km = out["key_modules"][0]
    assert km.get("example_snippet_actual") is not None
    meth = km["key_methods"][0]
    assert meth.get("evidence_snippet") is not None
    assert meth.get("computed_loc") is not None
    assert meth.get("computed_cyclomatic") == 5
    assert "confidence_adjusted" in meth
    assert meth["_validated"] is True


def test_validate_report_missing_file_logs_warning(tmp_path):
    """
    Report references a non-existent path -> validator should complete and include
    a warning in the _validation_summary and mark method as not validated.
    """
    repo = tmp_path / "repo2"
    repo.mkdir()

    # create report referencing a non-existent file path
    report = {
        "project_name": "X",
        "key_modules": [
            {
                "module_name": "m",
                "paths": ["nonexistent.py"],
                "key_methods": [
                    {
                        "name": "missingFn",
                        "confidence": 0.4,
                        "source_ref": {"path": "nonexistent.py", "start": 1, "end": 10},
                    }
                ],
            }
        ],
    }

    report_path = tmp_path / "report2.json"
    write_json(report_path, report)
    out_path = tmp_path / "report2_validated.json"

    vr.validate_report(report_path=report_path, repo_root=repo, out_path=out_path)
    out = json.loads(out_path.read_text(encoding="utf-8"))
    summary = out["_validation_summary"]
    assert summary["methods_total"] == 1
    assert summary["methods_validated"] == 0
    assert len(summary["warnings"]) >= 1
    assert "could not be validated" in summary["warnings"][0]


def test_read_file_lines_absolute_and_relative(tmp_path):
    """
    Ensure read_file_lines works with absolute and repo-relative paths.
    """
    repo = tmp_path / "repo3"
    repo.mkdir()
    f = repo / "a.py"
    write_text(f, "print('ok')")

    # absolute path
    lines_abs = vr.read_file_lines(repo, str(f.resolve()))
    assert lines_abs is not None and lines_abs[0].startswith("print")

    # relative path (repo root + relative)
    rel = str(f.relative_to(repo))
    lines_rel = vr.read_file_lines(repo, rel)
    assert lines_rel is not None and lines_rel[0].startswith("print")
