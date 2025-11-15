#!/usr/bin/env python3
"""
validate_report.py

Validates and enriches a lead-level JSON report produced by the analyzer.

Usage:
  python validate_report.py --report /path/to/extracted_knowledge_lead.json --repo /path/to/repo

Output:
  Writes a new JSON next to input named <original>_validated.json with added fields:
    - key_modules[].key_methods[].evidence_snippet
    - key_modules[].validated (bool)
    - key_modules[].key_methods[].computed_loc
    - key_modules[].key_methods[].computed_cyclomatic
    - key_modules[].key_methods[].confidence_adjusted
    - overall 'validated_at' timestamp
"""
import argparse
import json
from pathlib import Path
import traceback
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

# Try to import lizard for accurate cyclomatic metrics; fallback if not installed
try:
    import lizard
    HAS_LIZARD = True
except Exception:
    HAS_LIZARD = False

def read_file_lines(repo_root: Path, file_path: str) -> Optional[list]:
    p = Path(file_path)
    if not p.is_absolute():
        p = repo_root / file_path
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        return text.splitlines()
    except Exception:
        return None

def slice_lines(lines: list, start: int, end: int) -> str:
    # start and end are 1-based inclusive in the report
    if start < 1:
        start = 1
    if end < start:
        return ""
    # list indices 0-based
    start_idx = start - 1
    end_idx = min(len(lines), end)
    return "\n".join(lines[start_idx:end_idx])

def heuristic_cyclomatic(snippet: str) -> int:
    tokens = [" if ", " for ", " while ", " case ", "&&", "||", "elif ", "else if", "catch ", "?:", "switch"]
    s = snippet.lower()
    count = 1
    for t in tokens:
        count += s.count(t)
    return max(1, count)

def compute_cyclomatic_with_lizard(file_path: Path, target_start_line: int, target_end_line: int) -> Optional[int]:
    """
    Use lizard to analyze file and find the function covering target_start_line.
    Return cyclomatic complexity if found, else None.
    """
    try:
        analysis = lizard.analyze_file(str(file_path))
        for fn in analysis.function_list:
            if fn.start_line <= target_start_line <= fn.end_line or (fn.start_line <= target_end_line <= fn.end_line):
                return fn.cyclomatic_complexity or None
    except Exception:
        return None
    return None

def adjust_confidence(orig_conf: float, validated: bool, cyclo: Optional[int]) -> float:
    """Simple rule to adjust confidence up/down based on validation & complexity extremes."""
    conf = float(orig_conf) if orig_conf is not None else 0.5
    if not validated:
        conf = min(conf, 0.6)  # lower confidence if couldn't validate
    if cyclo is not None:
        if cyclo >= 15:
            conf = min(conf, 0.6)
        elif cyclo <= 3:
            conf = min(1.0, conf + 0.05)
    return round(conf, 3)

def validate_report(report_path: Path, repo_root: Path, out_path: Optional[Path] = None):
    if out_path is None:
        out_path = report_path.with_name(report_path.stem + "_validated.json")

    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ensure key structure
    modules = data.get("key_modules", [])
    repo_root = repo_root.resolve()

    # Preload lizard analysis per-file if available for speed
    lizard_cache: Dict[str, Any] = {}
    if HAS_LIZARD:
        # we'll lazily call lizard.analyze_file when needed
        pass

    summary_stats = {"modules_validated": 0, "methods_validated": 0, "methods_total": 0, "warnings": []}

    for mod in modules:
        mod_paths = mod.get("paths") or []
        mod_valid = False
        mod_evidence_snippet = None

        # Validate per path: if multiple paths, pick first that exists
        chosen_path = None
        for p in mod_paths:
            flines = read_file_lines(repo_root, p)
            if flines is not None:
                chosen_path = p
                break

        if chosen_path is None:
            # fallback: try path in mod['path'] if exists
            p0 = mod.get("path")
            if p0:
                flines = read_file_lines(repo_root, p0)
                if flines is not None:
                    chosen_path = p0
                else:
                    flines = None
            else:
                flines = None

        # If module has lines field, extract module snippet
        lines_info = mod.get("lines", {})
        try:
            start_line = int(lines_info.get("start", 1)) if lines_info else 1
            end_line = int(lines_info.get("end", start_line + 50)) if lines_info else min(200, len(flines) if flines else 0)
        except Exception:
            start_line, end_line = 1, (len(flines) if flines else 0)

        if flines is not None:
            mod_evidence_snippet = slice_lines(flines, start_line, end_line)
            mod["example_snippet_actual"] = mod_evidence_snippet
        else:
            mod["example_snippet_actual"] = None

        # Validate each key method
        km = mod.get("key_methods", [])
        for m in km:
            summary_stats["methods_total"] += 1
            src = m.get("source_ref") or {}
            spath = src.get("path") or chosen_path
            sstart = int(src.get("start") or 0)
            send = int(src.get("end") or 0)

            method_valid = False
            evidence = None
            computed_loc = None
            computed_cyclo = None

            if spath:
                lines = read_file_lines(repo_root, spath)
                if lines is not None:
                    # sanitize start/end
                    if sstart <= 0:
                        sstart = 1
                    if send <= 0:
                        # attempt to find a reasonable end: +20 lines
                        send = min(len(lines), sstart + 200)
                    if sstart <= len(lines):
                        evidence = slice_lines(lines, sstart, send)
                        computed_loc = evidence.count("\n") + 1 if evidence else 0

                        # try lizard if available
                        try:
                            if HAS_LIZARD:
                                cp = compute_cyclomatic_with_lizard(Path(spath), sstart, send)
                                if cp is not None:
                                    computed_cyclo = cp
                                else:
                                    computed_cyclo = heuristic_cyclomatic(evidence or "")
                            else:
                                computed_cyclo = heuristic_cyclomatic(evidence or "")
                        except Exception:
                            computed_cyclo = heuristic_cyclomatic(evidence or "")

                        method_valid = True
                        summary_stats["methods_validated"] += 1
                else:
                    # file missing
                    method_valid = False
            else:
                method_valid = False

            # store evidence & computed metrics
            m["evidence_snippet"] = evidence
            m["computed_loc"] = int(computed_loc) if computed_loc is not None else None
            m["computed_cyclomatic"] = int(computed_cyclo) if computed_cyclo is not None else None

            # adjust confidence
            orig_conf = m.get("confidence", 0.5)
            m["confidence_adjusted"] = adjust_confidence(orig_conf, method_valid, computed_cyclo)

            m["_validated"] = method_valid

            if not method_valid:
                summary_stats["warnings"].append(f"Method {m.get('name')} could not be validated (path {spath})")

            # update module-level validity
            if method_valid:
                mod_valid = True

        mod["_validated"] = mod_valid
        if mod_valid:
            summary_stats["modules_validated"] += 1

    data["_validated_at"] = datetime.utcnow().isoformat() + "Z"
    data["_validation_summary"] = summary_stats

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote validated report to {out_path}")
    print("Summary:", summary_stats)
    if summary_stats["warnings"]:
        print("Warnings (first 10):")
        for w in summary_stats["warnings"][:10]:
            print(" -", w)

def main():
    parser = argparse.ArgumentParser(description="Validate and augment JSON report with code evidence.")
    parser.add_argument("--report", required=True, help="Path to the JSON report produced by analyzer")
    parser.add_argument("--repo", required=True, help="Path to the checked-out repo root")
    parser.add_argument("--out", required=False, help="Optional output path for validated JSON")
    args = parser.parse_args()

    report_path = Path(args.report)
    repo_root = Path(args.repo)
    out_path = Path(args.out) if args.out else None

    try:
        validate_report(report_path, repo_root, out_path)
    except Exception as e:
        print("ERROR while validating report:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
