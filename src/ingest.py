# src/ingest.py
import os
import logging
from typing import List, Dict

logger = logging.getLogger("ingest")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# set of file extensions to include; adjust as needed
DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".kt", ".go", ".rs", ".c", ".cpp", ".h", ".cs",
    ".json", ".yaml", ".yml", ".md", ".xml", ".html", ".css", ".gradle", ".pom"
}

def _should_include(path: str, extensions=DEFAULT_EXTENSIONS) -> bool:
    # include by extension or include everything if extension set empty
    ext = os.path.splitext(path)[1].lower()
    return (not extensions) or (ext in extensions)

def load_codebase(root_path: str, max_files: int = 0) -> List[Dict[str, str]]:
    """
    Walk `root_path` and return list of {'path': relative_path, 'content': text}.
    If a file cannot be read as UTF-8, try latin-1. If still fails, log and skip.
    max_files: 0 = no limit
    """
    docs = []
    root_path = os.path.abspath(root_path)
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        # optionally skip hidden dirs (like .git)
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root_path)

            # filter by extension
            if not _should_include(full):
                continue

            try:
                # attempt utf-8 first
                with open(full, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except UnicodeDecodeError:
                # fallback to latin-1 (best-effort); log the fallback
                try:
                    with open(full, "r", encoding="latin-1") as fh:
                        text = fh.read()
                    logger.warning("File read with latin-1 fallback: %s", rel)
                except Exception as e:
                    logger.error("Fail to read source file %r (encoding fallback failed): %s", rel, e)
                    continue
            except FileNotFoundError as e:
                logger.error("Fail to read source file %r: file not found: %s", rel, e)
                continue
            except PermissionError as e:
                logger.error("Fail to read source file %r: permission error: %s", rel, e)
                continue
            except Exception as e:
                logger.error("Fail to read source file %r: unexpected error: %s", rel, e)
                continue

            docs.append({"path": rel, "content": text})

            file_count += 1
            if max_files and file_count >= max_files:
                logger.info("Reached max_files=%s; stopping file read.", max_files)
                return docs

    return docs


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Quick read test for ingest.load_codebase")
    parser.add_argument("repo", help="Path to repo (inside container)", nargs="?", default="/app/SakilaProject")
    args = parser.parse_args()
    docs = load_codebase(args.repo)
    print(f"Discovered {len(docs)} code documents.")
    for d in docs[:10]:
        print("-", d["path"])
