# src/ingest.py (patched)
import os
from typing import List, Dict
from tqdm import tqdm

# default set of file extensions we consider "code / text"
DEFAULT_EXTENSIONS = {
    ".java", ".py", ".md", ".xml", ".sql", ".js", ".ts", ".json", ".yaml", ".yml",
    ".html", ".css", ".gradle", ".properties", ".sh", ".kt"
}

# Maximum file size to read (bytes). Skip files larger than this to avoid OOM.
MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_BYTES", str(150 * 1024)))  # default 150 KB

# Folder name patterns to skip (generated artifacts, VCS, large deps)
SKIP_DIR_PATTERNS = [os.sep + ".mvn" + os.sep, os.sep + ".git" + os.sep, os.sep + "node_modules" + os.sep, os.sep + "target" + os.sep]

def is_probably_binary(path: str, num_bytes: int = 1024) -> bool:
    """
    Quick check: read the first `num_bytes` and see if we find null bytes.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(num_bytes)
            if b"\x00" in chunk:
                return True
    except Exception:
        return True
    return False

def read_code_files(base_path: str, extensions: set = DEFAULT_EXTENSIONS) -> List[str]:
    """
    Walk the repo and return a list of file paths whose extensions are in `extensions`.
    Skips large files, binary files and paths that match SKIP_DIR_PATTERNS.
    Returns paths in deterministic order.
    """
    base_path = os.path.abspath(base_path)
    files = []
    for root, _, filenames in os.walk(base_path):
        # skip directories that match the skip patterns
        if any(pat in root for pat in SKIP_DIR_PATTERNS):
            continue
        for fn in filenames:
            fp = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in extensions:
                # skip huge files early
                try:
                    size = os.path.getsize(fp)
                except Exception:
                    continue
                if size > MAX_FILE_BYTES:
                    # skip very large files
                    # (these are likely binaries, generated code, or huge single-file resources)
                    continue
                # quick binary check
                if is_probably_binary(fp):
                    continue
                files.append(fp)
    files.sort()
    return files

def load_codebase(base_path: str, extensions: set = DEFAULT_EXTENSIONS) -> List[Dict]:
    """
    Read files discovered by read_code_files and return a list of dicts:
      [{ "path": <path>, "content": <text> }, ...]
    Ignores unreadable files and returns successfully-read ones.
    """
    files = read_code_files(base_path, extensions=extensions)
    code_docs = []
    for path in tqdm(files, desc="Reading files", unit="file"):
        try:
            with open(path, "r", errors="ignore") as f:
                content = f.read()
                if not content or content.strip() == "":
                    continue
                code_docs.append({"path": path, "content": content})
        except Exception as e:
            # skip files we can't read
            print(f"[ingest] warning: failed to read {path}: {e}")
            continue
    return code_docs

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Quick read test for ingest.load_codebase")
    parser.add_argument("repo", help="Path to repo (inside container)", nargs="?", default="/app/SakilaProject")
    args = parser.parse_args()
    docs = load_codebase(args.repo)
    print(f"Discovered {len(docs)} code documents.")
    for d in docs[:10]:
        print("-", d["path"])
