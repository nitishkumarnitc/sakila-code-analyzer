# src/downloader.py
import os
import subprocess
from urllib.parse import urlparse

def repo_dir(workspace: str, repo_name: str):
    return os.path.join(workspace, repo_name)

def clone_or_update(git_url: str, workspace: str, repo_name: str = None):
    if repo_name is None:
        repo_name = os.path.splitext(os.path.basename(urlparse(git_url).path))[0]
    dst = repo_dir(workspace, repo_name)
    if os.path.exists(dst):
        # pull latest
        subprocess.check_call(["git", "-C", dst, "pull"])
    else:
        subprocess.check_call(["git", "clone", git_url, dst])
    return dst
