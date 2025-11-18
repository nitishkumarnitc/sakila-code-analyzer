# Writing a minimal unit test suite for src/downloader.py using Python's unittest and mocking.
# This will create src/downloader.py (from the user's provided content), then define & run tests.
# Tests cover:
#  - repo_dir returns correct path
#  - clone_or_update calls `git clone` when destination does not exist
#  - clone_or_update calls `git -C <dst> pull` when destination exists
#
# The tests use unittest.mock to patch `os.path.exists` and `subprocess.check_call` so no real git calls are made.

from pathlib import Path
import os

# Ensure src directory exists
Path("src").mkdir(exist_ok=True)

downloader_code = r'''import os
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
'''

# Write the downloader module to src/downloader.py
with open("src/downloader.py", "w", encoding="utf-8") as f:
    f.write(downloader_code)

# Now create unit tests using unittest
import unittest
from unittest import mock
import importlib

# Import the freshly written module
import src.downloader as downloader
importlib.reload(downloader)

class TestDownloader(unittest.TestCase):
    def test_repo_dir(self):
        self.assertEqual(downloader.repo_dir("/tmp/ws", "myrepo"), os.path.join("/tmp/ws", "myrepo"))

    @mock.patch("src.downloader.subprocess.check_call")
    @mock.patch("src.downloader.os.path.exists", return_value=False)
    def test_clone_called_when_dst_missing(self, mock_exists, mock_check_call):
        git_url = "https://github.com/user/repo.git"
        workspace = "/tmp/workspace"
        # ensure repo_name inferred from URL
        dst = downloader.clone_or_update(git_url, workspace, repo_name=None)
        expected_dst = os.path.join(workspace, "repo")
        self.assertEqual(dst, expected_dst)
        mock_check_call.assert_called_once_with(["git", "clone", git_url, expected_dst])

    @mock.patch("src.downloader.subprocess.check_call")
    @mock.patch("src.downloader.os.path.exists", return_value=True)
    def test_pull_called_when_dst_exists(self, mock_exists, mock_check_call):
        git_url = "https://github.com/user/repo.git"
        workspace = "/var/repos"
        dst = downloader.clone_or_update(git_url, workspace, repo_name=None)
        expected_dst = os.path.join(workspace, "repo")
        self.assertEqual(dst, expected_dst)
        mock_check_call.assert_called_once_with(["git", "-C", expected_dst, "pull"])

    def test_clone_with_explicit_repo_name(self):
        # when repo_name provided, use it instead of inferring
        with mock.patch("src.downloader.subprocess.check_call") as mock_check_call, \
             mock.patch("src.downloader.os.path.exists", return_value=False):
            git_url = "https://github.com/other/ignored.git"
            workspace = "/tmp/ws2"
            dst = downloader.clone_or_update(git_url, workspace, repo_name="custom-name")
            expected_dst = os.path.join(workspace, "custom-name")
            self.assertEqual(dst, expected_dst)
            mock_check_call.assert_called_once_with(["git", "clone", git_url, expected_dst])


# Run the tests and show output
suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDownloader)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
# Exit code style summary
print(f"Ran {result.testsRun} tests. Failures: {len(result.failures)}. Errors: {len(result.errors)}.")

