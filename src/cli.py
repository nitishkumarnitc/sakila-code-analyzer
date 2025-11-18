import argparse
from analyser import analyze_repo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Git URL or local path")
    parser.add_argument("--name", required=False, help="Repo local name (optional)")
    args = parser.parse_args()
    analyze_repo(args.repo, repo_name=args.name)

if __name__ == "__main__":
    main()
