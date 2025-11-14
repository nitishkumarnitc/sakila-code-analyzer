import argparse
from extract import extract_knowledge

def main():
    parser = argparse.ArgumentParser(description="Analyze SakilaProject codebase")
    parser.add_argument("--repo", required=True, help="Path to codebase to analyze")
    parser.add_argument("--output", required=True, help="Path to save JSON output")
    args = parser.parse_args()
    extract_knowledge(args.repo, args.output)

if __name__ == "__main__":
    main()
