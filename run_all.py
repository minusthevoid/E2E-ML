import argparse
from run_pipeline import run_pipeline
from run_classifier import run_classifier


def run_all(search_terms: str, num_images: int, base_dir: str) -> None:
    run_pipeline(search_terms, num_images, base_dir)
    run_classifier(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline and classifier")
    parser.add_argument("--search", required=True, help="Comma separated search terms")
    parser.add_argument("--num", type=int, default=10, help="Images per term")
    parser.add_argument("--dir", default="data", help="Output directory")
    args = parser.parse_args()
    run_all(args.search, args.num, args.dir)
