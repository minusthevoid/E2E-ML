import argparse
import os
from e2e_ml import download, preprocess, augment


def run_pipeline(search_terms: str, num_images: int, base_dir: str) -> None:
    download.download_images(search_terms, num_images, base_dir)
    terms = [term.strip() for term in search_terms.split(',') if term.strip()]
    for term in terms:
        folder = os.path.join(base_dir, term)
        preprocess.resize_images(folder)
        augment.augment_images(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end image pipeline")
    parser.add_argument("--search", required=True, help="Comma separated search terms")
    parser.add_argument("--num", type=int, default=10, help="Images per term")
    parser.add_argument("--dir", default="data", help="Output directory")
    args = parser.parse_args()
    run_pipeline(args.search, args.num, args.dir)
