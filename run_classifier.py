import argparse
from e2e_ml import model


def run_classifier(data_dir: str) -> None:
    model.train_classifier(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate classifier")
    parser.add_argument("--dir", default="data", help="Directory with processed images")
    args = parser.parse_args()
    run_classifier(args.dir)
