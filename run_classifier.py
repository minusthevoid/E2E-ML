import argparse
import os
from e2e_ml import download, preprocess, augment
from e2e_ml import model


def prepare_data(base_dir: str, labels: list, num_images: int, augment_data: bool):
    search = ",".join(labels)
    download.download_images(search, num_images, base_dir)
    for label in labels:
        folder = os.path.join(base_dir, label)
from e2e_ml import model


def run_classifier(data_dir: str) -> None:
    model.train_classifier(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate classifier")
    parser.add_argument("--dir", default="data", help="Directory with processed images")
    args = parser.parse_args()
    run_classifier(args.dir)
import os
from e2e_ml import download, preprocess, augment, model


def prepare_data(base_dir: str, terms: list, num_images: int, augment_data: bool = True):
    search = ",".join(terms)
    download.download_images(search, num_images, base_dir)
    for term in terms:
        folder = os.path.join(base_dir, term)
        preprocess.resize_images(folder)
        if augment_data:
            augment.augment_images(folder)


def run_pipeline_and_classify(base_dir: str, labels: list, train_num: int, test_num: int):
def run_pipeline(train_terms: list, test_terms: list, num_train: int, num_test: int, base_dir: str):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    prepare_data(train_dir, labels, train_num, augment_data=True)
    clf = model.train_classifier(train_dir, labels, os.path.join(base_dir, "model.joblib"))

    prepare_data(test_dir, labels, test_num, augment_data=False)
    acc = model.evaluate_classifier(clf, test_dir, labels)
    print(f"Test accuracy: {acc:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline and classifier")
    parser.add_argument("--train_num", type=int, default=10, help="Images per class for training")
    parser.add_argument("--test_num", type=int, default=5, help="Images per class for testing")
    parser.add_argument("--dir", default="data", help="Base directory for data")
    parser.add_argument("--labels", default="cats,dogs", help="Comma separated class labels")
    args = parser.parse_args()
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    run_pipeline_and_classify(args.dir, labels, args.train_num, args.test_num)
    prepare_data(train_dir, train_terms, num_train, augment_data=True)
    acc = model.train_model(train_dir, train_terms, model_path=os.path.join(base_dir, "model.joblib"))
    print(f"Validation accuracy: {acc:.2f}")

    prepare_data(test_dir, test_terms, num_test, augment_data=False)
    for term in test_terms:
        folder = os.path.join(test_dir, term)
        results = model.predict_folder(os.path.join(base_dir, "model.joblib"), folder, train_terms)
        for fname, label in results.items():
            print(f"{fname}: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate image classifier")
    parser.add_argument("--dir", default="data", help="Base directory for data")
    parser.add_argument("--train_num", type=int, default=10, help="Images per class for training")
    parser.add_argument("--test_num", type=int, default=5, help="Images per class for testing")
    args = parser.parse_args()
    train_terms = ["cats", "dogs"]
    run_pipeline(train_terms, train_terms, args.train_num, args.test_num, args.dir)
