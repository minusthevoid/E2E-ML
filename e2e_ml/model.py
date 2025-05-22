import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _load_images(data_dir: str):
    X = []
    y = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.startswith("resized_") and fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                path = os.path.join(class_dir, fname)
                img = Image.open(path).convert("RGB")
                X.append(np.asarray(img).flatten())
                y.append(label)
    return np.array(X), np.array(y)


def train_classifier(data_dir: str):
    """Train a logistic regression classifier on images in ``data_dir``."""
    X, y = _load_images(data_dir)
    if X.size == 0:
        raise ValueError("No training images found. Run the pipeline first.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.2f}")
    return clf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

IMAGE_SIZE = (64, 64)


def load_dataset(base_dir: str, terms):
    """Load images for the given class terms and return arrays and labels."""
    X, y = [], []
    for idx, term in enumerate(terms):
        folder = os.path.join(base_dir, term)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) and (
                fname.startswith("resized_") or fname.startswith("augmented_")
            ):
                path = os.path.join(folder, fname)
                img = Image.open(path).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                X.append(np.asarray(img).flatten())
                y.append(idx)
    return np.array(X), np.array(y)


def train_model(base_dir: str, terms, model_path: str = "model.joblib"):
    """Train a simple classifier and save it to disk."""
    X, y = load_dataset(base_dir, terms)
    if len(X) == 0:
        raise ValueError("No training images found.")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    joblib.dump(clf, model_path)
    return acc


def predict_folder(model_path: str, folder: str, terms):
    """Predict class labels for all images in a folder."""
    clf = joblib.load(model_path)
    results = {}
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder, fname)
            img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
            arr = np.asarray(img).flatten().reshape(1, -1)
            label_idx = clf.predict(arr)[0]
            results[fname] = terms[label_idx]
    return results
