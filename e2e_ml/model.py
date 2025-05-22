import os
import numpy as np
from skimage import io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def _load_images(folder: str):
    data = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder, fname)
            img = io.imread(path)
            data.append(img.reshape(-1))
    return np.array(data)


def _create_dataset(data_dir: str, labels: list):
    X = []
    y = []
    for idx, label in enumerate(labels):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        imgs = _load_images(folder)
        X.append(imgs)
        y.extend([idx] * len(imgs))
    if X:
        X = np.vstack(X)
    else:
        X = np.empty((0,))
    y = np.array(y)
    return X, y


def train_model(data_dir: str, labels: list, model_path: str = "model.joblib"):
    X, y = _create_dataset(data_dir, labels)
    if len(y) == 0:
        raise ValueError("No training data found")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    preds = clf.predict(X)
    return accuracy_score(y, preds)


def predict_folder(model_path: str, folder: str, labels: list):
    clf = joblib.load(model_path)
    X = _load_images(folder)
    results = {}
    if X.size == 0:
        return results
    preds = clf.predict(X)
    for fname, label_idx in zip(sorted(os.listdir(folder)), preds):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            results[fname] = labels[label_idx]
from typing import List
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
import joblib


def _load_images(base_dir: str, labels: List[str]):
    X = []
    y = []
    for idx, label in enumerate(labels):
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                path = os.path.join(folder, fname)
                img = Image.open(path).convert("L").resize((64, 64))
                X.append(np.array(img).flatten())
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


def train_classifier(data_dir: str, labels: List[str], model_path: str | None = None):
    X, y = _load_images(data_dir, labels)
    clf = LogisticRegression(max_iter=1000)
    if len(X) == 0:
        raise ValueError("No training data found")
    clf.fit(X, y)
    if model_path:
        joblib.dump(clf, model_path)
    return clf


def evaluate_classifier(clf: LogisticRegression, data_dir: str, labels: List[str]):
    X_test, y_test = _load_images(data_dir, labels)
    if len(X_test) == 0:
        raise ValueError("No test data found")
    return clf.score(X_test, y_test)


def predict_folder(model_path: str, folder: str, labels: List[str]):
    clf: LogisticRegression = joblib.load(model_path)
    results = {}
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img = Image.open(os.path.join(folder, fname)).convert("L").resize((64, 64))
            x = np.array(img).flatten().reshape(1, -1)
            pred = clf.predict(x)[0]
            results[fname] = labels[pred]
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
