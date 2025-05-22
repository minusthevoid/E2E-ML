import os
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
    return results
