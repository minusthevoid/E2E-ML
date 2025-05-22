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
    return results
