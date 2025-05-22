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
