"""
Face identity classifier.
No labels required — clusters faces with KMeans (eigenfaces + PCA),
then trains an SVM classifier on discovered identities.

Dependencies: Pillow, numpy, scikit-learn
"""

import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ── Config ────────────────────────────────────────────────────────────────────
FACES_DIR = Path("faces")
IMG_SIZE = (64, 64)  # resize all faces to this
N_COMPONENTS = 64  # PCA components (eigenfaces)
K_RANGE = range(2, 20)  # candidate identity counts to evaluate
MODEL_OUT = "face_classifier.pkl"
# ─────────────────────────────────────────────────────────────────────────────


def load_images(folder: Path) -> tuple[np.ndarray, list[str]]:
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    vectors, names = [], []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32).ravel() / 255.0
        vectors.append(arr)
        names.append(p.name)

    return np.stack(vectors), names


def find_best_k(X: np.ndarray, k_range) -> int:
    """Pick k that maximises silhouette score."""
    best_k, best_score = k_range.start, -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"  k={k:3d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def main():
    # 1. Load
    print(f"Loading images from '{FACES_DIR}' ...")
    X_raw, names = load_images(FACES_DIR)
    print(f"  {len(names)} images, {X_raw.shape[1]}-dim feature vectors")

    # 2. PCA (eigenfaces)
    n_components = min(N_COMPONENTS, len(names) - 1, X_raw.shape[1])
    print(f"\nRunning PCA (n_components={n_components}) ...")
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X = pca.fit_transform(X_raw)
    X = normalize(X)  # unit-norm before clustering
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained:.1%}")

    # 3. Find best number of identities
    max_k = min(K_RANGE.stop, len(names) // 2)
    k_range = range(K_RANGE.start, max_k)
    print(f"\nSearching for best k in {list(k_range)} ...")
    best_k = find_best_k(X, k_range)
    print(f"\n  → Best k = {best_k}")

    # 4. Final clustering
    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    cluster_labels = km.fit_predict(X)

    # 5. Train SVM classifier on cluster labels
    print("\nTraining SVM classifier on discovered identities ...")
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
    clf.fit(X, cluster_labels)
    train_acc = (clf.predict(X) == cluster_labels).mean()
    print(f"  Training accuracy: {train_acc:.1%}  (on cluster-assigned labels)")

    # 6. Copy images into per-identity folders
    import shutil
    from collections import defaultdict
    out_root = FACES_DIR.parent / "faces_by_identity"
    if out_root.exists():
        shutil.rmtree(out_root)
    groups = defaultdict(list)
    for name, label in zip(names, cluster_labels):
        groups[label].append(name)
    for label in sorted(groups):
        dest = out_root / f"identity_{label:02d}"
        dest.mkdir(parents=True)
        for fname in groups[label]:
            shutil.copy(FACES_DIR / fname, dest / fname)

    print(f"\n{'─'*50}")
    print(f"Discovered {best_k} identities → '{out_root}'\n")
    for label in sorted(groups):
        files = groups[label]
        preview = ", ".join(files[:5]) + ("..." if len(files) > 5 else "")
        print(f"  identity_{label:02d}/ ({len(files):3d} images): {preview}")

    # 7. Save model
    model = {"pca": pca, "clf": clf, "img_size": IMG_SIZE, "n_identities": best_k}
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to '{MODEL_OUT}'")


def predict(image_path: str) -> int:
    """Load saved model and predict identity for a new image."""
    with open(MODEL_OUT, "rb") as f:
        model = pickle.load(f)
    pca, clf, img_size = model["pca"], model["clf"], model["img_size"]

    img = Image.open(image_path).convert("RGB").resize(img_size)
    x = np.asarray(img, dtype=np.float32).ravel() / 255.0
    x = normalize(pca.transform(x.reshape(1, -1)))
    label = clf.predict(x)[0]
    probs = clf.predict_proba(x)[0]
    print(f"Identity: {label}  (confidence: {probs[label]:.1%})")
    return label


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        predict(sys.argv[1])
    else:
        main()
