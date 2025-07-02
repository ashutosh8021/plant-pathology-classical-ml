from pathlib import Path
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from tqdm import tqdm
from config import IMG_SIZE

# ── individual feature helpers ──────────────────────────────────────────
def _resize(img):                       # keep private; called inside extract_features
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def colour_hist(img, bins=(8, 8, 8)):
    """3-D RGB histogram → 512-D."""
    hist = cv2.calcHist([img], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def texture_lbp(img, P=8, R=1.0):
    """Uniform Local Binary Pattern histogram – 10 bins."""
    gray = rgb2gray(img)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, P + 3),
                           range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def shape_hu(img):
    """Hu image moments – 7-D."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()

# ── composite extractor ────────────────────────────────────────────────
def extract_features(image_paths):
    """
    image_paths : iterable[str | Path]
    returns     : np.ndarray shape (n_samples, 529)
    """
    feats = []
    for p in tqdm(image_paths, desc="Extracting"):
        img = cv2.imread(str(p))
        img = _resize(img)
        vec = np.hstack([
            colour_hist(img),     # 512
            texture_lbp(img),     # 10
            shape_hu(img)         # 7
        ])
        feats.append(vec)
    return np.vstack(feats)
