from __future__ import annotations
import numpy as np
from typing import Tuple
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# HOG params can be exposed via config if you want
HOG_IMG_SIZE = (256, 256)      # (H, W)
HOG_PIX_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HOG_BLOCK_NORM = "L2-Hys"

def hog_extract(img_gray_uint8: np.ndarray) -> np.ndarray:
    # img_gray_uint8 shape (H, W), dtype uint8
    return hog(
        img_gray_uint8,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIX_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    ).astype(np.float32)

def build_hog_svm(num_classes: int, **kwargs) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced")),
    ])
