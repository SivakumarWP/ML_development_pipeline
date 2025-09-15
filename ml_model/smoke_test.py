# smoke_test.py
# Minimal end-to-end smoketest for tiny datasets (1–5 images).
# Place this file in your ml_model/ directory and run:
#   python smoke_test.py --model resnet18 \
#     --images-dir /path/to/images \
#     --metadata-path /path/to/metadata.csv
#
# Notes:
# - Forces all items into the train split to avoid num_samples=0 errors.
# - Accepts CSV or TSV; flexible filename column names.
# - For DL models: trains for a few steps; don't expect meaningful accuracy.
# - For HOG+SVM: fits on the tiny set and prints a quick report if possible.

from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import Dict, List, Tuple
import csv, ast

import numpy as np
from PIL import Image, ImageOps

# --- Torch / torchvision (for DL paths) ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Skimage / sklearn (for HOG+SVM path) ---
from skimage.feature import hog
from skimage.transform import resize as sk_resize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# --- Import your model builders from local files in this folder ---
# Ensure these files exist in the same folder as this script:
#   resnet18.py, efficientnet_b0.py, hog_svm.py
from models.resnet18 import build_resnet18
from models.efficientnet_b0 import build_efficientnet_b0

try:
    from models.hog_svm import make_hog_svm as build_hog_svm
except ImportError:
    from models.hog_svm import build_hog_svm  # type: ignore


# ------------------ tiny utils ------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------ robust metadata parsing ------------------
def parse_labels_str(lbl_str: str) -> List[str]:
    """
    Accepts label strings like:
      - "['C0']" or '["C0"]'  (list-like)
      - "C0"                  (single token)
    Returns [label] or [] if un-parseable.
    """
    s = (lbl_str or "").strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and obj:
            return [str(obj[0])]
    except Exception:
        pass
    # fallback: treat as single class token
    return [s]


def load_items_robust(
    metadata_path: Path,
    images_dir: Path,
    class_map: Dict[str, int],
) -> List[Tuple[Path, int]]:
    """
    Robust loader that:
      - sniffs delimiter (csv/tsv/semicolon),
      - accepts flexible filename column names (case-insensitive),
      - accepts labels like "C0" or "['C0']"
      - supports absolute or relative file paths:
            * if value is absolute -> use as-is
            * if relative -> join with images_dir
      - ignores rows with missing files or unknown labels
    """
    metadata_path = Path(metadata_path)
    images_dir = Path(images_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not images_dir.exists():
        print(f"[WARN] Images directory not found: {images_dir} (will still try absolute paths in metadata)")

    # Sniff delimiter
    with metadata_path.open("r", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
            delim = dialect.delimiter
        except Exception:
            delim = "\t" if metadata_path.suffix.lower() == ".tsv" else ","
        reader = csv.DictReader(fh, delimiter=delim)

        # Case-insensitive header mapping
        headers = {h.lower(): h for h in (reader.fieldnames or [])}

        filename_keys_ci = ("output_filename","filename","file","image","img","path","image_path","filepath","file_path")
        label_keys_ci    = ("labels","label","class","category","class_label","target")

        def get_any(row, keys_ci):
            for k in keys_ci:
                if k in headers:
                    v = row.get(headers[k], "")
                    if v is not None and str(v).strip() != "":
                        return str(v).strip()
            return ""

        items: List[Tuple[Path, int]] = []
        dropped_missing_file = 0
        dropped_unknown_label = 0
        dropped_no_filename = 0
        dropped_no_label = 0

        for row in reader:
            # filename
            fn = get_any(row, filename_keys_ci)
            if not fn:
                dropped_no_filename += 1
                continue

            # Build image path: absolute -> use as-is; else join with images_dir
            candidate = Path(fn)
            if candidate.is_absolute():
                img_path = candidate
            else:
                img_path = (images_dir / fn)

            if not img_path.exists():
                # As a last resort, try resolving relative to the metadata file
                alt = metadata_path.parent / fn
                if alt.exists():
                    img_path = alt
                else:
                    dropped_missing_file += 1
                    continue

            # label(s)
            lbl_str = get_any(row, label_keys_ci)
            if not lbl_str:
                dropped_no_label += 1
                continue
            labels = parse_labels_str(lbl_str)
            if not labels:
                dropped_no_label += 1
                continue

            cname = labels[0]
            if cname not in class_map:
                dropped_unknown_label += 1
                continue

            items.append((img_path, class_map[cname]))

    if not items:
        # Helpful debug
        print("[DEBUG] No items created from metadata.")
        print(f"[DEBUG] metadata: {metadata_path}")
        print(f"[DEBUG] images_dir: {images_dir}")
        print(f"[DEBUG] delimiter guessed: {delim!r}")
        print(f"[DEBUG] available columns: {list(headers.keys())}")
        print(f"[DEBUG] dropped_no_filename: {dropped_no_filename}")
        print(f"[DEBUG] dropped_no_label: {dropped_no_label}")
        print(f"[DEBUG] dropped_missing_file: {dropped_missing_file}")
        print(f"[DEBUG] dropped_unknown_label (not in classes): {dropped_unknown_label}")
        print(f"[DEBUG] classes provided: {list(class_map.keys())}")

    return items


# ------------------ simple class config ------------------
class ClassConfig:
    def __init__(self, class_names: List[str] | None = None):
        # default to a single class "C0" for tiny smoketests
        self.class_names = class_names or ["C0"]
        self.class_map = {c: i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)


# ------------------ Letterbox Dataset (DL models) ------------------
class LetterboxSquare:
    """Resize keeping aspect ratio, then pad to (target x target)."""
    def __init__(self, target: int = 224, pad_color=(255, 255, 255)):
        self.target = int(target)
        self.pad_color = pad_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.target / max(w, h)
        new_w, new_h = int(round(w * s)), int(round(h * s))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        pad_l = (self.target - new_w) // 2
        pad_t = (self.target - new_h) // 2
        pad_r = self.target - new_w - pad_l
        pad_b = self.target - new_h - pad_t
        return ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=self.pad_color)


class TinyLetterboxDataset(Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, items: List[Tuple[Path, int]], img_size: int):
        self.items = items
        self.letterbox = LetterboxSquare(target=img_size, pad_color=(255, 255, 255))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        img = Image.open(p).convert("L").convert("RGB")
        img = self.letterbox(img)
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # HWC
        x = x.permute(2, 0, 1)  # CHW
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        return x, y


# ------------------ HOG helpers (for hog_svm) ------------------
HOG_IMG_SIZE = (256, 256)
def to_gray_uint8(p: Path, size_hw=HOG_IMG_SIZE) -> np.ndarray:
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != size_hw:
        arr = sk_resize(arr, size_hw, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return arr

def hog_extract(img_gray_uint8: np.ndarray) -> np.ndarray:
    return hog(
        img_gray_uint8,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


# ------------------ Training loops (DL) ------------------
def train_epoch_torch(model, loader, optim, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser("Smoketest trainer (tiny dataset)")
    ap.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18", "efficientnet_b0", "hog_svm"])
    ap.add_argument("--images-dir", type=str, required=True,
                    help="Folder containing images referenced by metadata")
    ap.add_argument("--metadata-path", type=str, required=True,
                    help="CSV/TSV with columns like output_filename,labels")
    ap.add_argument("--classes", type=str, nargs="+", default=["C0"],
                    help="Class names (default: C0). Ensure metadata labels match.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    images_dir = Path(args.images_dir)
    meta_path  = Path(args.metadata_path)

    CLASSES = ClassConfig(class_names=args.classes)

    # Load items (robust)
    items = load_items_robust(meta_path, images_dir, CLASSES.class_map)
    if not items:
        raise RuntimeError(
            "No items loaded. Check paths, metadata delimiter/columns, labels, and image filenames."
        )

    # Force ALL items into train to avoid empty splits in tiny tests
    tr = items
    print(f"[SMOKE] Loaded {len(items)} items. Using all for training.")

    model_name = args.model.lower()

    # -------------- HOG + SVM path --------------
    if model_name == "hog_svm":
        # Build HOG features
        feats, labels = [], []
        for p, y in tr:
            g = to_gray_uint8(p, size_hw=HOG_IMG_SIZE)
            feats.append(hog_extract(g))
            labels.append(y)
        X = np.vstack(feats).astype(np.float32)
        y = np.array(labels, dtype=np.int64)

        clf: Pipeline = build_hog_svm(num_classes=CLASSES.num_classes)
        clf.fit(X, y)

        # Tiny report (train-as-val, just to print something)
        y_pred = clf.predict(X)
        print("\n=== HOG+SVM (train set report) ===")
        print(classification_report(y, y_pred, target_names=CLASSES.class_names, digits=4))
        return

    # -------------- Torch models path --------------
    # Build dataset/loader
    train_ds = TinyLetterboxDataset(tr, img_size=args.img_size)
    if len(train_ds) == 0:
        raise RuntimeError("Empty training dataset after loading. Check metadata and paths.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    if model_name == "resnet18":
        model = build_resnet18(num_classes=CLASSES.num_classes, feature_extract=True).to(device)
    elif model_name == "efficientnet_b0":
        model = build_efficientnet_b0(num_classes=CLASSES.num_classes, feature_extract=True).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # Tiny training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch_torch(model, train_loader, optimizer, device)
        print(f"[{epoch:02d}/{args.epochs}] train loss={tr_loss:.4f} acc={tr_acc:.4f}")

    print("Smoketest complete ✅ (model trained on tiny set).")


if __name__ == "__main__":
    main()
