# ml_model/train.py
# Unified trainer (ResNet18 / EfficientNet-B0 / HOG+SVM) with:
# - Letterbox (no-stretch) preprocessing for DL models
# - Best checkpoint saving + "last" checkpoint
# - Early stopping (patience on val acc)
# - Extra metrics (precision/recall/F1, confusion matrix) on val & test
# - AMP (mixed precision) + optional cosine LR schedule
#
# Usage:
#   python -m ml_model.train --model resnet18
#   python -m ml_model.train --model efficientnet_b0
#   python -m ml_model.train --model hog_svm

from __future__ import annotations
import argparse, random, time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import joblib

from ml_config import CLASSES, RESNET18
from diff2_dataset import load_items, stratified_split  # reuse metadata parsing
from models import MODEL_REGISTRY
from models.hog_svm import hog_extract, HOG_IMG_SIZE
from skimage.transform import resize as sk_resize


# ------------------ utils ------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------ Letterbox Dataset (DL models) ------------------
class LetterboxSquare:
    """Resize keeping aspect ratio then pad to (target x target)."""
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


class Diff2LetterboxDataset(Dataset):
    """
    Minimal dataset that applies letterbox + ToTensor + Normalize.
    Inputs are list of (path, class_idx).
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, items: List[Tuple[Path, int]], img_size: int, train: bool):
        self.items = items
        self.letterbox = LetterboxSquare(target=img_size, pad_color=(255, 255, 255))
        self.train = train

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("L").convert("RGB")  # diff images → grayscale→RGB
        img = self.letterbox(img)  # keep aspect ratio, pad to square

        # light augments only on train
        if self.train:
            # very light horizontal flip (diffs are sparse; avoid heavy augments)
            if random.random() < 0.20:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # HWC
        x = x.permute(2, 0, 1)  # CHW
        # normalize
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        return x, y


# ------------------ HOG helpers (for hog_svm) ------------------
def to_gray_uint8(p: Path, size_hw=HOG_IMG_SIZE) -> np.ndarray:
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.uint8)
    # resize to HOG size (H, W)
    if arr.shape != size_hw:
        arr = (sk_resize(arr, size_hw, preserve_range=True, anti_aliasing=True)).astype(np.uint8)
    return arr

def build_hog_dataset(items: List[Tuple[Path, int]]) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for p, y in items:
        g = to_gray_uint8(p, size_hw=HOG_IMG_SIZE)
        feats.append(hog_extract(g))
        labels.append(y)
    X = np.vstack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ------------------ Train/Eval (DL) ------------------
def train_epoch_torch(model, loader, optim, device, scaler: torch.cuda.amp.GradScaler | None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)

@torch.no_grad()
def evaluate_torch(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    all_y = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    all_p = np.concatenate(all_p) if all_p else np.array([], dtype=np.int64)
    acc = correct / max(1, total)
    # extra metrics
    if all_y.size:
        prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_p, labels=list(range(len(CLASSES.class_names))), average="macro", zero_division=0)
        cm = confusion_matrix(all_y, all_p, labels=list(range(len(CLASSES.class_names))))
    else:
        prec = rec = f1 = 0.0
        cm = np.zeros((len(CLASSES.class_names), len(CLASSES.class_names)), dtype=int)
    return (loss_sum / max(1, total), acc, prec, rec, f1, cm)


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser("Unified trainer + letterbox + callbacks")
    ap.add_argument("--model", type=str, default="resnet18",
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--epochs", type=int, default=RESNET18.epochs)
    ap.add_argument("--batch-size", type=int, default=RESNET18.batch_size)
    ap.add_argument("--lr", type=float, default=RESNET18.lr)
    ap.add_argument("--weight-decay", type=float, default=RESNET18.weight_decay)
    ap.add_argument("--num-workers", type=int, default=RESNET18.num_workers)
    ap.add_argument("--seed", type=int, default=RESNET18.seed)
    ap.add_argument("--feature-extract", action="store_true", default=RESNET18.feature_extract)
    ap.add_argument("--img-size", type=int, default=RESNET18.img_size)
    ap.add_argument("--out-dir", type=str, default=str(RESNET18.out_dir.parent))  # base outputs/
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--use-cosine", action="store_true", help="Use CosineAnnealingLR")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (AMP)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="max_norm for gradient clipping; 0 disables")
    args = ap.parse_args()

    set_seed(args.seed)

    images_dir = Path(RESNET18.images_dir)
    meta_tsv = Path(RESNET18.metadata_tsv)
    #images_dir = RESNET18.images_dir
    #meta_tsv   = RESNET18.metadata_tsv
    base_out   = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Load & split once
    items = load_items(meta_tsv, images_dir, CLASSES.class_map)
    tr, va, te = stratified_split(
        items,
        train=RESNET18.split_train,
        val=RESNET18.split_val,
        test=RESNET18.split_test,
        seed=args.seed
    )

    model_name = args.model.lower()

    # -------- HOG + SVM path --------
    if model_name in ("hog_svm",):
        out_dir = base_out / "hog_svm"
        out_dir.mkdir(parents=True, exist_ok=True)

        X_tr, y_tr = build_hog_dataset(tr)
        X_va, y_va = build_hog_dataset(va)
        X_te, y_te = build_hog_dataset(te)

        factory = MODEL_REGISTRY[model_name]
        clf = factory(num_classes=CLASSES.num_classes)
        clf.fit(X_tr, y_tr)

        print("=== Validation ===")
        yva = clf.predict(X_va)
        print(classification_report(y_va, yva, target_names=CLASSES.class_names, digits=4))

        print("=== Test ===")
        yte = clf.predict(X_te)
        print(classification_report(y_te, yte, target_names=CLASSES.class_names, digits=4))

        joblib.dump({
            "pipeline": clf,
            "class_names": CLASSES.class_names,
            "class_map": CLASSES.class_map,
        }, out_dir / "hog_svm_model.joblib")
        return

    # -------- Torch models (resnet18 / efficientnet_b0) with letterbox --------
    out_dir = base_out / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Diff2LetterboxDataset(tr, img_size=args.img_size, train=True)
    val_ds   = Diff2LetterboxDataset(va, img_size=args.img_size, train=False)
    test_ds  = Diff2LetterboxDataset(te, img_size=args.img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factory = MODEL_REGISTRY[model_name]
    model = factory(num_classes=CLASSES.num_classes, feature_extract=args.feature_extract).to(device)

    # Optimizer / Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_cosine else None

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # ----- Training loop with early stopping & best checkpoint -----
    best_val_acc = 0.0
    best_epoch = 0
    patience = args.early_stop_patience
    wait = 0

    best_path = out_dir / f"{model_name}_best.pt"
    last_path = out_dir / f"{model_name}_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train
        tr_loss, tr_acc = train_epoch_torch(model, train_loader, optimizer, device, scaler)

        # optional grad clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)

        if scheduler is not None:
            scheduler.step()

        # validate (with extra metrics)
        va_loss, va_acc, va_prec, va_rec, va_f1, va_cm = evaluate_torch(model, val_loader, device)
        dt = time.time() - t0

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} "
              f"(P {va_prec:.4f} R {va_rec:.4f} F1 {va_f1:.4f}) | {dt:.1f}s")

        # save last checkpoint each epoch
        torch.save({
            "model": model.state_dict(),
            "class_names": CLASSES.class_names,
            "class_map": CLASSES.class_map
        }, last_path)

        # best checkpoint on val acc
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "class_names": CLASSES.class_names,
                "class_map": CLASSES.class_map
            }, best_path)
            wait = 0
        else:
            wait += 1

        # simple early stopping
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val_acc={best_val_acc:.4f}).")
            break

    # ----- Final test with best -----
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    te_loss, te_acc, te_prec, te_rec, te_f1, te_cm = evaluate_torch(model, test_loader, device)
    print("\n=== TEST RESULTS ===")
    print(f"loss {te_loss:.4f} | acc {te_acc:.4f} | P {te_prec:.4f} R {te_rec:.4f} F1 {te_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(te_cm)

    # Save metrics
    with (out_dir / "metrics.txt").open("w") as f:
        f.write(f"best_val_acc={best_val_acc:.6f}\n")
        f.write(f"test_loss={te_loss:.6f}\n")
        f.write(f"test_acc={te_acc:.6f}\n")
        f.write(f"test_precision_macro={te_prec:.6f}\n")
        f.write(f"test_recall_macro={te_rec:.6f}\n")
        f.write(f"test_f1_macro={te_f1:.6f}\n")
        f.write("labels=" + ",".join(CLASSES.class_names) + "\n")
        f.write("confusion_matrix=\n")
        for row in te_cm:
            f.write(",".join(map(str, row)) + "\n")

if __name__ == "__main__":
    main()
