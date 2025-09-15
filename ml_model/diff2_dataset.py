from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import csv, ast, random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def parse_labels_str(lbl_str: str) -> List[str]:
    try:
        obj = ast.literal_eval(lbl_str)
        if isinstance(obj, list) and obj:
            return [str(obj[0])]
    except Exception:
        pass
    return []

def load_items(metadata_tsv: Path, images_dir: Path, class_map: Dict[str, int]) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with metadata_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out_name = (row.get("output_filename") or "").strip()
            labels_str = (row.get("labels") or "").strip()
            if not out_name:
                continue
            img_path = images_dir / out_name
            if not img_path.exists():
                continue
            labels = parse_labels_str(labels_str)
            if not labels:
                continue
            cname = labels[0]
            if cname not in class_map:
                continue
            items.append((img_path, class_map[cname]))
    return items

def stratified_split(
    items: List[Tuple[Path, int]], train=0.7, val=0.15, test=0.15, seed: int = 42
):
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    rng = random.Random(seed)
    train_set: List[Tuple[Path, int]] = []
    val_set: List[Tuple[Path, int]] = []
    test_set: List[Tuple[Path, int]] = []
    for cls, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_tr = int(round(n * train))
        n_va = int(round(n * val))
        if n_tr + n_va > n:
            n_va = max(0, n - n_tr)
        n_te = n - n_tr - n_va
        train_set.extend(lst[:n_tr])
        val_set.extend(lst[n_tr:n_tr+n_va])
        test_set.extend(lst[n_tr+n_va:])
    rng.shuffle(train_set); rng.shuffle(val_set); rng.shuffle(test_set)
    return train_set, val_set, test_set

class Diff2Dataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], img_size: int, train: bool):
        self.items = items
        if train:
            self.tfm = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("L").convert("RGB")
        x = self.tfm(img)
        return x, y
