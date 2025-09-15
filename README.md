## 📂 Project Structure

```

ml\_model/
│
├── models/
│   ├── resnet18.py          # ResNet-18 builder
│   ├── efficientnet\_b0.py   # EfficientNet-B0 builder
│   ├── hog\_svm.py           # HOG + SVM pipeline
│   └── **init**.py          # MODEL\_REGISTRY
│
├── diff2\_dataset.py         # Dataset + stratified split
├── ml\_config.py             # Global config (paths, hyperparams, classes)
├── train.py                 # Full trainer (DL + SVM)
├── smoke\_test.py            # Lightweight smoketest (tiny dataset)
└── **init**.py

````

---

## 🔧 Setup

```bash
# Create & activate venv (if not already done)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
````

**Dependencies:**

* Python 3.9+
* PyTorch + torchvision
* scikit-learn, scikit-image
* Pillow, numpy

---

## 📊 Dataset Format

Your dataset needs:

1. An **images folder** (e.g. `data/processed/ML_processed_image/`)
2. A **metadata file** (`metadata.csv` or `metadata.tsv`) with at least:

```csv
output_filename,labels
img1.jpg,C0
img2.jpg,C1
img3.jpg,C0
```

* `output_filename`: filename (relative to `--images-dir`, or absolute path)
* `labels`: class label (e.g. `C0`, `C1`, …)

Classes must also be listed in `ml_config.py`:

```python
class ClassConfig:
    class_names = ["C0", "C1", "C2", "C3"]
```

---

## 🚀 Running Training

From the **project root**, run:

### ResNet-18

```bash
python -m ml_model.train --model resnet18 --epochs 10
```

### EfficientNet-B0

```bash
python -m ml_model.train --model efficientnet_b0 --epochs 10
```

### HOG + SVM

```bash
python -m ml_model.train --model hog_svm
```

### Optional flags

* `--batch-size 32`
* `--lr 0.001`
* `--feature-extract` (freeze backbone, train only classifier head)
* `--use-cosine` (cosine LR schedule)
* `--amp` (mixed precision on GPU)
* `--grad-clip 1.0`
* `--images-dir /path/to/images`
* `--metadata-path /path/to/metadata.csv`

Outputs (checkpoints + metrics) are written to the `outputs/` folder.

---

## 🧪 Smoketest (tiny sanity run)

For quick pipeline verification with a few images:

```bash
python -m ml_model.smoke_test \
  --model resnet18 \
  --images-dir ml_model/models/dummy_training \
  --metadata-path ml_model/models/dummy_training/metadata.csv \
  --classes C0
```

* Loads a handful of images
* Forces all samples into train split
* Runs 1 epoch, prints loss/acc
* ✅ Confirms your pipeline runs end-to-end

By default, the smoketest does **not** save checkpoints.
To save, add at the end of `smoke_test.py`:

```python
torch.save(model.state_dict(), "smoke_model.pt")
```

---

## 📈 Outputs

* `outputs/{model_name}/best.pt` – best model checkpoint (DL models)
* `outputs/{model_name}/last.pt` – last checkpoint
* `outputs/{model_name}/metrics.txt` – test accuracy, precision, recall, F1, confusion matrix
* For HOG+SVM: saves a `joblib` pipeline under `outputs/hog_svm/`

---

## ⚡ Notes

* For real training, use dozens/hundreds of images per class.
* For debugging only, the smoketest can run with just 3 images.
* Update `ml_config.py` to change defaults like epochs, batch size, paths.

---
