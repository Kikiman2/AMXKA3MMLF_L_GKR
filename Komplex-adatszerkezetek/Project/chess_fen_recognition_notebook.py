# Chess FEN Recognition — Jupyter-friendly script (jupytext / .py notebook style)
# Cells are separated with '# %%' so you can open this file directly in VSCode/Jupyter with jupytext
# Save as .py or convert to .ipynb with jupytext (or open in Colab after converting).

# %%
"""
Overview
- This notebook trains a per-square classifier to read FEN from board images whose filenames contain the FEN
  using hyphen '-' as rank separators (e.g. r7-8-1K3nR1-...png).
- Steps:
  1. Parse filenames -> FEN piece placement (replace '-' with '/')
  2. Detect and warp chessboard (OpenCV)
  3. Split into 64 crops and save pre-extracted squares
  4. Build PyTorch dataset and train ResNet18 (per-square classification: 13 classes)
  5. Evaluate and reconstruct FEN per image

Notes:
- This script is written in cells for execution in a notebook. Adjust paths and parameters at the top.
- Requirements: Python 3.8+, torch, torchvision, opencv-python, pillow, tqdm
"""

# %%
# CONFIGURATION
IMAGES_DIR = "./images"          # directory with your 80k images
OUT_EXTRACT_DIR = "./extracted"  # where per-square crops will be stored
CSV_INDEX = "image_index.csv"    # optional index: image_path,fen_piece_placement
SQUARE_SIZE = 128                 # saved crop size
WARPED_SIZE = 800                 # size to warp full board to (must be divisible by 8)
NUM_WORKERS = 8

# %%
# IMPORTS
import os
import re
import csv
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

# PyTorch imports (import later in training cell to allow cpu-only preprocessing)

# %%
# Helper: parse filename -> fen piece placement
def filename_to_fen(filename: str) -> str:
    """Convert filename containing hyphen-separated ranks into FEN piece placement.
    Example: r7-8-1K3nR1-p2P1NRq-8-6r1-5Pn1-1k2b3 -> r7/8/1K3nR1/...
    Strips extension and any leading path.
    """
    base = Path(filename).stem
    # Some filenames may include extra tokens (like suffixes). Find longest segment that looks like FEN ranks.
    # We assume the file's base is exactly the fen-like string. If not, user can adapt.
    fen = base.replace('-', '/')
    # validation: must have 8 ranks
    parts = fen.split('/')
    if len(parts) != 8:
        # try to extract 8-rank-looking substring with regex of 8 groups separated by - or /
        # fallback: return fen and let validation fail later
        return fen
    return fen

# %%
# FEN <-> 64 labels helpers
PIECE_CLASSES = ['.', 'P','N','B','R','Q','K', 'p','n','b','r','q','k']
CLASS_TO_IDX = {c:i for i,c in enumerate(PIECE_CLASSES)}
IDX_TO_CLASS = {i:c for i,c in enumerate(PIECE_CLASSES)}


def fen_to_64_labels(fen_piece_placement: str):
    ranks = fen_piece_placement.split('/')
    if len(ranks) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(ranks)}: {fen_piece_placement}")
    labels = []
    for r in ranks:
        for ch in r:
            if ch.isdigit():
                n = int(ch)
                labels.extend([CLASS_TO_IDX['.']]*n)
            else:
                if ch not in CLASS_TO_IDX:
                    raise ValueError(f"Invalid piece char: {ch}")
                labels.append(CLASS_TO_IDX[ch])
    if len(labels) != 64:
        raise ValueError(f"Parsed labels length != 64: {len(labels)}")
    return labels


def labels64_to_fen(labels64):
    rows = []
    for r in range(8):
        row = ''
        empties = 0
        for c in range(8):
            idx = labels64[r*8 + c]
            sym = IDX_TO_CLASS[idx]
            if sym == '.':
                empties += 1
            else:
                if empties:
                    row += str(empties); empties = 0
                row += sym
        if empties:
            row += str(empties)
        rows.append(row)
    return '/'.join(rows)

# %%
# BOARD DETECTION & WARP (best-effort) using OpenCV

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_board_corners(img, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:200]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            pts = approx.reshape(4,2)
            rect = order_points(pts)
            return rect
    # fallback: try Hough lines + intersection box (not implemented here)
    return None


def warp_board(img, corners, out_size=WARPED_SIZE):
    tl, tr, br, bl = corners
    src = np.array([tl, tr, br, bl], dtype="float32")
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (out_size, out_size))
    return warped, M

# %%
# SPLIT INTO 64 SQUARES

def split_board_into_squares(warped_img, size=WARPED_SIZE):
    s = size // 8
    crops = []
    for r in range(8):
        for c in range(8):
            y0, x0 = r*s, c*s
            crop = warped_img[y0:y0+s, x0:x0+s]
            crops.append(crop)
    return crops

# %%
# PRE-EXTRACTION: scan images, parse filename fen, detect & warp, save 64 crops with labels in filename

def preextract_all(images_dir=IMAGES_DIR, out_dir=OUT_EXTRACT_DIR, square_size=SQUARE_SIZE, warped_size=WARPED_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted([str(p) for p in Path(images_dir).glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    idx = 0
    failures = []
    for img_path in tqdm(image_paths, desc="Preextract"):
        try:
            fen = filename_to_fen(img_path)
            labels64 = fen_to_64_labels(fen)
        except Exception as e:
            failures.append((img_path, str(e)))
            continue
        img = cv2.imread(img_path)
        if img is None:
            failures.append((img_path, 'imread failed'))
            continue
        corners = detect_board_corners(img)
        if corners is None:
            # attempt a simpler fallback: assume image is already warped and square-sized
            h,w = img.shape[:2]
            if abs(h-w) < 10:
                warped = cv2.resize(img, (warped_size, warped_size))
            else:
                failures.append((img_path, 'board detection failed'))
                continue
        else:
            warped, _ = warp_board(img, corners, out_size=warped_size)
        crops = split_board_into_squares(warped, size=warped_size)
        base = Path(img_path).stem
        for i, crop in enumerate(crops):
            lbl = labels64[i]
            fname = f"{base}_{i:02d}_{lbl}.png"
            outp = Path(out_dir) / fname
            small = cv2.resize(crop, (square_size, square_size))
            cv2.imwrite(str(outp), small)
        idx += 1
    print(f"Done. Processed {idx} images. Failures: {len(failures)}")
    if failures:
        print("Examples:")
        for p,e in failures[:10]:
            print(p, e)

# %%
# RUN PRE-EXTRACTION (uncomment when ready)
# preextract_all()

# %%
# AFTER PRE-EXTRACTION: build PyTorch dataset (reads extracted files)

# The extraction saves files named: <originalbase>_NN_CC.png where NN is square index 00..63 and CC is label idx
# We'll create a dataset that reads those.

# %%
# PyTorch Dataset & DataLoader (per-square classifier)

# Note: imports here to avoid requiring torch during preprocessing-only runs
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ExtractedSquareDataset(Dataset):
    def __init__(self, extracted_dir, transform=None):
        self.paths = []
        for p in Path(extracted_dir).glob('*.png'):
            # parse label from filename
            m = re.match(r"^(.+)_([0-9]{2})_([0-9]+)\.png$", p.name)
            if not m:
                continue
            label = int(m.group(3))
            self.paths.append((str(p), label))
        self.transform = transform or T.Compose([T.Grayscale(), T.ToTensor(), T.Resize((128,128))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p,label = self.paths[idx]
        img = Image.open(p).convert('RGB')
        img = self.transform(img)
        return img, label

# %%
# MODEL (ResNet18 adapted for 1-channel)
import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=13, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # change first conv to accept single-channel if using Grayscale
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# %%
# TRAINING LOOP (simple)

def train_from_extracted(extracted_dir, epochs=10, batch_size=256, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([T.Grayscale(), T.Resize((128,128)), T.ToTensor()])
    ds = ExtractedSquareDataset(extracted_dir, transform=transform)
    # simple train/val split
    n = len(ds)
    idxs = list(range(n))
    split = int(n*0.9)
    train_idx, val_idx = idxs[:split], idxs[split:]
    from torch.utils.data import Subset
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = get_resnet18(num_classes=len(PIECE_CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optim.zero_grad(); loss.backward(); optim.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        # validation
        model.eval()
        vloss = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device); labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                vloss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                vcorrect += (preds==labels).sum().item()
                vtotal += imgs.size(0)
        val_loss = vloss / vtotal
        val_acc = vcorrect / vtotal
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        # checkpoint
        torch.save(model.state_dict(), f"square_model_epoch{epoch}.pth")
    return model

# %%
# INFERENCE: reconstruct FEN from original image using trained model

def infer_image_to_fen(img_path, model, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    img = cv2.imread(img_path)
    corners = detect_board_corners(img)
    if corners is None:
        h,w = img.shape[:2]
        if abs(h-w) < 10:
            warped = cv2.resize(img, (WARPED_SIZE, WARPED_SIZE))
        else:
            raise RuntimeError('Board detection failed')
    else:
        warped, _ = warp_board(img, corners, out_size=WARPED_SIZE)
    crops = split_board_into_squares(warped, size=WARPED_SIZE)
    transform = T.Compose([T.Grayscale(), T.Resize((128,128)), T.ToTensor()])
    labels_pred = []
    for crop in crops:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        t = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            pred = out.argmax(dim=1).item()
        labels_pred.append(pred)
    fen = labels64_to_fen(labels_pred)
    return fen

# %%
# USAGE SUMMARY
usage = """
1) Put your images in IMAGES_DIR and ensure filenames are exactly the FEN-like string with '-' separators.
   Example: r7-8-1K3nR1-p2P1NRq-8-6r1-5Pn1-1k2b3.png
2) Run preextract_all() (may take time). This writes per-square crops to OUT_EXTRACT_DIR.
3) Run train_from_extracted(OUT_EXTRACT_DIR, epochs=..., batch_size=...)
4) Use infer_image_to_fen(img, trained_model) to get predicted FEN.

Notes:
- If filenames contain extra text, edit filename_to_fen() accordingly to extract the FEN substring.
- Consider sharding extraction across multiple machines or using multiprocessing for scale.
"""
print(usage)
