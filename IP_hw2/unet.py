import os
import cv2
import numpy as np
from glob import glob
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------
# 同層設定
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 同層 carpalTunnel/
DATA_ROOT = os.path.join(PROJECT_DIR, "carpalTunnel")

# 同層 runs_ctftmn_unet/
OUT_DIR = os.path.join(PROJECT_DIR, "runs_ctftmn_unet")

# 模型輸入尺寸
MODEL_SIZE = 352

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 避免 multiprocessing + albumentations
NUM_WORKERS = 0

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


# 支援 Windows 中文路徑
def cv2_imread_unicode(path: str, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


# -------------------------
# Dataset 工具
def list_subjects(root: str) -> List[str]:
    subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    subs = sorted(subs, key=lambda x: int(x))
    return subs


def _numsort(paths: List[str]) -> List[str]:
    def key(p):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            return (0, int(name))
        except Exception:
            return (1, name)
    return sorted(paths, key=key)


def _grab_imgs(folder: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths += glob(os.path.join(folder, ext))
    return _numsort(paths)


def build_samples(root: str, subject_ids: List[str]) -> List[Dict]:
    samples = []
    for sid in subject_ids:
        base = os.path.join(root, sid)

        t1s = _grab_imgs(os.path.join(base, "T1"))
        t2s = _grab_imgs(os.path.join(base, "T2"))
        cts = _grab_imgs(os.path.join(base, "CT"))
        fts = _grab_imgs(os.path.join(base, "FT"))
        mns = _grab_imgs(os.path.join(base, "MN"))

        n = min(len(t1s), len(t2s), len(cts), len(fts), len(mns))
        for i in range(n):
            samples.append({
                "sid": sid,
                "t1": t1s[i], "t2": t2s[i],
                "ct": cts[i], "ft": fts[i], "mn": mns[i]
            })
    return samples


class CarpalTunnelDataset(Dataset):
    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform

    def _read_gray(self, path: str) -> np.ndarray:
        img = cv2_imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"OpenCV 讀不到(中文路徑/檔案損毀)：{path}")
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        t1 = self._read_gray(s["t1"])
        t2 = self._read_gray(s["t2"])

        ct = (self._read_gray(s["ct"]) > 127).astype(np.uint8)
        ft = (self._read_gray(s["ft"]) > 127).astype(np.uint8)
        mn = (self._read_gray(s["mn"]) > 127).astype(np.uint8)

        # image: H,W,2
        img = np.stack([t1, t2], axis=-1)

        # mask: H,W,3
        msk = np.stack([ct, ft, mn], axis=-1)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img = aug["image"].float()   # 保證 float
            msk = aug["mask"].float()    # mask 保證 float
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            msk = torch.from_numpy(msk.transpose(2, 0, 1)).float()

        return img, msk


# -------------------------
# Augmentation（2-channel 相容）
def get_train_tf():
    return A.Compose([
        A.Resize(MODEL_SIZE, MODEL_SIZE),

        A.HorizontalFlip(p=0.5),

        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.90, 1.10),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8
        ),

        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.08, p=1.0),
            A.ElasticTransform(alpha=20, sigma=6,
                               border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        ], p=0.2),

        # 支援任意 channel（包含 2-channel）
        A.OneOf([
            A.RandomBrightnessContrast(0.12, 0.12, p=1.0),
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
        ], p=0.7),

        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),

        A.ToFloat(max_value=255.0),
        ToTensorV2(transpose_mask=True),
    ])




def get_val_tf():
    return A.Compose([
        A.Resize(MODEL_SIZE, MODEL_SIZE),
        A.ToFloat(max_value=255.0),
        ToTensorV2(transpose_mask=True),
    ])


# -------------------------
# UNetSmall
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=2, out_ch=3, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.b = DoubleConv(base * 4, base * 8)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        xb = self.b(self.p3(x3))

        y3 = self.u3(xb)
        y3 = self.c3(torch.cat([y3, x3], dim=1))
        y2 = self.u2(y3)
        y2 = self.c2(torch.cat([y2, x2], dim=1))
        y1 = self.u1(y2)
        y1 = self.c1(torch.cat([y1, x1], dim=1))
        return self.out(y1)  # logits


# -------------------------
# Loss / Dice + plots
def dice_per_channel(prob: torch.Tensor, gt: torch.Tensor, eps=1e-6):
    dices = []
    for c in range(prob.shape[1]):
        p = prob[:, c].reshape(prob.shape[0], -1)
        g = gt[:, c].reshape(gt.shape[0], -1)
        inter = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        d = (2 * inter + eps) / (denom + eps)
        dices.append(d.mean())
    return torch.stack(dices, dim=0)


class DiceLoss(nn.Module):
    def forward(self, logits, gt):
        prob = torch.sigmoid(logits)
        d = dice_per_channel(prob, gt)
        return 1.0 - d.mean()


class FocalBCEWithLogits(nn.Module):
    """
    不做 class weighting，只做 hard example mining（focal）
    targets 需為 float {0,1}
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss
        return loss.mean()


def save_plots(history: dict, out_dir: str):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    for cls in ["CT", "FT", "MN"]:
        plt.figure()
        plt.plot(history[f"train_dice_{cls}"], label=f"train_dice_{cls}")
        plt.plot(history[f"val_dice_{cls}"], label=f"val_dice_{cls}")
        plt.legend(); plt.xlabel("epoch"); plt.ylabel("dice"); plt.ylim(0, 1)
        plt.title(f"Dice Curve - {cls}")
        plt.savefig(os.path.join(out_dir, f"dice_curve_{cls}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    plt.figure()
    plt.plot(history["train_dice_mean"], label="train_dice_mean")
    plt.plot(history["val_dice_mean"], label="val_dice_mean")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("dice"); plt.ylim(0, 1)
    plt.title("Mean Dice Curve")
    plt.savefig(os.path.join(out_dir, "dice_curve_mean.png"), dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
#  main
def main():
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(
            "找不到資料夾 carpalTunnel。\n"
            "請把資料夾放在與 train_ctftmn_unet.py 同一層：\n"
            f"{DATA_ROOT}\n"
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    subjects = list_subjects(DATA_ROOT)
    if len(subjects) < 3:
        raise ValueError("subjects 太少，至少需要 3 個資料夾（例如 0,1,2...）")

    # 最後 1 個資料夾不用來訓練/驗證當 test
    train_sub = subjects[:-2]     # 0~6
    val_sub   = subjects[-2:-1]   # 7
    

    train_samples = build_samples(DATA_ROOT, train_sub)
    val_samples = build_samples(DATA_ROOT, val_sub)

    print("Subjects:", subjects)
    print("Train subjects:", train_sub, "=>", len(train_samples), "samples")
    print("Val subjects  :", val_sub,   "=>", len(val_samples), "samples")
    print("Test subject  :", subjects[-1], "(unused in training)")

    if len(train_samples) == 0:
        raise RuntimeError(
            "train_samples = 0，代表程式沒有抓到任何影像。\n"
            "請確認 carpalTunnel/0~8/CT|FT|MN|T1|T2 內有圖片。"
        )

    train_ds = CarpalTunnelDataset(train_samples, transform=get_train_tf())
    val_ds = CarpalTunnelDataset(val_samples, transform=get_val_tf())

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )

    model = UNetSmall(in_ch=2, out_ch=3, base=32).to(DEVICE)

    focal = FocalBCEWithLogits(alpha=0.25, gamma=2.0)
    dloss = DiceLoss()

    # lr 為 3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice_CT": [], "train_dice_FT": [], "train_dice_MN": [],
        "val_dice_CT": [], "val_dice_FT": [], "val_dice_MN": [],
        "train_dice_mean": [], "val_dice_mean": [],
    }

    best = -1
    patience = 18
    bad = 0

    for epoch in range(1, 201):
        # ---- train ----
        model.train()
        tr_losses = []
        tr_dsum = torch.zeros(3, device=DEVICE)

        for img, gt in tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            img = img.to(DEVICE).float()
            gt  = gt.to(DEVICE).float()

            logits = model(img)
            loss = focal(logits, gt) + dloss(logits, gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                tr_dsum += dice_per_channel(prob, gt)

            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses))
        tr_dice = (tr_dsum / len(train_loader)).detach().cpu().numpy()
        tr_mean = float(tr_dice.mean())

        # ---- val ----
        model.eval()
        va_losses = []
        va_dsum = torch.zeros(3, device=DEVICE)

        with torch.no_grad():
            for img, gt in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
                img = img.to(DEVICE).float()
                gt  = gt.to(DEVICE).float()

                logits = model(img)
                loss = focal(logits, gt) + dloss(logits, gt)
                va_losses.append(loss.item())

                prob = torch.sigmoid(logits)
                va_dsum += dice_per_channel(prob, gt)

        va_loss = float(np.mean(va_losses))
        va_dice = (va_dsum / len(val_loader)).detach().cpu().numpy()
        va_mean = float(va_dice.mean())

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        history["train_dice_CT"].append(float(tr_dice[0]))
        history["train_dice_FT"].append(float(tr_dice[1]))
        history["train_dice_MN"].append(float(tr_dice[2]))
        history["val_dice_CT"].append(float(va_dice[0]))
        history["val_dice_FT"].append(float(va_dice[1]))
        history["val_dice_MN"].append(float(va_dice[2]))

        history["train_dice_mean"].append(tr_mean)
        history["val_dice_mean"].append(va_mean)

        print(
            f"[{epoch:03d}] train loss {tr_loss:.4f} meanDice {tr_mean:.3f} | "
            f"val loss {va_loss:.4f} meanDice {va_mean:.3f} "
            f"(CT {va_dice[0]:.3f}, FT {va_dice[1]:.3f}, MN {va_dice[2]:.3f})"
        )

        sch.step(va_mean)

        if va_mean > best + 1e-4:
            best = va_mean
            bad = 0
            torch.save(
                {"model_state": model.state_dict(), "model_size": MODEL_SIZE},
                os.path.join(OUT_DIR, "best_unet_ctftmn.pt")
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop. Best val meanDice={best:.3f}")
                break

    np.save(os.path.join(OUT_DIR, "history.npy"), history, allow_pickle=True)
    save_plots(history, OUT_DIR)

    print("\nDONE.")
    print("Data root:", DATA_ROOT)
    print("Outputs :", OUT_DIR)
    print("Model   :", os.path.join(OUT_DIR, "best_unet_ctftmn.pt"))


if __name__ == "__main__":
    main()




