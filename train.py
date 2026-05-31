"""
圖像分類專題 — 遷移學習訓練程式
模型：MobileNetV3-Small（可換成 EfficientNet-B0 / ResNet-50）
框架：PyTorch + torchvision
"""

import os
import copy
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 不需要 GUI 視窗

# ─────────────────────────────────────────
# 1. 設定區（只需修改這裡）
# ─────────────────────────────────────────

CONFIG = {
    "data_dir": "data",        # 指定剛剛建立的資料夾
    "pre_split": False,        # 設為 False！讓程式幫你自動 80/20 切分訓練與驗證集
    "num_classes": 3,          # 設為 3！因為我們只放了 cat, dog, bird 三個資料夾
    "batch_size": 16,          # 建議先設 16，避免電腦記憶體爆掉
    "num_epochs": 50,          # 10 個 Epoch 就能看到初步成效了
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "image_size": 224,         
    "num_workers": 0,          # Windows 系統務必保持 0
    "save_dir": "outputs",     
    "model_name": "mobilenet", 
    "device": "auto",          
}

# ─────────────────────────────────────────
# 2. 工具函式
# ─────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def build_transforms(image_size: int):
    """回傳 train / val 的資料前處理 pipeline。"""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


def load_datasets(cfg: dict):
    """
    載入資料集。
    - pre_split=True : data/train/ 和 data/val/ 分開存在
    - pre_split=False: data/ 下所有圖片，自動切 80% train / 20% val
    """
    train_tf, val_tf = build_transforms(cfg["image_size"])

    if cfg["pre_split"]:
        train_ds = datasets.ImageFolder(
            os.path.join(cfg["data_dir"], "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(
            os.path.join(cfg["data_dir"], "val"),   transform=val_tf)
    else:
        full_ds = datasets.ImageFolder(cfg["data_dir"], transform=train_tf)
        n_val   = int(len(full_ds) * 0.2)
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        # val_ds 套上 val_tf（不做資料增強）
        val_ds.dataset = copy.deepcopy(full_ds)
        val_ds.dataset.transform = val_tf

    return train_ds, val_ds


def build_model(cfg: dict) -> nn.Module:
    """
    建立預訓練模型，替換最後的分類層。
    支援 mobilenet / efficientnet / resnet50。
    """
    name = cfg["model_name"]
    n    = cfg["num_classes"]

    if name == "mobilenet":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # 凍結前段特徵擷取層（可選）
        for param in model.features.parameters():
            param.requires_grad = False
        # 替換分類頭
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, n)

    elif name == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n)

    elif name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        # 只凍結前 6 個 layer block
        layers_to_freeze = [model.layer1, model.layer2, model.layer3]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, n)

    else:
        raise ValueError(f"未知模型名稱: {name}，請選 mobilenet/efficientnet/resnet50")

    return model


# ─────────────────────────────────────────
# 3. 訓練 / 驗證一個 epoch
# ─────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────
# 4. 訓練主迴圈
# ─────────────────────────────────────────

def train(cfg: dict):
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = get_device(cfg)
    print(f"使用裝置: {device}")

    # 資料載入
    train_ds, val_ds = load_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])

    # 取得類別名稱（用於儲存與報告）
    if hasattr(train_ds, "classes"):
        class_names = train_ds.classes
    elif hasattr(train_ds, "dataset"):
        class_names = train_ds.dataset.classes
    else:
        class_names = [str(i) for i in range(cfg["num_classes"])]

    print(f"類別: {class_names}")
    print(f"訓練集大小: {len(train_ds)}，驗證集大小: {len(val_ds)}")

    # 模型
    model     = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing 提升泛化
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=1e-5)

    # 記錄歷史
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_acc   = 0.0
    best_model = None

    print(f"\n開始訓練，共 {cfg['num_epochs']} 個 epoch\n{'─'*55}")

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, is_train=False)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 儲存最佳模型
        if val_acc > best_acc:
            best_acc   = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model,
                       os.path.join(cfg["save_dir"], "best_model.pth"))

        print(
            f"Epoch {epoch:02d}/{cfg['num_epochs']} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Best: {best_acc:.4f} | {elapsed:.1f}s"
        )

    print(f"\n訓練完成！最佳驗證準確率: {best_acc:.4f}")

    # 儲存最後一個 epoch 的模型
    torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "last_model.pth"))

    # 儲存類別對照表（Streamlit Demo 會用到）
    with open(os.path.join(cfg["save_dir"], "class_names.json"), "w") as f:
        json.dump(class_names, f, ensure_ascii=False)

    # 儲存訓練歷史
    with open(os.path.join(cfg["save_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history, class_names


# ─────────────────────────────────────────
# 5. 畫訓練曲線（放進報告）
# ─────────────────────────────────────────

def plot_history(history: dict, save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss 曲線
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#185FA5")
    axes[0].plot(epochs, history["val_loss"],   "s--", label="Val Loss",   color="#D85A30")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy 曲線
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#185FA5")
    axes[1].plot(epochs, history["val_acc"],   "s--", label="Val Acc",   color="#D85A30")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"訓練曲線已儲存: {out_path}")


# ─────────────────────────────────────────
# 6. 混淆矩陣（放進報告）
# ─────────────────────────────────────────

def plot_confusion_matrix(cfg: dict, class_names: list):
    """載入最佳模型，對驗證集預測，畫出混淆矩陣。"""
    try:
        import numpy as np
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except ImportError:
        print("請安裝 scikit-learn: pip install scikit-learn")
        return

    device = get_device(cfg)
    _, val_tf = build_transforms(cfg["image_size"])

    if cfg["pre_split"]:
        val_ds = datasets.ImageFolder(
            os.path.join(cfg["data_dir"], "val"), transform=val_tf)
    else:
        full_ds = datasets.ImageFolder(cfg["data_dir"], transform=val_tf)
        n_val   = int(len(full_ds) * 0.2)
        n_train = len(full_ds) - n_val
        _, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])

    model = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(os.path.join(cfg["save_dir"], "best_model.pth"),
                   map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Validation Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(cfg["save_dir"], "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩陣已儲存: {out_path}")


# ─────────────────────────────────────────
# 7. 單張圖片推論（測試用）
# ─────────────────────────────────────────

def predict_single(image_path: str, cfg: dict, class_names: list):
    """對單張圖片進行預測，回傳類別與信心分數。"""
    from PIL import Image

    device = get_device(cfg)
    _, val_tf = build_transforms(cfg["image_size"])

    img    = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(device)

    model = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(os.path.join(cfg["save_dir"], "best_model.pth"),
                   map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu()

    top3 = probs.topk(min(3, len(class_names)))
    print(f"\n預測結果 ({image_path}):")
    for score, idx in zip(top3.values, top3.indices):
        print(f"  {class_names[idx]:20s} {score.item()*100:.1f}%")

    return class_names[probs.argmax()], probs.max().item()


# ─────────────────────────────────────────
# 8. 主程式入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    # ── 訓練 ──
    history, class_names = train(CONFIG)

    # ── 畫圖（報告必備）──
    plot_history(history, CONFIG["save_dir"])
    plot_confusion_matrix(CONFIG, class_names)

    print("\n所有輸出已儲存到:", CONFIG["save_dir"])
    print("  best_model.pth      ← 最佳模型權重")
    print("  last_model.pth      ← 最後一個 epoch 的模型")
    print("  class_names.json    ← 類別對照表（Demo 使用）")
    print("  history.json        ← 訓練歷史數據")
    print("  training_curves.png ← 訓練曲線圖（放報告）")
    print("  confusion_matrix.png← 混淆矩陣圖（放報告）")