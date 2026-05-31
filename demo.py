"""
圖像分類 Demo 介面（Streamlit）
執行方式：streamlit run demo.py
需先完成 train.py 的訓練，outputs/ 目錄下要有 best_model.pth 與 class_names.json
"""

import json
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# ─────────────────────────────────────────
# 設定（需與 train.py 相同）
# ─────────────────────────────────────────

MODEL_NAME  = "mobilenet"   # 需與訓練時一致
IMAGE_SIZE  = 224
SAVE_DIR    = "outputs"

# ─────────────────────────────────────────
# 載入模型（快取避免重複載入）
# ─────────────────────────────────────────

@st.cache_resource
def load_model():
    with open(f"{SAVE_DIR}/class_names.json") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    if MODEL_NAME == "mobilenet":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif MODEL_NAME == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif MODEL_NAME == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(f"{SAVE_DIR}/best_model.pth", map_location=device))
    model.to(device).eval()

    return model, class_names, device


def preprocess(img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


def predict(model, tensor, device, class_names):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu()
    top_probs, top_idxs = probs.topk(len(class_names))
    return [
        {"label": class_names[i], "score": float(p)}
        for p, i in zip(top_probs, top_idxs)
    ]


# ─────────────────────────────────────────
# Streamlit 頁面
# ─────────────────────────────────────────

st.set_page_config(
    page_title="圖像分類 Demo",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 圖像分類 Demo")
st.caption("上傳一張圖片，模型會預測它屬於哪個類別。")

# 載入模型
try:
    model, class_names, device = load_model()
    st.success(f"模型載入成功！共 {len(class_names)} 個類別：{', '.join(class_names)}")
except FileNotFoundError:
    st.error("找不到模型檔案，請先執行 train.py 完成訓練。")
    st.stop()

st.divider()

# 上傳區
uploaded = st.file_uploader(
    "選擇一張圖片",
    type=["jpg", "jpeg", "png", "webp"],
    help="支援 JPG / PNG / WebP"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img, caption="上傳的圖片", use_column_width=True)

    with col2:
        with st.spinner("預測中..."):
            tensor  = preprocess(img)
            results = predict(model, tensor, device, class_names)

        # 主要預測結果
        top = results[0]
        st.metric(
            label="預測類別",
            value=top["label"],
            delta=f"信心度 {top['score']*100:.1f}%"
        )

        st.write("**各類別信心分數：**")
        for r in results:
            pct = r["score"] * 100
            st.write(f"`{r['label']}`")
            st.progress(r["score"], text=f"{pct:.1f}%")

st.divider()

# 側邊說明
with st.sidebar:
    st.header("關於此專題")
    st.markdown("""
    **模型架構**
    - MobileNetV3-Small
    - ImageNet 預訓練 + 遷移學習

    **訓練資料**
    - 請填入你的資料集說明

    **輸出類別**
    """)
    if "class_names" in dir():
        for i, name in enumerate(class_names):
            st.write(f"{i+1}. {name}")

    st.markdown("""
    ---
    **使用方式**
    ```bash
    pip install streamlit torch torchvision pillow
    streamlit run demo.py
    ```
    """)