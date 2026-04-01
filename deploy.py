import os
import re
import cv2
import tempfile
import numpy as np
import streamlit as st
import pytesseract
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="ALPR Plate Detection + OCR (Cloud)", layout="wide")
st.title("ALPR Plate Detection + OCR (Cloud Demo)")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
PROJECT_OUT = BASE_DIR / "outputs"
PROJECT_OUT.mkdir(exist_ok=True)

# Streamlit Cloud Linux tesseract path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Malaysia plate regex (general):
# 1-3 letters + 1-4 digits + optional trailing letter
PLATE_REGEX = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def normalize_plate(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def is_valid_plate(text: str) -> bool:
    return bool(PLATE_REGEX.fullmatch(text))


def preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 20:
        return None

    up = cv2.resize(crop_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def run_ocr_stable(crop_bgr: np.ndarray) -> str:
    h, w = crop_bgr.shape[:2]
    if w < 90 or h < 25:
        return ""

    proc = preprocess_for_ocr(crop_bgr)
    if proc is None:
        return ""

    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    txt = pytesseract.image_to_string(proc, config=cfg)
    txt = normalize_plate(txt)

    # accept valid MY format first
    if is_valid_plate(txt):
        return txt

    # fallback: keep plausible raw OCR for display/debug
    if 4 <= len(txt) <= 10:
        return txt

    return ""


def vote_plates(candidates):
    if not candidates:
        return ""

    # Prefer valid MY formats
    valid = [c for c in candidates if is_valid_plate(c)]
    target = valid if valid else candidates
    return Counter(target).most_common(1)[0][0]


def get_best_plate_crop(frame_bgr: np.ndarray, result):
    """
    Returns:
      best_crop, best_box, best_score, annotated_frame
      or None crop if no detection
    """
    annotated = result.plot()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None, None, 0.0, annotated

    h, w = frame_bgr.shape[:2]
    best_score = 0.0
    best_crop = None
    best_box = None

    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        conf = float(b.conf[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy.tolist()

        # small padding helps avoid cutting characters
        pad_x = int((x2 - x1) * 0.03)
        pad_y = int((y2 - y1) * 0.08)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)

        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area = bw * bh
        if area <= 0:
            continue

        score = conf * np.sqrt(area)
        if score > best_score:
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                best_score = score
                best_crop = crop
                best_box = (x1, y1, x2, y2)

    return best_crop, best_box, best_score, annotated


def show_result(crop_bgr: np.ndarray, text: str, prefix: str):
    c1, c2 = st.columns(2)
    with c1:
        st.image(crop_bgr[:, :, ::-1], caption=f"{prefix} Plate Crop", use_column_width=True)
    with c2:
        if text and is_valid_plate(text):
            st.success(f"{prefix} OCR (valid MY plate): {text}")
        elif text:
            st.warning(f"{prefix} OCR (raw, format uncertain): {text}")
        else:
            st.warning(f"{prefix} OCR: No text detected.")


# -----------------------------
# Load model + UI
# -----------------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence (conf)", 0.0, 1.0, 0.30, 0.05)
imgsz = st.sidebar.selectbox("Image Size (imgsz)", [320, 480, 640, 960, 1280], index=2)
video_stride = st.sidebar.slider("Video frame stride (process every Nth frame)", 1, 10, 2)
vote_samples = st.sidebar.slider("Vote samples (video)", 3, 30, 12)

tab1, tab2 = st.tabs(["Image Upload", "Video Upload"])

# -----------------------------
# Tab 1: Image
# -----------------------------
with tab1:
    st.subheader("Upload an image")
    img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_uploader")

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Failed to read image.")
        else:
            result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            crop, _, _, annotated = get_best_plate_crop(frame, result)

            st.image(
                annotated[:, :, ::-1],
                caption=f"Detections (conf={conf}, imgsz={imgsz})",
                use_column_width=True,
            )

            if crop is not None:
                txt = run_ocr_stable(crop)
                show_result(crop, txt, "Image")
            else:
                st.warning("No plate detected in this image.")

# -----------------------------
# Tab 2: Video Upload
# -----------------------------
with tab2:
    st.subheader("Upload a video")
    vid_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"], key="vid_uploader")

    if vid_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vid_file.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            st.error("Cannot open uploaded video.")
        else:
            frame_idx = 0
            best_score = 0.0
            best_crop = None
            best_annotated = None
            ocr_candidates = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            progress = st.progress(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % video_stride != 0:
                    continue

                result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                crop, _, score, annotated = get_best_plate_crop(frame, result)

                if crop is not None:
                    if score > best_score:
                        best_score = score
                        best_crop = crop.copy()
                        best_annotated = annotated.copy()

                    txt = run_ocr_stable(crop)
                    if txt:
                        ocr_candidates.append(txt)
                        if len(ocr_candidates) > vote_samples:
                            ocr_candidates.pop(0)

                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            progress.progress(1.0)

            if best_crop is not None:
                st.image(best_annotated[:, :, ::-1], caption="Best captured frame", use_column_width=True)

                voted = vote_plates(ocr_candidates)
                if not voted:
                    voted = run_ocr_stable(best_crop)

                show_result(best_crop, voted, "Video")
                if ocr_candidates:
                    st.caption(f"OCR samples: {ocr_candidates}")
            else:
                st.warning("No plate detected in video.")