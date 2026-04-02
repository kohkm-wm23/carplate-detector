import re
import cv2
import numpy as np
import streamlit as st
from paddleocr import PaddleOCR
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Car Plate Detection System Yolov8", layout="wide")
st.title("Car Plate Detection System (Yolov8 + Brand + Car Model)")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

PLATE_MODEL_PATH = MODELS_DIR / "carplate" / "best.pt"
BRAND_MODEL_PATH = MODELS_DIR / "carbrand" / "best.pt"
CAR_MODEL_PATH = MODELS_DIR / "carmodel" / "best.pt"

PROJECT_OUT = BASE_DIR / "outputs"
PROJECT_OUT.mkdir(exist_ok=True)

# Malaysia plate (general): 1-3 letters + 1-4 digits + optional trailing letter
PLATE_REGEX = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$")


# -----------------------------
# Model + OCR loaders
# -----------------------------
@st.cache_resource
def load_plate_model():
    if not PLATE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Plate model not found: {PLATE_MODEL_PATH}")
    return YOLO(str(PLATE_MODEL_PATH))


@st.cache_resource
def load_brand_model():
    if not BRAND_MODEL_PATH.exists():
        return None
    return YOLO(str(BRAND_MODEL_PATH))


@st.cache_resource
def load_car_model():
    if not CAR_MODEL_PATH.exists():
        return None
    return YOLO(str(CAR_MODEL_PATH))


@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def normalize_plate(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def is_valid_plate(text: str) -> bool:
    return bool(PLATE_REGEX.fullmatch(text))


def preprocess_for_ocr(crop_bgr: np.ndarray):
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 20:
        return None

    up = cv2.resize(crop_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def run_ocr_stable(crop_bgr: np.ndarray, ocr_engine) -> str:
    h, w = crop_bgr.shape[:2]
    if w < 90 or h < 25:
        return ""

    proc = preprocess_for_ocr(crop_bgr)
    if proc is None:
        return ""

    try:
        result = ocr_engine.ocr(proc, cls=True)
    except Exception:
        return ""

    texts = []
    if result and len(result) > 0 and result[0]:
        for line in result[0]:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                t = normalize_plate(str(line[1][0]))
                if t:
                    texts.append(t)

    if not texts:
        return ""

    valid = [t for t in texts if is_valid_plate(t)]
    if valid:
        return max(valid, key=len)

    candidates = [t for t in texts if 4 <= len(t) <= 10]
    if candidates:
        return max(candidates, key=len)

    return ""


def get_best_plate_crop(frame_bgr: np.ndarray, result):
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


def detect_car_brand(frame_bgr, brand_model, conf: float = 0.25, imgsz: int = 960):
    result = brand_model.predict(source=frame_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    annotated = result.plot()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "", 0.0, annotated

    confs = boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    cls_id = int(boxes.cls[best_i].cpu().numpy())
    score = float(confs[best_i])

    names = brand_model.names
    if isinstance(names, dict):
        label = names.get(cls_id, str(cls_id))
    else:
        label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
    return str(label), score, annotated


def predict_car_model(frame_bgr, car_model, conf: float = 0.25, cls_imgsz: int = 224, det_imgsz: int = 960):
    task = getattr(car_model, "task", None)
    if task == "classify":
        result = car_model.predict(source=frame_bgr, imgsz=cls_imgsz, verbose=False)[0]
    else:
        result = car_model.predict(source=frame_bgr, conf=conf, imgsz=det_imgsz, verbose=False)[0]

    annotated = result.plot()
    probs = getattr(result, "probs", None)
    if probs is not None:
        idx = int(probs.top1)
        score = float(probs.top1conf)
        names = car_model.names
        if isinstance(names, dict):
            label = names.get(idx, str(idx))
        else:
            label = names[idx] if 0 <= idx < len(names) else str(idx)
        return str(label), score, annotated

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "", 0.0, annotated

    confs = boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    cls_id = int(boxes.cls[best_i].cpu().numpy())
    score = float(confs[best_i])

    names = car_model.names
    if isinstance(names, dict):
        label = names.get(cls_id, str(cls_id))
    else:
        label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
    return str(label), score, annotated


def show_brand_and_model_ui(frame_bgr, brand_model, car_model, conf, imgsz, cls_imgsz):
    if brand_model is not None:
        bl, bs, im_b = detect_car_brand(frame_bgr, brand_model, conf=max(0.2, conf), imgsz=imgsz)
        st.image(im_b[:, :, ::-1], caption=f"Brand: {bl or 'none'}", use_container_width=True)
        if bl:
            st.success(f"Brand: {bl} ({bs:.2f})")
        else:
            st.warning("Brand: not detected.")
    else:
        st.info("Brand model not deployed (missing `models/carbrand/best.pt`).")

    if car_model is not None:
        cl, cs, im_c = predict_car_model(
            frame_bgr, car_model, conf=max(0.2, conf), cls_imgsz=cls_imgsz, det_imgsz=imgsz
        )
        st.image(im_c[:, :, ::-1], caption=f"Car model: {cl or 'none'}", use_container_width=True)
        if cl:
            st.success(f"Car model: {cl} ({cs:.2f})")
        else:
            st.warning("Car model: no prediction.")
    else:
        st.info("Car model not deployed (missing `models/carmodel/best.pt`).")


def show_result(crop_bgr: np.ndarray, text: str, prefix: str):
    c1, c2 = st.columns(2)
    with c1:
        st.image(crop_bgr[:, :, ::-1], caption=f"{prefix} Plate Crop", use_container_width=True)
    with c2:
        if text and is_valid_plate(text):
            st.success(f"{prefix} OCR (valid MY plate): {text}")
        elif text:
            st.warning(f"{prefix} OCR (raw, format uncertain): {text}")
        else:
            st.warning(f"{prefix} OCR: No text detected.")


# -----------------------------
# Load models + UI
# -----------------------------
try:
    plate_model = load_plate_model()
    brand_model = load_brand_model()
    car_model = load_car_model()
    ocr_engine = load_ocr()
except Exception as e:
    st.error(f"Model/OCR load failed: {e}")
    st.stop()

st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence (conf)", 0.0, 1.0, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Image Size (imgsz)", [320, 480, 640, 960, 1280], index=4)
car_cls_imgsz = st.sidebar.selectbox("Car model imgsz (classify)", [224, 320, 384, 448], index=0)

st.subheader("Upload an image")
img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_uploader")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Failed to read image.")
    else:
        result = plate_model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        crop, _, _, annotated = get_best_plate_crop(frame, result)

        st.image(
            annotated[:, :, ::-1],
            caption=f"Plate detections (conf={conf}, imgsz={imgsz})",
            use_container_width=True,
        )

        if crop is not None:
            txt = run_ocr_stable(crop, ocr_engine)
            show_result(crop, txt, "Image")
        else:
            st.warning("No plate detected in this image.")

        st.subheader("Car brand & model (full image)")
        show_brand_and_model_ui(frame, brand_model, car_model, conf, imgsz, car_cls_imgsz)