import html
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


_RCARD_BOX = (
    "border:1px solid rgba(128,132,149,0.22);border-radius:14px;padding:1.25rem 1.5rem;"
    "background:rgba(128,132,149,0.06);box-sizing:border-box;"
)


def _result_card_title(label: str) -> str:
    return (
        f'<p style="margin:0 0 10px 0;font-size:11px;font-weight:700;letter-spacing:0.12em;'
        f'color:rgba(128,132,149,0.92);">{html.escape(label)}</p>'
    )


def _result_pill(kind: str, text: str) -> str:
    if kind == "ok":
        bg, fg = "rgba(46,125,50,0.16)", "rgba(30,95,35,0.95)"
    elif kind == "warn":
        bg, fg = "rgba(230,126,34,0.18)", "rgba(140,80,0,0.95)"
    else:
        bg, fg = "rgba(128,132,149,0.14)", "rgba(70,72,82,0.95)"
    return (
        f'<span style="display:inline-block;padding:5px 14px;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:12px;font-weight:600;">{html.escape(text)}</span>'
    )


def _brand_or_model_card_html(
    title: str,
    deployed: bool,
    label: str,
    score: float,
    weights_hint: str = "",
) -> str:
    head = _result_card_title(title)
    if not deployed:
        pill = _result_pill("muted", "Unavailable")
        hint = (
            f" Missing weights: <code style='font-size:12px;'>{html.escape(weights_hint)}</code>."
            if weights_hint
            else ""
        )
        body = (
            "<p style='margin:8px 0 0 0;font-size:14px;line-height:1.5;color:rgba(128,132,149,0.88);'>"
            f"Model not deployed for this app.{hint}</p>"
        )
    elif not label:
        pill = _result_pill("warn", "No detection")
        body = (
            "<p style='margin:8px 0 0 0;font-size:14px;color:rgba(128,132,149,0.88);'>"
            "Nothing confident enough to report for this image.</p>"
        )
    else:
        pill = _result_pill("ok", "Detected")
        esc = html.escape(label)
        body = (
            f"<p style='margin:10px 0 6px 0;font-size:1.35rem;font-weight:750;line-height:1.3;'>{esc}</p>"
            f"<p style='margin:0;font-size:13px;color:rgba(128,132,149,0.85);'>"
            f"Confidence · {score:.2f}</p>"
        )
    return (
        f'<div style="{_RCARD_BOX} min-height:118px;">{head}'
        f'<div style="margin-bottom:10px;">{pill}</div>{body}</div>'
    )


def render_results_section(
    crop_bgr,
    plate_txt: str,
    brand_label: str,
    brand_score: float,
    car_label: str,
    car_score: float,
    brand_model,
    car_model,
    prefix: str = "Image",
):
    st.markdown("### Results")
    st.markdown(
        "<hr style='margin:0.4rem 0 1.2rem 0;border:none;border-top:1px solid rgba(128,132,149,0.2);' />",
        unsafe_allow_html=True,
    )

    plate_row_min_h = "min-height:280px;"

    if crop_bgr is not None:
        img_col, detail_col = st.columns(2, gap="large")
        with img_col:
            st.image(
                crop_bgr[:, :, ::-1],
                caption=f"{prefix} · plate crop",
                use_column_width=True,
            )
        with detail_col:
            if plate_txt and is_valid_plate(plate_txt):
                pill = _result_pill("ok", "Valid MY pattern")
                sub = "OCR matches the general Malaysia plate pattern."
            elif plate_txt:
                pill = _result_pill("warn", "Review format")
                sub = "Raw OCR does not match the default pattern — verify on the crop."
            else:
                pill = _result_pill("warn", "No OCR text")
                sub = "No reliable characters were read from the crop."
            val = html.escape(plate_txt) if plate_txt else "—"
            st.markdown(
                f'<div style="{_RCARD_BOX} {plate_row_min_h} display:flex;flex-direction:column;justify-content:center;">'
                f"{_result_card_title('License plate · OCR')}"
                f'<div style="margin-bottom:12px;">{pill}</div>'
                f'<p style="margin:6px 0 10px 0;font-size:1.85rem;font-weight:800;font-family:ui-monospace,Consolas,monospace;'
                f'letter-spacing:0.1em;line-height:1.2;">{val}</p>'
                f'<p style="margin:0;font-size:13px;line-height:1.45;color:rgba(128,132,149,0.88);">'
                f"{html.escape(sub)}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div style="{_RCARD_BOX} min-height:100px;">'
            f"{_result_card_title('License plate')}"
            f'<div style="margin-bottom:10px;">{_result_pill("warn", "Not detected")}</div>'
            f'<p style="margin:0;font-size:15px;line-height:1.5;color:rgba(128,132,149,0.9);">'
            "No plate region was found on this upload.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    row_b, row_m = st.columns(2, gap="large")
    with row_b:
        st.markdown(
            _brand_or_model_card_html(
                "Car brand",
                brand_model is not None,
                brand_label,
                brand_score,
                weights_hint="models/carbrand/best.pt",
            ),
            unsafe_allow_html=True,
        )
    with row_m:
        st.markdown(
            _brand_or_model_card_html(
                "Car model",
                car_model is not None,
                car_label,
                car_score,
                weights_hint="models/carmodel/best.pt",
            ),
            unsafe_allow_html=True,
        )


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
conf = st.sidebar.slider("Confidence (conf)", 0.0, 1.0, 0.15, 0.05)
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
        crop, _, _, annotated_plate = get_best_plate_crop(frame, result)

        brand_label, brand_score, im_brand = "", 0.0, None
        if brand_model is not None:
            brand_label, brand_score, im_brand = detect_car_brand(
                frame, brand_model, conf=max(0.2, conf), imgsz=imgsz
            )

        car_label, car_score, im_car = "", 0.0, None
        if car_model is not None:
            car_label, car_score, im_car = predict_car_model(
                frame, car_model, conf=max(0.2, conf), cls_imgsz=car_cls_imgsz, det_imgsz=imgsz
            )

        plate_txt = run_ocr_stable(crop, ocr_engine) if crop is not None else ""

        st.subheader("Detections")
        row1a, row1b = st.columns(2)
        with row1a:
            st.image(
                annotated_plate[:, :, ::-1],
                caption=f"Plate detection (conf={conf}, imgsz={imgsz})",
                use_column_width=True,
            )
        with row1b:
            if brand_model is not None:
                st.image(im_brand[:, :, ::-1], caption="Brand detection", use_column_width=True)
            else:
                st.caption("Brand model not deployed (`models/carbrand/best.pt`).")

        row2a, row2b = st.columns(2)
        with row2a:
            if car_model is not None:
                st.image(im_car[:, :, ::-1], caption="Car model detection", use_column_width=True)
            else:
                st.caption("Car model not deployed (`models/carmodel/best.pt`).")
        with row2b:
            st.caption("Reserved")

        render_results_section(
            crop,
            plate_txt,
            brand_label,
            brand_score,
            car_label,
            car_score,
            brand_model,
            car_model,
            prefix="Image",
        )