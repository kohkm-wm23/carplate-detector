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
st.set_page_config(page_title="Car Plate + Brand (YOLOv8)", layout="wide")
st.title("Car Plate Detection System (Yolov8 + Brand)")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

PLATE_MODEL_PATH = MODELS_DIR / "carplate" / "best.pt"
BRAND_MODEL_PATH = MODELS_DIR / "carbrand" / "best.pt"

PROJECT_OUT = BASE_DIR / "outputs"
PROJECT_OUT.mkdir(exist_ok=True)

# Malaysia plate (general): 1-3 letters + 1-4 digits + optional trailing letter
PLATE_REGEX = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$")

# Match run_ocr_stable() minimums so we prefer crops OCR can actually use
MIN_PLATE_CROP_W = 90
MIN_PLATE_CROP_H = 25
_PAD_X_STD = 0.03
_PAD_Y_STD = 0.08
_PAD_X_EXPAND = 0.10
_PAD_Y_EXPAND = 0.20


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


def preprocess_for_ocr_grayscale_bgr(crop_bgr: np.ndarray):
    """Upscaled grayscale as BGR — second pass when Otsu binary fails OCR."""
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 20:
        return None
    up = cv2.resize(crop_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _ocr_lines_to_texts(result) -> list:
    texts = []
    if result and len(result) > 0 and result[0]:
        for line in result[0]:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                t = normalize_plate(str(line[1][0]))
                if t:
                    texts.append(t)
    return texts


def _pick_plate_string_from_texts(texts: list) -> str:
    if not texts:
        return ""
    valid = [t for t in texts if is_valid_plate(t)]
    if valid:
        return max(valid, key=len)
    candidates = [t for t in texts if 4 <= len(t) <= 10]
    if candidates:
        return max(candidates, key=len)
    return ""


def run_ocr_stable(crop_bgr: np.ndarray, ocr_engine) -> str:
    h, w = crop_bgr.shape[:2]
    if w < MIN_PLATE_CROP_W or h < MIN_PLATE_CROP_H:
        return ""

    def ocr_image(arr) -> str:
        if arr is None:
            return ""
        try:
            res = ocr_engine.ocr(arr, cls=True)
        except Exception:
            return ""
        return _pick_plate_string_from_texts(_ocr_lines_to_texts(res))

    proc_bin = preprocess_for_ocr(crop_bgr)
    if proc_bin is not None:
        out = ocr_image(proc_bin)
        if out:
            return out

    proc_gray = preprocess_for_ocr_grayscale_bgr(crop_bgr)
    out = ocr_image(proc_gray)
    if out:
        return out
    return ""


def _extract_padded_crop(frame_bgr, x1, y1, x2, y2, fh, fw, pad_x_frac: float, pad_y_frac: float):
    """xyxy from detector (not yet padded). Returns crop and clip coordinates after padding."""
    pad_x = int((x2 - x1) * pad_x_frac)
    pad_y = int((y2 - y1) * pad_y_frac)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(fw - 1, x2 + pad_x)
    y2p = min(fh - 1, y2 + pad_y)
    if x2p <= x1p or y2p <= y1p:
        return None, None
    crop = frame_bgr[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None, None
    return crop, (x1p, y1p, x2p, y2p)


def get_best_plate_crop(frame_bgr: np.ndarray, result):
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None, None, 0.0

    fh, fw = frame_bgr.shape[:2]
    scored = []
    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        det_conf = float(b.conf[0].cpu().numpy())
        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area = bw * bh
        if area <= 0:
            continue
        score = det_conf * np.sqrt(area)
        scored.append((score, x1, y1, x2, y2))

    if not scored:
        return None, None, 0.0

    scored.sort(key=lambda t: t[0], reverse=True)

    def first_meeting_min(pxf, pyf):
        for score, x1, y1, x2, y2 in scored:
            crop, box = _extract_padded_crop(frame_bgr, x1, y1, x2, y2, fh, fw, pxf, pyf)
            if crop is None:
                continue
            ch, cw = crop.shape[:2]
            if cw >= MIN_PLATE_CROP_W and ch >= MIN_PLATE_CROP_H:
                return crop, box, score
        return None, None, None

    crop, box, score = first_meeting_min(_PAD_X_STD, _PAD_Y_STD)
    if crop is not None:
        return crop, box, score

    crop, box, score = first_meeting_min(_PAD_X_EXPAND, _PAD_Y_EXPAND)
    if crop is not None:
        return crop, box, score

    top_score, x1, y1, x2, y2 = scored[0]
    crop, box = _extract_padded_crop(frame_bgr, x1, y1, x2, y2, fh, fw, _PAD_X_STD, _PAD_Y_STD)
    if crop is None:
        return None, None, 0.0
    return crop, box, top_score


def _cls_name(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, str(cls_id)))
    if 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _draw_detection_boxes_bgr(
    im_bgr: np.ndarray,
    boxes,
    names,
    color_bgr: tuple,
    tag: str,
) -> None:
    if boxes is None or len(boxes) == 0:
        return
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        cls_id = int(clss[i])
        cf = float(confs[i])
        lab = _cls_name(names, cls_id)
        label = f"{tag}{lab} {cf:.2f}"
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        y_txt = max(y1 - 6, 18)
        cv2.putText(
            im_bgr,
            label,
            (x1, y_txt),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            2,
            cv2.LINE_AA,
        )


def draw_combined_detection_plot(
    frame_bgr: np.ndarray,
    plate_result,
    brand_result,
    plate_model,
    brand_model,
) -> np.ndarray:
    """Single BGR image: plate + brand overlays."""
    canvas = frame_bgr.copy()
    h = canvas.shape[0]

    COL_PLATE = (0, 215, 255)  # BGR amber / plate
    COL_BRAND = (255, 128, 0)  # BGR brand accent

    if plate_result is not None and plate_result.boxes is not None:
        _draw_detection_boxes_bgr(canvas, plate_result.boxes, plate_model.names, COL_PLATE, "plate ")

    if brand_model is not None and brand_result is not None and brand_result.boxes is not None:
        _draw_detection_boxes_bgr(canvas, brand_result.boxes, brand_model.names, COL_BRAND, "brand ")

    leg = "Plate (amber)  Brand (orange)"
    cv2.putText(
        canvas,
        leg,
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def detect_car_brand(frame_bgr, brand_model, conf: float = 0.25, imgsz: int = 960):
    result = brand_model.predict(source=frame_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "", 0.0, result

    confs = boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    cls_id = int(boxes.cls[best_i].cpu().numpy())
    score = float(confs[best_i])

    names = brand_model.names
    label = _cls_name(names, cls_id)
    return str(label), score, result


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


def _brand_card_html(
    deployed: bool,
    label: str,
    score: float,
    weights_hint: str = "",
) -> str:
    title = "Car brand"
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
    has_plate_crop: bool,
    plate_txt: str,
    brand_label: str,
    brand_score: float,
    brand_model,
):
    st.markdown("### Results")
    st.markdown(
        "<hr style='margin:0.4rem 0 1.2rem 0;border:none;border-top:1px solid rgba(128,132,149,0.2);' />",
        unsafe_allow_html=True,
    )

    plate_card_min_h = "min-height:160px;"
    col_plate, col_brand = st.columns(2, gap="large")

    with col_plate:
        if has_plate_crop:
            if plate_txt and is_valid_plate(plate_txt):
                pill = _result_pill("ok", "Valid MY pattern")
                sub = "OCR matches the general Malaysia plate pattern."
            elif plate_txt:
                pill = _result_pill("warn", "Review format")
                sub = "Raw OCR does not match the default pattern — verify visually if needed."
            else:
                pill = _result_pill("warn", "No OCR text")
                sub = "No reliable characters were read from the detected plate region."
            val = html.escape(plate_txt) if plate_txt else "—"
            st.markdown(
                f'<div style="{_RCARD_BOX} {plate_card_min_h} display:flex;flex-direction:column;justify-content:center;">'
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

    with col_brand:
        st.markdown(
            _brand_card_html(
                brand_model is not None,
                brand_label,
                brand_score,
                weights_hint="models/carbrand/best.pt",
            ),
            unsafe_allow_html=True,
        )


# -----------------------------
# Load models + UI
# -----------------------------
try:
    plate_model = load_plate_model()
    brand_model = load_brand_model()
    ocr_engine = load_ocr()
except Exception as e:
    st.error(f"Model/OCR load failed: {e}")
    st.stop()

st.sidebar.header("Settings")
plate_conf = st.sidebar.slider(
    "Plate detection confidence",
    0.0,
    1.0,
    0.15,
    0.05,
    help="YOLO threshold for the license-plate model only. Lower keeps more plate boxes (feeds OCR).",
)
det_conf = st.sidebar.slider(
    "Brand detection confidence",
    0.0,
    1.0,
    0.25,
    0.05,
    help="YOLO threshold for the car brand model only (not used for OCR).",
)
imgsz = st.sidebar.selectbox("Image Size (imgsz)", [320, 480, 640, 960, 1280], index=4)

st.subheader("Upload an image")
img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_uploader")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Failed to read image.")
    else:
        plate_result = plate_model.predict(source=frame, conf=plate_conf, imgsz=imgsz, verbose=False)[0]
        crop, _, _ = get_best_plate_crop(frame, plate_result)

        brand_use_conf = max(0.2, det_conf)
        brand_label, brand_score = "", 0.0
        brand_result = None
        if brand_model is not None:
            brand_label, brand_score, brand_result = detect_car_brand(
                frame, brand_model, conf=brand_use_conf, imgsz=imgsz
            )

        combined_det = draw_combined_detection_plot(
            frame,
            plate_result,
            brand_result,
            plate_model,
            brand_model,
        )

        plate_txt = run_ocr_stable(crop, ocr_engine) if crop is not None else ""

        if crop is not None:
            ch, cw = crop.shape[:2]
            st.sidebar.caption(
                f"Plate crop: {cw}×{ch} px (OCR min {MIN_PLATE_CROP_W}×{MIN_PLATE_CROP_H})"
            )
        else:
            st.sidebar.caption("Plate crop: none at this plate confidence.")

        st.subheader("Detections")
        st.image(
            combined_det[:, :, ::-1],
            caption=f"Combined: plate + brand (plate_conf={plate_conf}, imgsz={imgsz}; legend at bottom)",
            use_column_width=True,
        )
        if brand_model is None:
            st.caption("Brand head not loaded — only plate boxes are drawn.")

        render_results_section(
            crop is not None,
            plate_txt,
            brand_label,
            brand_score,
            brand_model,
        )