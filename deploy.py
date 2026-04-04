import html
import io
import re
import cv2
import numpy as np
import streamlit as st
from paddleocr import PaddleOCR
from pathlib import Path
from ultralytics import YOLO

_HEIF_OPENER_REGISTERED = False


def _register_heif_opener():
    global _HEIF_OPENER_REGISTERED
    if _HEIF_OPENER_REGISTERED:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_OPENER_REGISTERED = True
    except ImportError:
        pass


def decode_upload_to_bgr(uploaded_file) -> np.ndarray | None:
    """JPEG/PNG via OpenCV; HEIC/HEIF via Pillow + pillow-heif (iPhone photos)."""
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix in (".heic", ".heif"):
        _register_heif_opener()
        try:
            from PIL import Image

            im = Image.open(io.BytesIO(raw))
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    buf = np.asarray(bytearray(raw), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

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


def _crop_for_single_plate_box(frame_bgr, x1, y1, x2, y2, fh, fw):
    """Padded crop for one plate box (same padding strategy as former single-plate path)."""
    crop, _ = _extract_padded_crop(frame_bgr, x1, y1, x2, y2, fh, fw, _PAD_X_STD, _PAD_Y_STD)
    if crop is not None:
        ch, cw = crop.shape[:2]
        if cw >= MIN_PLATE_CROP_W and ch >= MIN_PLATE_CROP_H:
            return crop
    crop, _ = _extract_padded_crop(frame_bgr, x1, y1, x2, y2, fh, fw, _PAD_X_EXPAND, _PAD_Y_EXPAND)
    if crop is not None:
        ch, cw = crop.shape[:2]
        if cw >= MIN_PLATE_CROP_W and ch >= MIN_PLATE_CROP_H:
            return crop
    crop, _ = _extract_padded_crop(frame_bgr, x1, y1, x2, y2, fh, fw, _PAD_X_STD, _PAD_Y_STD)
    return crop


def enumerate_plates_left_to_right(frame_bgr: np.ndarray, plate_result, max_cars: int) -> tuple:
    """
    Plate detections: keep up to max_cars with largest bbox area (drops tiny / distant plates),
    then sort left → right. Returns (plates_ltr, n_detected_raw).
    """
    boxes = plate_result.boxes
    if boxes is None or len(boxes) == 0:
        return [], 0
    fh, fw = frame_bgr.shape[:2]
    xyxy_all = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    rows = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in xyxy_all[i]]
        if x2 <= x1 or y2 <= y1:
            continue
        area = float((x2 - x1) * (y2 - y1))
        xc = 0.5 * (x1 + x2)
        crop = _crop_for_single_plate_box(frame_bgr, x1, y1, x2, y2, fh, fw)
        rows.append(
            {
                "x_center": xc,
                "xyxy": (x1, y1, x2, y2),
                "crop": crop,
                "conf": float(confs[i]),
                "area": area,
                "cls_id": int(clss[i]),
            }
        )
    n_raw = len(rows)
    mc = max(1, int(max_cars))
    if len(rows) > mc:
        rows.sort(key=lambda r: r["area"], reverse=True)
        rows = rows[:mc]
    rows.sort(key=lambda r: r["x_center"])
    return rows, n_raw


def extract_brands_left_to_right(brand_result, brand_model) -> list:
    """All brand boxes sorted by horizontal center (left → right)."""
    if brand_result is None or brand_result.boxes is None or len(brand_result.boxes) == 0:
        return []
    boxes = brand_result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    names = brand_model.names
    rows = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        xc = 0.5 * (x1 + x2)
        cid = int(clss[i])
        rows.append(
            {
                "x_center": xc,
                "label": _cls_name(names, cid),
                "score": float(confs[i]),
                "xyxy": (int(x1), int(y1), int(x2), int(y2)),
            }
        )
    rows.sort(key=lambda r: r["x_center"])
    return rows


def pair_plates_with_brands(plates_ltr: list, brands_ltr: list) -> list:
    """
    One row per plate (left → right). Each plate gets at most one brand:
    greedy match by nearest x_center among unused brands.
    """
    if not plates_ltr:
        return []
    used = set()
    paired = []
    for p in plates_ltr:
        best_j = None
        best_d = 1e18
        for j, b in enumerate(brands_ltr):
            if j in used:
                continue
            d = abs(p["x_center"] - b["x_center"])
            if d < best_d:
                best_d = d
                best_j = j
        brand = None
        if best_j is not None:
            used.add(best_j)
            brand = brands_ltr[best_j]
        paired.append({"plate": p, "brand": brand})
    return paired


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


def _draw_plate_rows_bgr(im_bgr: np.ndarray, plates_ltr: list, names) -> None:
    """Draw only the plate boxes we kept (after max-cars filter)."""
    col = (0, 215, 255)
    for p in plates_ltr:
        x1, y1, x2, y2 = p["xyxy"]
        cf = p["conf"]
        cid = p.get("cls_id", 0)
        lab = _cls_name(names, cid)
        label = f"plate {lab} {cf:.2f}"
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), col, 2)
        y_txt = max(y1 - 6, 18)
        cv2.putText(
            im_bgr,
            label,
            (x1, y_txt),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            2,
            cv2.LINE_AA,
        )


def _draw_plate_car_index_badges(im_bgr: np.ndarray, plates_ltr: list) -> None:
    """Draw 1, 2, … above each plate (left → right) to match Results · Car N."""
    col = (0, 215, 255)
    for k, p in enumerate(plates_ltr, start=1):
        x1, y1, _, _ = p["xyxy"]
        cv2.putText(
            im_bgr,
            str(k),
            (x1, max(y1 - 6, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            col,
            2,
            cv2.LINE_AA,
        )


def draw_combined_detection_plot(
    frame_bgr: np.ndarray,
    brand_result,
    plate_model,
    brand_model,
    plates_ltr=None,
) -> np.ndarray:
    """Single BGR image: kept plate boxes + brand overlays; Car 1,2,… on plates."""
    canvas = frame_bgr.copy()
    h = canvas.shape[0]

    COL_BRAND = (255, 128, 0)  # BGR brand accent

    if plates_ltr:
        _draw_plate_rows_bgr(canvas, plates_ltr, plate_model.names)

    if brand_model is not None and brand_result is not None and brand_result.boxes is not None:
        _draw_detection_boxes_bgr(canvas, brand_result.boxes, brand_model.names, COL_BRAND, "brand ")

    if plates_ltr:
        _draw_plate_car_index_badges(canvas, plates_ltr)

    leg = "Plate (amber) = kept only  ·  Brand (orange)  ·  numbers = Car 1,2,… (left→right)"
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


def _plate_ocr_card_html(plate_txt: str, has_crop: bool) -> str:
    if not has_crop:
        body = (
            "<p style='margin:0;font-size:14px;color:rgba(128,132,149,0.88);'>"
            "Could not build a crop for OCR (box too small or invalid).</p>"
        )
        pill = _result_pill("warn", "No crop")
        val = "—"
        return (
            f'<div style="{_RCARD_BOX} min-height:140px;">'
            f"{_result_card_title('License plate · OCR')}"
            f'<div style="margin-bottom:10px;">{pill}</div>'
            f'<p style="margin:6px 0 8px 0;font-size:1.65rem;font-weight:800;font-family:ui-monospace,Consolas,monospace;">'
            f"{html.escape(val)}</p>{body}</div>"
        )

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
    return (
        f'<div style="{_RCARD_BOX} min-height:140px;">'
        f"{_result_card_title('License plate · OCR')}"
        f'<div style="margin-bottom:10px;">{pill}</div>'
        f'<p style="margin:6px 0 8px 0;font-size:1.85rem;font-weight:800;font-family:ui-monospace,Consolas,monospace;'
        f'letter-spacing:0.1em;line-height:1.2;">{val}</p>'
        f'<p style="margin:0;font-size:13px;line-height:1.45;color:rgba(128,132,149,0.88);">'
        f"{html.escape(sub)}</p>"
        f"</div>"
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


def render_results_section(paired_rows: list, brand_model):
    """paired_rows: from pair_plates_with_brands, each with plate_txt set on plate dict."""
    st.markdown("### Results")
    st.markdown(
        "<hr style='margin:0.4rem 0 1.2rem 0;border:none;border-top:1px solid rgba(128,132,149,0.2);' />",
        unsafe_allow_html=True,
    )

    if not paired_rows:
        st.markdown(
            f'<div style="{_RCARD_BOX} min-height:100px;">'
            f"{_result_card_title('License plate')}"
            f'<div style="margin-bottom:10px;">{_result_pill("warn", "Not detected")}</div>'
            f'<p style="margin:0;font-size:15px;line-height:1.5;color:rgba(128,132,149,0.9);">'
            "No plate regions were found on this upload.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    st.caption("Cars are ordered left → right by plate position (same as numbers on the detection image).")

    for i, row in enumerate(paired_rows, start=1):
        st.markdown(f"#### Car {i}")
        plate = row["plate"]
        brand = row["brand"]
        plate_txt = plate.get("plate_txt", "")
        has_crop = plate.get("crop") is not None

        col_p, col_b = st.columns(2, gap="large")
        with col_p:
            st.markdown(_plate_ocr_card_html(plate_txt, has_crop), unsafe_allow_html=True)
        with col_b:
            bl = brand["label"] if brand else ""
            bs = brand["score"] if brand else 0.0
            st.markdown(
                _brand_card_html(
                    brand_model is not None,
                    bl,
                    bs,
                    weights_hint="models/carbrand/best.pt",
                ),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


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
max_cars = st.sidebar.number_input(
    "Max cars (plates)",
    min_value=1,
    max_value=15,
    value=2,
    step=1,
    help="Uses the largest plate boxes in the frame (drops smaller / usually farther cars), then orders left→right.",
)

st.subheader("Upload an image")
img_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "heic", "heif"],
    key="img_uploader",
    help="iPhone photos: use .heic / .heif or convert to JPEG in Photos before upload.",
)

if img_file:
    frame = decode_upload_to_bgr(img_file)

    if frame is None:
        suf = Path(img_file.name or "").suffix.lower()
        if suf in (".heic", ".heif"):
            st.error(
                "Could not read this HEIC/HEIF file. Install **pillow-heif** "
                "(`pip install pillow-heif`, also in requirements.txt), restart the app, and try again."
            )
        else:
            st.error("Failed to read image.")
    else:
        plate_result = plate_model.predict(source=frame, conf=plate_conf, imgsz=imgsz, verbose=False)[0]
        plates_ltr, n_plates_raw = enumerate_plates_left_to_right(frame, plate_result, max_cars)
        for p in plates_ltr:
            cr = p.get("crop")
            p["plate_txt"] = run_ocr_stable(cr, ocr_engine) if cr is not None else ""

        brand_use_conf = max(0.2, det_conf)
        brand_result = None
        if brand_model is not None:
            brand_result = brand_model.predict(
                source=frame, conf=brand_use_conf, imgsz=imgsz, verbose=False
            )[0]

        brands_ltr = (
            extract_brands_left_to_right(brand_result, brand_model)
            if brand_model is not None and brand_result is not None
            else []
        )
        paired_rows = pair_plates_with_brands(plates_ltr, brands_ltr)

        combined_det = draw_combined_detection_plot(
            frame,
            brand_result,
            plate_model,
            brand_model,
            plates_ltr=plates_ltr,
        )

        n_plates = len(plates_ltr)
        if n_plates_raw:
            kept = f"{n_plates} kept" + (
                f" of {n_plates_raw} detected" if n_plates_raw != n_plates else ""
            )
            st.sidebar.caption(
                f"{kept} (left→right, max {max_cars}). OCR min crop {MIN_PLATE_CROP_W}×{MIN_PLATE_CROP_H} px."
            )
        else:
            st.sidebar.caption("No plates at this plate confidence.")

        st.subheader("Detections")
        st.image(
            combined_det[:, :, ::-1],
            caption=(
                f"Combined: plates kept (max {max_cars} by size) + brand · "
                f"plate_conf={plate_conf}, imgsz={imgsz}"
            ),
            use_column_width=True,
        )
        if brand_model is None:
            st.caption("Brand head not loaded — only plate boxes are drawn.")

        render_results_section(paired_rows, brand_model)