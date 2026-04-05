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


def _body_paint_roi_xyxy(plate_xyxy: tuple, fh: int, fw: int):
    """Band above the plate (lower body / bumper paint), image coordinates."""
    x1, y1, x2, y2 = plate_xyxy
    ph = y2 - y1
    pw = x2 - x1
    if ph <= 0 or pw <= 0:
        return None
    roi_h = int(max(ph * 2.0, 28))
    roi_w = int(pw * 1.45)
    cx = 0.5 * (x1 + x2)
    x1b = int(round(cx - roi_w / 2))
    x2b = int(round(cx + roi_w / 2))
    y2b = y1 - 3
    y1b = y2b - roi_h
    x1b = max(0, min(fw - 1, x1b))
    x2b = max(0, min(fw - 1, x2b))
    if x2b <= x1b:
        x1b, x2b = min(x1b, x2b), max(x1b, x2b) + 1
        x2b = min(fw - 1, x2b)
    y1b = max(0, min(fh - 1, y1b))
    y2b = max(0, min(fh - 1, y2b))
    if y2b <= y1b or x2b <= x1b:
        return None
    if (y2b - y1b) < 6 or (x2b - x1b) < 6:
        return None
    return (x1b, y1b, x2b, y2b)


def _hsv_center_to_color_label(h: float, s: float, v: float) -> str:
    """Map dominant HSV (OpenCV ranges) to a coarse label; order: achromatic → hue."""
    h, s, v = float(h), float(s), float(v)
    if v < 38:
        return "Black (approx)"
    if s < 38:
        if v > 195:
            return "White (approx)"
        if v > 135:
            return "Silver / light gray (approx)"
        if v > 75:
            return "Gray (approx)"
        return "Dark gray (approx)"
    if (h <= 11 or h >= 170) and v > 42:
        return "Red (approx)"
    if 11 < h <= 24:
        return "Orange (approx)"
    if 24 < h <= 40:
        return "Yellow (approx)"
    if 40 < h <= 88:
        return "Green (approx)"
    if 88 < h <= 130:
        return "Blue (approx)"
    if 130 < h < 170:
        return "Purple (approx)"
    return "Other (approx)"


def estimate_body_color_from_plate(frame_bgr: np.ndarray, plate_xyxy: tuple) -> tuple:
    """
    OpenCV: ROI above plate → HSV → k-means dominant cluster → rule-based name.
    Returns (label, sample_bgr) for UI swatch; sample_bgr may be None.
    """
    fh, fw = frame_bgr.shape[:2]
    roi_box = _body_paint_roi_xyxy(plate_xyxy, fh, fw)
    if roi_box is None:
        return "Unknown", None
    x1b, y1b, x2b, y2b = roi_box
    roi = frame_bgr[y1b:y2b, x1b:x2b]
    if roi is None or roi.size < 30:
        return "Unknown", None

    mh = max(roi.shape[0], roi.shape[1])
    if mh > 220:
        scale = 220.0 / mh
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    n = pixels.shape[0]
    if n < 12:
        return "Unknown", None

    pixels = np.ascontiguousarray(pixels)

    v_ch = pixels[:, 2]
    mask = (v_ch > 14) & (v_ch < 253)
    if np.count_nonzero(mask) > n * 0.15:
        pixels = pixels[mask]
        n = pixels.shape[0]
    if n < 8:
        return "Unknown", None

    k = 3 if n >= 60 else (2 if n >= 25 else 1)
    if k == 1:
        ch, cs, cv = np.mean(pixels, axis=0)
        label = _hsv_center_to_color_label(ch, cs, cv)
        samp = cv2.cvtColor(
            np.uint8([[[int(round(ch)), int(round(cs)), int(round(cv))]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        return label, (int(samp[0]), int(samp[1]), int(samp[2]))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    attempts = 3
    try:
        _comp, lbls, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            attempts,
            cv2.KMEANS_PP_CENTERS,
        )
    except Exception:
        ch, cs, cv = np.mean(pixels, axis=0)
        label = _hsv_center_to_color_label(ch, cs, cv)
        samp = cv2.cvtColor(
            np.uint8([[[int(round(ch)), int(round(cs)), int(round(cv))]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        return label, (int(samp[0]), int(samp[1]), int(samp[2]))

    labels = lbls.flatten().astype(int)
    counts = np.bincount(labels, minlength=k)
    dom = int(np.argmax(counts))
    ch, cs, cv = centers[dom]
    label = _hsv_center_to_color_label(ch, cs, cv)
    samp = cv2.cvtColor(
        np.uint8([[[int(round(ch)), int(round(cs)), int(round(cv))]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    return label, (int(samp[0]), int(samp[1]), int(samp[2]))


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


def _draw_body_color_captions(im_bgr: np.ndarray, plates_ltr: list) -> None:
    """Short color label under each plate box (if estimated)."""
    col = (220, 230, 255)
    for p in plates_ltr:
        lab = p.get("body_color", "") or ""
        if not lab or lab == "Unknown":
            continue
        x1, _, _, y2 = p["xyxy"]
        short = lab.replace(" (approx)", "")
        if "/" in short:
            short = short.split("/")[0].strip()
        short = short[:22]
        y = min(y2 + 16, im_bgr.shape[0] - 4)
        cv2.putText(
            im_bgr,
            short,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            col,
            1,
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
        _draw_body_color_captions(canvas, plates_ltr)

    leg = "Plate (amber)  ·  Brand (orange)  ·  # = Car 1,2…  ·  text under plate = color (approx)"
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

# Results: fixed table columns + equal card height (Streamlit-safe alignment)
_RESULT_CARD_MIN_PX = 300
_RCARD_CELL = (
    f"{_RCARD_BOX} min-height:{_RESULT_CARD_MIN_PX}px;height:100%;width:100%;max-width:100%;"
    + "display:flex;flex-direction:column;align-items:stretch;box-sizing:border-box;"
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
            "<p style='margin:0;font-size:14px;line-height:1.45;color:rgba(128,132,149,0.88);'>"
            "Could not build a crop for OCR (box too small or invalid).</p>"
        )
        pill = _result_pill("warn", "No crop")
        val = "—"
        return (
            f'<div style="{_RCARD_CELL}">'
            f"{_result_card_title('License plate · OCR')}"
            f'<div style="margin-bottom:10px;">{pill}</div>'
            f'<p style="margin:0 0 8px 0;font-size:1.65rem;font-weight:800;font-family:ui-monospace,Consolas,monospace;'
            f'word-break:break-word;overflow-wrap:anywhere;">'
            f"{html.escape(val)}</p>"
            f'<div style="flex:1;min-height:8px;"></div>'
            f'<div style="margin-top:auto;">{body}</div></div>'
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
        f'<div style="{_RCARD_CELL}">'
        f"{_result_card_title('License plate · OCR')}"
        f'<div style="margin-bottom:10px;">{pill}</div>'
        f'<p style="margin:0 0 8px 0;font-size:1.85rem;font-weight:800;font-family:ui-monospace,Consolas,monospace;'
        f'letter-spacing:0.1em;line-height:1.2;word-break:break-word;overflow-wrap:anywhere;">{val}</p>'
        f'<div style="flex:1;min-height:8px;"></div>'
        f'<p style="margin:0;margin-top:auto;font-size:13px;line-height:1.45;color:rgba(128,132,149,0.88);">'
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
    grow = '<div style="flex:1;min-height:4px;"></div>'
    if not deployed:
        pill = _result_pill("muted", "Unavailable")
        hint = (
            f" Missing weights: <code style='font-size:12px;'>{html.escape(weights_hint)}</code>."
            if weights_hint
            else ""
        )
        main = (
            "<p style='margin:0;font-size:14px;line-height:1.5;color:rgba(128,132,149,0.88);'>"
            f"Model not deployed for this app.{hint}</p>"
        )
        inner = f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center;min-height:0;">{main}</div>'
    elif not label:
        pill = _result_pill("warn", "No detection")
        main = (
            "<p style='margin:0;font-size:14px;line-height:1.45;color:rgba(128,132,149,0.88);'>"
            "Nothing confident enough to report for this image.</p>"
        )
        inner = f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center;min-height:0;">{main}</div>'
    else:
        pill = _result_pill("ok", "Detected")
        esc = html.escape(label)
        main = (
            f"<p style='margin:0 0 6px 0;font-size:1.35rem;font-weight:750;line-height:1.3;"
            f"word-break:break-word;overflow-wrap:anywhere;'>{esc}</p>"
            f"<p style='margin:0;font-size:13px;color:rgba(128,132,149,0.85);'>"
            f"Confidence · {score:.2f}</p>"
        )
        inner = f'<div style="flex:1;display:flex;flex-direction:column;min-height:0;">{main}{grow}</div>'
    return (
        f'<div style="{_RCARD_CELL}">{head}'
        f'<div style="margin-bottom:10px;">{pill}</div>'
        f"{inner}</div>"
    )


def _body_color_card_html(label: str) -> str:
    hint = "OpenCV: Lightning sensitive"
    if label and label != "Unknown":
        pill = _result_pill("ok", "Estimate")
    else:
        pill = _result_pill("warn", "Unknown")
    esc = html.escape(label or "Unknown")
    return (
        f'<div style="{_RCARD_CELL}">'
        f"{_result_card_title('Body color (OpenCV)')}"
        f'<div style="margin-bottom:10px;">{pill}</div>'
        f'<p style="margin:0 0 4px 0;font-size:1.15rem;font-weight:750;line-height:1.35;'
        f'word-break:break-word;overflow-wrap:anywhere;">{esc}</p>'
        f'<div style="flex:1;min-height:8px;"></div>'
        f'<p style="margin:0;margin-top:auto;font-size:12px;line-height:1.4;color:rgba(128,132,149,0.85);">'
        f"{html.escape(hint)}</p>"
        f"</div>"
    )


def _build_results_table_html(paired_rows: list, brand_model) -> str:
    """One HTML table: fixed 34/33/33 columns, aligned across all cars (avoids Streamlit grid bugs)."""
    col_pad = (
        "padding:0 10px 0 0;border:none;vertical-align:top;width:34%;",
        "padding:0 10px;border:none;vertical-align:top;width:33%;",
        "padding:0 0 0 10px;border:none;vertical-align:top;width:33%;",
    )
    chunks = [
        '<div style="width:100%;max-width:100%;box-sizing:border-box;">',
        '<table role="presentation" style="width:100%;max-width:100%;table-layout:fixed;'
        "border-collapse:collapse;border:none;margin:0;padding:0;">",
        "<colgroup>",
        '<col style="width:34%" />',
        '<col style="width:33%" />',
        '<col style="width:33%" />',
        "</colgroup>",
        "<tbody>",
    ]
    n_cars = len(paired_rows)
    for i, row in enumerate(paired_rows, start=1):
        plate = row["plate"]
        brand = row["brand"]
        plate_txt = plate.get("plate_txt", "")
        has_crop = plate.get("crop") is not None
        bl = brand["label"] if brand else ""
        bs = brand["score"] if brand else 0.0
        bc = plate.get("body_color", "Unknown")
        ph = _plate_ocr_card_html(plate_txt, has_crop)
        bh = _brand_card_html(
            brand_model is not None,
            bl,
            bs,
            weights_hint="models/carbrand/best.pt",
        )
        ch = _body_color_card_html(bc)
        chunks.append(
            f'<tr><td colspan="3" style="padding:22px 0 12px 0;border:none;">'
            f'<div style="font-size:1.08rem;font-weight:650;letter-spacing:0.03em;'
            f'color:inherit;border-bottom:1px solid rgba(128,132,149,0.28);'
            f'padding-bottom:10px;margin:0;">Car {i}</div></td></tr>'
        )
        chunks.append('<tr style="vertical-align:top;">')
        for cell_html, cst in zip((ph, bh, ch), col_pad):
            chunks.append(
                f'<td style="{cst}">'
                f'<div style="display:flex;align-items:stretch;min-height:{_RESULT_CARD_MIN_PX}px;'
                f'height:100%;width:100%;box-sizing:border-box;">{cell_html}</div>'
                f"</td>"
            )
        chunks.append("</tr>")
        if i < n_cars:
            chunks.append(
                '<tr><td colspan="3" style="padding:0;height:16px;border:none;font-size:0;line-height:0;">'
                "&nbsp;</td></tr>"
            )
    chunks.append("</tbody></table></div>")
    return "".join(chunks)


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

    st.markdown(_build_results_table_html(paired_rows, brand_model), unsafe_allow_html=True)


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
            clab, cbgr = estimate_body_color_from_plate(frame, p["xyxy"])
            p["body_color"] = clab
            p["body_color_bgr"] = cbgr

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