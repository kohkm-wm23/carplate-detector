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

#decode upload file to bgr
def decode_upload_to_bgr(uploaded_file) -> np.ndarray | None:
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

#Configuration
st.set_page_config(page_title="Car Plate + Brand (YOLOv8)", layout="wide")
st.title("Car Plate Detection System (Yolov8 + Brand)")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

PLATE_MODEL_PATH = MODELS_DIR / "carplate" / "best.pt"
BRAND_MODEL_PATH = MODELS_DIR / "carbrand" / "best.pt"
CAR_MODEL_PATH = MODELS_DIR / "carobject" / "best.pt"

PROJECT_OUT = BASE_DIR / "outputs"
PROJECT_OUT.mkdir(exist_ok=True)
METRICS_IMAGES = [
    ("Car Plate Metrics", PROJECT_OUT / "metrics_car_plate.png"),
    ("Car Object Metrics", PROJECT_OUT / "metrics_car_object.png"),
    ("Brand Logo Metrics", PROJECT_OUT / "metrics_brand_logo.png"),
]

PLATE_REGEX = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$")

MIN_PLATE_CROP_W = 90
MIN_PLATE_CROP_H = 25
_PAD_X_STD = 0.03
_PAD_Y_STD = 0.08
_PAD_X_EXPAND = 0.10
_PAD_Y_EXPAND = 0.20


#Model + OCR loaders
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

#preprocess for ocr grayscale bgr
def preprocess_for_ocr_grayscale_bgr(crop_bgr: np.ndarray):
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

#extract padded crop
def _extract_padded_crop(frame_bgr, x1, y1, x2, y2, fh, fw, pad_x_frac: float, pad_y_frac: float):
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

#crop for single plate box
def _crop_for_single_plate_box(frame_bgr, x1, y1, x2, y2, fh, fw):
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

#enumerate plates left to right
def enumerate_plates_left_to_right(frame_bgr: np.ndarray, plate_result, max_cars: int) -> tuple:
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


def _iou_xyxy(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    den = max(1.0, area_a + area_b - inter)
    return inter / den


def _dedupe_overlapping_cars(rows: list, iou_thr: float = 0.55) -> list:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: (r["conf"], r["area"]), reverse=True)
    kept = []
    for r in ordered:
        if all(_iou_xyxy(r["xyxy"], k["xyxy"]) < iou_thr for k in kept):
            kept.append(r)
    return kept

#enumerate cars left to right
def enumerate_cars_left_to_right(frame_bgr: np.ndarray, car_result, max_cars: int) -> tuple:
    boxes = None if car_result is None else car_result.boxes
    if boxes is None or len(boxes) == 0:
        return [], 0
    xyxy_all = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    rows = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in xyxy_all[i]]
        if x2 <= x1 or y2 <= y1:
            continue
        area = float((x2 - x1) * (y2 - y1))
        rows.append(
            {
                "x_center": 0.5 * (x1 + x2),
                "xyxy": (x1, y1, x2, y2),
                "conf": float(confs[i]),
                "area": area,
                "cls_id": int(clss[i]),
            }
        )
    n_raw = len(rows)
    rows = _dedupe_overlapping_cars(rows, iou_thr=0.55)
    mc = max(1, int(max_cars))
    if len(rows) > mc:
        rows.sort(key=lambda r: r["area"], reverse=True)
        rows = rows[:mc]
    rows.sort(key=lambda r: r["x_center"])
    return rows, n_raw

#extract brands left to right
def extract_brands_left_to_right(brand_result, brand_model) -> list:
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

#hsv center to single color label
def _hsv_center_to_single_color_label(h: float, s: float, v: float) -> str:
    h, s, v = float(h), float(s), float(v)
    if v < 38:
        return "Black"
    if s < 38:
        if v > 200:
            return "White"
        if v > 125:
            return "Gray"
        return "Dark gray"
    if h <= 10 or h >= 170:
        return "Red"
    if h <= 22:
        return "Orange"
    if h <= 35:
        return "Yellow"
    if h <= 85:
        return "Green"
    if h <= 130:
        return "Blue"
    if h <= 160:
        return "Purple"
    return "Brown"

#estimate body color from car crop
def estimate_body_color_from_car_crop(car_crop_bgr: np.ndarray) -> tuple:
    if car_crop_bgr is None or car_crop_bgr.size < 30:
        return "Unknown", 0.0, None
    roi = car_crop_bgr
    mh = max(roi.shape[0], roi.shape[1])
    if mh > 260:
        scale = 260.0 / mh
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Focus on central body region to reduce road/background bias.
    rh, rw = roi.shape[:2]
    y1, y2 = int(rh * 0.20), int(rh * 0.85)
    x1, x2 = int(rw * 0.10), int(rw * 0.90)
    core = roi[y1:y2, x1:x2]
    if core is not None and core.size > 30:
        roi = core

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    n0 = pixels.shape[0]
    if n0 < 20:
        return "Unknown", 0.0, None

    # Drop very dark/bright extremes to reduce shadow/highlight bias.
    v_ch = pixels[:, 2]
    s_ch = pixels[:, 1]
    mask = (v_ch > 20) & (v_ch < 245) & (s_ch > 10)
    if np.count_nonzero(mask) > n0 * 0.20:
        pixels = pixels[mask]
    n = pixels.shape[0]
    if n < 12:
        return "Unknown", 0.0, None

    k = 4 if n >= 100 else (3 if n >= 50 else 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
    try:
        _comp, lbls, centers = cv2.kmeans(
            np.ascontiguousarray(pixels),
            k,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = lbls.flatten().astype(int)
        counts = np.bincount(labels, minlength=k)
        dom = int(np.argmax(counts))
        pct = 100.0 * float(counts[dom]) / max(1.0, float(np.sum(counts)))
        ch, cs, cv = centers[dom]
    except Exception:
        ch, cs, cv = np.mean(pixels, axis=0)
        pct = 100.0

    label = _hsv_center_to_single_color_label(ch, cs, cv)
    samp = cv2.cvtColor(
        np.uint8([[[int(round(ch)), int(round(cs)), int(round(cv))]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    return label, pct, (int(samp[0]), int(samp[1]), int(samp[2]))


def _cls_name(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, str(cls_id)))
    if 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _draw_plate_rows_bgr(im_bgr: np.ndarray, plates_ltr: list, names) -> None:
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


def _draw_car_rows_bgr(im_bgr: np.ndarray, cars_ltr: list, names) -> None:
    col = (240, 110, 30)
    for i, c in enumerate(cars_ltr, start=1):
        x1, y1, x2, y2 = c["xyxy"]
        cf = c["conf"]
        cid = c.get("cls_id", 0)
        lab = _cls_name(names, cid)
        label = f"car {lab} {cf:.2f}"
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            im_bgr,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            col,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            im_bgr,
            str(i),
            (x1 + 4, min(y2 - 8, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.84,
            col,
            2,
            cv2.LINE_AA,
        )


def _draw_brand_rows_bgr(im_bgr: np.ndarray, brand_rows: list) -> None:
    col = (255, 128, 0)
    for b in brand_rows:
        x1, y1, x2, y2 = b["xyxy"]
        label = f'brand {b["label"]} {b["score"]:.2f}'
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            im_bgr,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            col,
            2,
            cv2.LINE_AA,
        )


def draw_car_first_detection_plot(
    frame_bgr: np.ndarray,
    car_rows: list,
    plate_model,
    car_model,
    n_cars_raw: int = 0,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    h = canvas.shape[0]

    if car_rows:
        cars_ltr = [r["car"] for r in car_rows]
        _draw_car_rows_bgr(canvas, cars_ltr, car_model.names)

    plates = []
    for r in car_rows:
        p = r["plate"]
        if p.get("xyxy") is not None:
            plates.append(p)
    if plates:
        _draw_plate_rows_bgr(canvas, plates, plate_model.names)
        _draw_body_color_captions(canvas, plates)

    brands = []
    for r in car_rows:
        b = r.get("brand")
        if b is not None:
            brands.append(b)
    if brands:
        _draw_brand_rows_bgr(canvas, brands)

    n_kept = len(car_rows)
    n_raw = int(n_cars_raw) if n_cars_raw else 0
    count_txt = f"Cars detected: {n_raw if n_raw > 0 else n_kept}"
    if n_cars_raw > 0 and n_kept != n_cars_raw:
        count_txt += f" (showing {n_kept})"
    cv2.putText(
        canvas,
        count_txt,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    leg = "Car (orange) · Brand (blue) · Plate (amber) · # = Car 1,2,... · under plate = color"
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

#fixed table columns + equal card height
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
    title = "Car brand / model"
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


def _body_color_card_html(label: str, pct: float = 0.0) -> str:
    hint = "Dominant color from detected car object"
    if label and label != "Unknown":
        pill = _result_pill("ok", "Estimate")
    else:
        pill = _result_pill("warn", "Unknown")
    shown = label or "Unknown"
    if label and label != "Unknown" and pct > 0:
        shown = f"{label} ({pct:.0f}%)"
    esc = html.escape(shown)
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
    col_pad = (
        "padding:0 10px 0 0;border:none!important;border-width:0!important;"
        "vertical-align:top;width:34%;",
        "padding:0 10px;border:none!important;border-width:0!important;"
        "vertical-align:top;width:33%;",
        "padding:0 0 0 10px;border:none!important;border-width:0!important;"
        "vertical-align:top;width:33%;",
    )
    chunks = [
        '<div class="cdr-results-table-wrap" style="width:100%;max-width:100%;box-sizing:border-box;">',
        "<style>"
        ".cdr-results-table-wrap table,.cdr-results-table-wrap tbody,.cdr-results-table-wrap tr,"
        ".cdr-results-table-wrap td,.cdr-results-table-wrap th{"
        "border:none!important;border-width:0!important;border-style:none!important;"
        "outline:none!important;box-shadow:none!important;background:transparent!important;}"
        ".cdr-results-table-wrap table{border-spacing:0!important;border-collapse:collapse!important;}"
        ".cdr-results-table-wrap td+td,.cdr-results-table-wrap tr+tr td{border-left:none!important;"
        "border-top:none!important;}"
        "</style>",
        '<table role="presentation" style="width:100%;max-width:100%;table-layout:fixed;'
        "border-collapse:collapse!important;border-spacing:0!important;border:none!important;"
        'margin:0;padding:0;">',
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
        bp = float(plate.get("body_color_pct", 0.0) or 0.0)
        ph = _plate_ocr_card_html(plate_txt, has_crop)
        bh = _brand_card_html(
            brand_model is not None,
            bl,
            bs,
            weights_hint="models/carbrand/best.pt",
        )
        ch = _body_color_card_html(bc, bp)
        chunks.append(
            f'<tr><td colspan="3" style="padding:22px 0 12px 0;border:none!important;border-width:0!important;">'
            f'<div style="font-size:1.08rem;font-weight:650;letter-spacing:0.03em;'
            f'color:inherit;padding-bottom:4px;margin:0;">Car {i}</div></td></tr>'
        )
        chunks.append(
            '<tr style="vertical-align:top;border:none!important;border-width:0!important;">'
        )
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
                '<tr><td colspan="3" style="padding:0;height:16px;border:none!important;border-width:0!important;'
                'font-size:0;line-height:0;">'
                "&nbsp;</td></tr>"
            )
    chunks.append("</tbody></table></div>")
    return "".join(chunks)


def render_results_section(paired_rows: list, brand_model, n_cars_raw: int = 0):
    st.markdown("### Results")
    st.markdown(
        "<hr style='margin:0.4rem 0 1.2rem 0;border:none;border-top:1px solid rgba(128,132,149,0.2);' />",
        unsafe_allow_html=True,
    )
    n_rows = len(paired_rows)
    if n_cars_raw > 0:
        st.markdown(
            (
                '<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;'
                'margin:0 0 8px 0;">'
                f'<div style="{_RCARD_BOX} padding:0.75rem 1rem;">'
                f'{_result_card_title("Total cars detected")}'
                f'<p style="margin:0;font-size:1.5rem;font-weight:800;line-height:1.2;">{n_cars_raw}</p>'
                '</div>'
                f'<div style="{_RCARD_BOX} padding:0.75rem 1rem;">'
                f'{_result_card_title("Showing")}'
                f'<p style="margin:0;font-size:1.5rem;font-weight:800;line-height:1.2;">{n_rows}</p>'
                '</div>'
                '</div>'
                '<p style="margin:0 0 10px 0;font-size:12px;line-height:1.4;'
                'color:rgba(128,132,149,0.92);">Order from left to right.</p>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    if not paired_rows:
        st.markdown(
            f'<div style="{_RCARD_BOX} min-height:100px;">'
            f"{_result_card_title('Car objects')}"
            f'<div style="margin-bottom:10px;">{_result_pill("warn", "Not detected")}</div>'
            f'<p style="margin:0;font-size:15px;line-height:1.5;color:rgba(128,132,149,0.9);">'
            "No car objects were found on this upload.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(_build_results_table_html(paired_rows, brand_model), unsafe_allow_html=True)


def render_model_comparison_tab():
    st.markdown("### Model Comparison")
    st.caption("Static performance charts for YOLOv8 vs RetinaNet vs Faster R-CNN.")

    if not any(path.exists() for _, path in METRICS_IMAGES):
        st.info(
            "No comparison graphs found in `outputs/`. Run `python resultGraph.py` to generate them."
        )
        return

    top_left, top_right = st.columns(2)
    for idx, (title, image_path) in enumerate(METRICS_IMAGES[:2]):
        with (top_left if idx == 0 else top_right):
            st.markdown(f"#### {title}")
            if image_path.exists():
                st.image(str(image_path), use_column_width=True)
            else:
                st.warning(f"Missing graph: `{image_path.name}`")

    st.markdown("#### Full Comparison Snapshot")
    last_title, last_path = METRICS_IMAGES[2]
    if last_path.exists():
        st.image(str(last_path), use_column_width=True, caption=last_title)
    else:
        st.warning(f"Missing graph: `{last_path.name}`")


#Load models + UI
try:
    car_model = load_car_model()
    plate_model = load_plate_model()
    brand_model = load_brand_model()
    ocr_engine = load_ocr()
except Exception as e:
    st.error(f"Model/OCR load failed: {e}")
    st.stop()

st.sidebar.header("Settings")
car_conf = st.sidebar.slider(
    "Car object confidence",
    0.0,
    1.0,
    0.25,
    0.05,
    help="YOLO threshold for the car-object model (first stage).",
)
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
    "Max cars (objects)",
    min_value=1,
    max_value=15,
    value=2,
    step=1,
    help="Keeps largest car boxes in the frame, then orders left→right.",
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
        if car_model is None:
            st.error("Car-object model not found at `models/carobject/best.pt`.")
            st.stop()

        car_result = car_model.predict(source=frame, conf=car_conf, imgsz=imgsz, verbose=False)[0]
        cars_ltr, n_cars_raw = enumerate_cars_left_to_right(frame, car_result, max_cars)

        paired_rows = []
        for c in cars_ltr:
            x1, y1, x2, y2 = c["xyxy"]
            car_crop = frame[y1:y2, x1:x2]
            if car_crop is None or car_crop.size == 0:
                continue

            plate = {
                "x_center": c["x_center"],
                "xyxy": None,
                "crop": None,
                "conf": 0.0,
                "area": 0.0,
                "cls_id": 0,
                "plate_txt": "",
            }
            plate_result = plate_model.predict(source=car_crop, conf=plate_conf, imgsz=imgsz, verbose=False)[0]
            local_plates, _ = enumerate_plates_left_to_right(car_crop, plate_result, 1)
            if local_plates:
                lp = local_plates[0]
                lx1, ly1, lx2, ly2 = lp["xyxy"]
                plate["xyxy"] = (x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2)
                plate["crop"] = lp.get("crop")
                plate["conf"] = lp.get("conf", 0.0)
                plate["area"] = lp.get("area", 0.0)
                plate["cls_id"] = lp.get("cls_id", 0)
                plate["plate_txt"] = (
                    run_ocr_stable(plate["crop"], ocr_engine) if plate["crop"] is not None else ""
                )

            brand = None
            if brand_model is not None:
                brand_result_car = brand_model.predict(
                    source=car_crop, conf=max(0.2, det_conf), imgsz=imgsz, verbose=False
                )[0]
                brands_local = extract_brands_left_to_right(brand_result_car, brand_model)
                if brands_local:
                    # Prefer confident, sufficiently large logos/features to reduce noisy tiny hits.
                    def _brand_rank(b):
                        bx1, by1, bx2, by2 = b["xyxy"]
                        b_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
                        c_area = max(1.0, float((x2 - x1) * (y2 - y1)))
                        area_ratio = b_area / c_area
                        return b["score"] * (0.8 + min(0.6, area_ratio * 8.0))

                    best_brand = max(brands_local, key=_brand_rank)
                    bx1, by1, bx2, by2 = best_brand["xyxy"]
                    brand = {
                        "x_center": c["x_center"],
                        "label": best_brand["label"],
                        "score": best_brand["score"],
                        "xyxy": (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2),
                    }

            clab, cpct, cbgr = estimate_body_color_from_car_crop(car_crop)
            plate["body_color"] = clab
            plate["body_color_bgr"] = cbgr
            plate["body_color_pct"] = cpct
            paired_rows.append({"car": c, "plate": plate, "brand": brand})

        combined_det = draw_car_first_detection_plot(
            frame,
            paired_rows,
            plate_model,
            car_model,
            n_cars_raw=n_cars_raw,
        )

        n_shown = len(paired_rows)
        if n_cars_raw:
            kept = f"{n_shown} kept" + (
                f" of {n_cars_raw} detected" if n_cars_raw != n_shown else ""
            )
            st.sidebar.caption(
                f"{kept} (left→right, max {max_cars}). OCR min crop {MIN_PLATE_CROP_W}×{MIN_PLATE_CROP_H} px."
            )
        else:
            st.sidebar.caption("No cars at this car confidence.")

        tab_results, tab_comparison = st.tabs(["Detection & Results", "Model Comparison"])

        with tab_results:
            st.subheader("Detections")
            st.image(
                combined_det[:, :, ::-1],
                caption=(
                    f"Combined: car objects + per-car plate/brand/color · "
                    f"car_conf={car_conf}, plate_conf={plate_conf}, imgsz={imgsz}"
                ),
                use_column_width=True,
            )
            if brand_model is None:
                st.caption("Brand head not loaded — car + plate + color are still shown.")

            render_results_section(paired_rows, brand_model, n_cars_raw=n_cars_raw)

        with tab_comparison:
            render_model_comparison_tab()