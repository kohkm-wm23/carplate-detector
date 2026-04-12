"""
Generate three bar charts (plate / car object / brand logo).
Layout: X-axis = metrics; grouped bars = one color per model (YOLOv8, RetinaNet, Faster R-CNN).

YOLOv8 / Faster R-CNN: test-set metrics as below.
RetinaNet column: teammate eval (eval_plate.py / eval_car.py / eval_brand.py). F1 = 2PR/(P+R); 0 if P+R=0.

Run: python resultGraph.py
Outputs: outputs/metrics_car_plate.png, metrics_car_object.png, metrics_brand_logo.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_OUT_DIR = Path(__file__).resolve().parent / "outputs"
_MODELS = ["YOLOv8", "RetinaNet", "Faster R-CNN"]
# Distinct colors per model (similar idea to grouped comparison charts)
_MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

_METRIC_LABELS = [
    "Precision",
    "Recall",
    "F1-score",
    "mAP@0.5",
]


def _save_metrics_bar_chart(
    *,
    out_name: str,
    title: str,
    models: list[str],
    precision: list[float],
    recall: list[float],
    f1_score: list[float],
    map50: list[float],
) -> Path:
    n_models = len(models)
    if not (n_models == len(precision) == len(recall) == len(f1_score) == len(map50)):
        raise ValueError("models and all metric lists must have the same length")

    # rows[model_idx][metric_idx]
    rows = [
        [precision[i], recall[i], f1_score[i], map50[i]] for i in range(n_models)
    ]

    n_metrics = len(_METRIC_LABELS)
    x = np.arange(n_metrics, dtype=float)
    # Bar width so three bars fit under each metric tick
    bar_w = 0.22
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_w

    fig, ax = plt.subplots(figsize=(10, 6))

    for mi, model in enumerate(models):
        heights = rows[mi]
        bars = ax.bar(
            x + offsets[mi],
            heights,
            bar_w,
            label=model,
            color=_MODEL_COLORS[mi % len(_MODEL_COLORS)],
            edgecolor="white",
            linewidth=0.6,
        )
        for b in bars:
            h = b.get_height()
            if h <= 0:
                continue
            lbl = f"{h:.4f}" if h < 0.01 else f"{h:.2f}"
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h + 0.015,
                lbl,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x, _METRIC_LABELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# Order per list: YOLOv8, RetinaNet, Faster R-CNN
# RetinaNet: plate 120 img | car 23 img | brand 100 img (eval scripts). F1 computed as 2*P*R/(P+R).

PLATE_PRECISION = [0.956, 0.9531, 0.7233]
PLATE_RECALL = [0.937, 0.7667, 0.7651]
PLATE_F1 = [0.946, 0.8498, 0.7437]
PLATE_MAP50 = [0.965, 0.9531, 0.9347]

CAR_OBJECT_PRECISION = [0.996, 0.9553, 0.9213]
CAR_OBJECT_RECALL = [0.926, 0.9111, 0.9333]
CAR_OBJECT_F1 = [0.960, 0.9327, 0.9273]
CAR_OBJECT_MAP50 = [0.99, 0.9553, 0.9960]

BRAND_PRECISION = [0.925, 0.8476, 0.8069]
BRAND_RECALL = [0.869, 0.8194, 0.8417]
BRAND_F1 = [0.896, 0.8333, 0.8239]
BRAND_MAP50 = [0.955, 0.8476, 0.9673]


def main() -> None:
    charts = [
        {
            "out_name": "metrics_car_plate.png",
            "title": "License plate detection — model comparison (test set)",
            "precision": PLATE_PRECISION,
            "recall": PLATE_RECALL,
            "f1_score": PLATE_F1,
            "map50": PLATE_MAP50,
        },
        {
            "out_name": "metrics_car_object.png",
            "title": "Car object detection — model comparison",
            "precision": CAR_OBJECT_PRECISION,
            "recall": CAR_OBJECT_RECALL,
            "f1_score": CAR_OBJECT_F1,
            "map50": CAR_OBJECT_MAP50,
        },
        {
            "out_name": "metrics_brand_logo.png",
            "title": "Car brand (logo) detection — model comparison",
            "precision": BRAND_PRECISION,
            "recall": BRAND_RECALL,
            "f1_score": BRAND_F1,
            "map50": BRAND_MAP50,
        },
    ]

    for c in charts:
        _save_metrics_bar_chart(
            out_name=c["out_name"],
            title=c["title"],
            models=_MODELS,
            precision=c["precision"],
            recall=c["recall"],
            f1_score=c["f1_score"],
            map50=c["map50"],
        )

    print(f"Done. {len(charts)} charts in {_OUT_DIR}")


if __name__ == "__main__":
    main()
