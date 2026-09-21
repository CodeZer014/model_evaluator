"""
core.py
=======
Shared engine behind evaluate_model.py (CLI) and evaluate_app.py (Streamlit
upload UI). Everything that isn't "how do I get the file" lives here:

  load_model(path, model_key=None)   -> ModelWrapper, meta
  guess_label_column / guess_feature_columns
  run_evaluation(wrapper, X, y_true, ...) -> EvalResult
  render_html_report(result, ...)    -> self-contained HTML string

Supported model files out of the box:
  .joblib / .pkl / .pickle   scikit-learn estimator or Pipeline, OR a dict
                              that contains one (e.g. {"focus_area": pipe,
                              "features": [...]}) as v1-v3 in this project
                              save them.
  .h5 / .keras / SavedModel  Keras/TensorFlow (needs `tensorflow` installed)
  .tflite                    TensorFlow Lite (needs `tensorflow` installed)
  .pt / .pth                 A pickled full torch.nn.Module, not a bare
                              state_dict (needs `torch` installed)
  .onnx                      ONNX Runtime (needs `onnxruntime` installed)

The optional formats import their library lazily, so this file works with
only scikit-learn/joblib/pandas/numpy/matplotlib installed.
"""
from __future__ import annotations

import base64
import io
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                              balanced_accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_recall_curve, precision_recall_fscore_support,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

COMMON_LABEL_NAMES = ("target", "label", "labels", "y", "class", "focus_area", "outcome")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class ModelWrapper:
    """Normalizes .predict()/.predict_proba() across model formats."""

    def __init__(self, kind: str, model: Any, class_names=None):
        self.kind = kind
        self.model = model
        self.class_names = list(class_names) if class_names is not None else None

    # -- public API ---------------------------------------------------
    def classes(self, y_true=None, y_pred=None):
        if self.class_names is not None:
            return self.class_names
        if self.kind == "sklearn":
            model = self.model
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                for step in reversed(list(model.named_steps.values())):
                    if hasattr(step, "classes_"):
                        classes = step.classes_
                        break
            if classes is not None:
                self.class_names = list(classes)
                return self.class_names
        fallback = sorted(set(list(y_true or [])) | set(list(y_pred or [])))
        return fallback

    def predict(self, X):
        if self.kind == "sklearn":
            return np.asarray(self.model.predict(X))
        proba = self.predict_proba(X)
        if proba is None:
            raise RuntimeError(f"{self.kind} model exposes no way to get predictions")
        return self._proba_to_labels(proba)

    def predict_proba(self, X):
        if self.kind == "sklearn":
            if hasattr(self.model, "predict_proba"):
                return np.asarray(self.model.predict_proba(X))
            return None
        if self.kind == "keras":
            return np.asarray(self.model.predict(np.asarray(X, dtype="float32"), verbose=0))
        if self.kind == "tflite":
            return self._tflite_predict_proba(X)
        if self.kind == "torch":
            import torch
            self.model.eval()
            with torch.no_grad():
                out = self.model(torch.as_tensor(np.asarray(X, dtype=np.float32)))
                if out.ndim > 1 and out.shape[-1] > 1:
                    proba = torch.softmax(out, dim=-1).numpy()
                else:
                    proba = torch.sigmoid(out).reshape(-1, 1).numpy()
            return proba
        if self.kind == "onnx":
            input_name = self.model.get_inputs()[0].name
            out = self.model.run(None, {input_name: np.asarray(X, dtype=np.float32)})[0]
            return np.asarray(out)
        return None

    # -- internals ------------------------------------------------------
    def _proba_to_labels(self, proba):
        proba = np.asarray(proba)
        if proba.ndim == 1 or proba.shape[-1] == 1:
            idx = (proba.reshape(-1) >= 0.5).astype(int)
        else:
            idx = proba.argmax(axis=-1)
        if self.class_names is not None:
            names = np.asarray(self.class_names)
            return names[idx]
        return idx

    def _tflite_predict_proba(self, X):
        interpreter = self.model
        in_detail = interpreter.get_input_details()[0]
        out_detail = interpreter.get_output_details()[0]
        X = np.asarray(X, dtype=np.float32)
        rows = []
        for row in X:
            row = row.reshape(in_detail["shape"]).astype(in_detail["dtype"])
            interpreter.set_tensor(in_detail["index"], row)
            interpreter.invoke()
            rows.append(np.asarray(interpreter.get_tensor(out_detail["index"])).reshape(-1))
        return np.array(rows)


def _unwrap_sklearn_object(obj, model_key=None):
    """Handle a bare estimator/Pipeline OR a dict that contains one.

    This project's joblib files look like:
        {"focus_area": pipe, "features": [...], "trained_rows": N, ...}
    so unwrapping a dict is the common case here, not an edge case.
    """
    if hasattr(obj, "predict"):
        return obj, {}

    if isinstance(obj, dict):
        candidates = {k: v for k, v in obj.items() if hasattr(v, "predict")}
        if not candidates:
            raise ValueError(
                "Loaded a dict but found no estimator inside it (nothing with "
                f".predict). Keys present: {list(obj.keys())}"
            )
        if model_key:
            if model_key not in candidates:
                raise ValueError(
                    f"--model-key '{model_key}' is not an estimator in this file. "
                    f"Estimator keys available: {list(candidates.keys())}"
                )
            chosen_key = model_key
        else:
            chosen_key = next(iter(candidates))
            if len(candidates) > 1:
                warnings.warn(
                    f"Multiple estimators found in the pickled dict {list(candidates.keys())}; "
                    f"using '{chosen_key}'. Pass model_key= to pick another."
                )
        feature_cols = None
        for key in ("features", "feature_columns", "FEATURE_COLUMNS", "columns"):
            if key in obj and isinstance(obj[key], (list, tuple)):
                feature_cols = list(obj[key])
                break
        meta = {"features": feature_cols, "dict_keys": list(obj.keys()), "chosen_key": chosen_key}
        return candidates[chosen_key], meta

    raise ValueError(
        f"Loaded object of type {type(obj).__name__} has neither a .predict method "
        "nor is a dict containing one."
    )


def load_model(path, model_key: str | None = None, extra_sys_paths: list | None = None):
    """Returns (ModelWrapper, meta_dict). meta_dict may contain 'features'.

    scikit-learn pickles that reference custom functions/classes (e.g. a
    FunctionTransformer built from a project-specific module, as this
    project's v3_features.add_derived) need that module importable at
    unpickling time. The model's own directory is added to sys.path
    automatically since that's where such helper modules conventionally
    live (as in v1/v2/v3 here); extra_sys_paths adds more if needed.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".joblib", ".pkl", ".pickle"):
        import sys
        for p in [str(path.parent.resolve())] + list(extra_sys_paths or []):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            obj = joblib.load(path)
        except ModuleNotFoundError as e:
            missing = e.name or ""
            search_dirs = [str(path.parent.resolve())] + list(extra_sys_paths or [])
            looks_like_project_file = any(
                (Path(d) / f"{missing.split('.')[0]}.py").exists() for d in search_dirs
            )
            if missing.startswith("_") or not looks_like_project_file:
                raise ModuleNotFoundError(
                    f"{e}. '{missing}' looks like an internal library module (e.g. "
                    "scikit-learn's own private submodules), not project code — this "
                    "usually means the model was trained with a different "
                    "scikit-learn/joblib version than what's installed here. Pin "
                    "scikit-learn (and joblib/numpy) in requirements.txt to match the "
                    "versions used when the model was saved."
                ) from e
            raise ModuleNotFoundError(
                f"{e}. This pickle references a custom Python module that isn't "
                "importable. It's usually the .py file sitting next to the model "
                "file (already added to the import path); if it lives elsewhere, "
                "pass its directory via extra_sys_paths / --pythonpath."
            ) from e
        except Exception:
            import pickle
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        model, meta = _unwrap_sklearn_object(obj, model_key)
        return ModelWrapper(kind="sklearn", model=model), meta

    if ext in (".h5", ".keras") or path.is_dir():
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "Loading a Keras/TensorFlow model needs `pip install tensorflow`."
            ) from e
        model = tf.keras.models.load_model(path)
        return ModelWrapper(kind="keras", model=model), {}

    if ext == ".tflite":
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(path))
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path=str(path))
            except ImportError as e:
                raise ImportError(
                    "Loading a .tflite model needs `pip install tensorflow` "
                    "(or `pip install tflite-runtime`)."
                ) from e
        interpreter.allocate_tensors()
        return ModelWrapper(kind="tflite", model=interpreter), {}

    if ext in (".pt", ".pth"):
        try:
            import torch
        except ImportError as e:
            raise ImportError("Loading a .pt/.pth model needs `pip install torch`.") from e
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not callable(obj) and not hasattr(obj, "forward"):
            raise ValueError(
                "This .pt/.pth file looks like a bare state_dict (weights only), "
                "not a full model. This tool needs the original model class to "
                "reconstruct it from a state_dict — save the full model instead "
                "(torch.save(model, path)) or load it into its class yourself."
            )
        return ModelWrapper(kind="torch", model=obj), {}

    if ext == ".onnx":
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("Loading a .onnx model needs `pip install onnxruntime`.") from e
        sess = ort.InferenceSession(str(path))
        return ModelWrapper(kind="onnx", model=sess), {}

    raise ValueError(f"Unsupported model file type: '{ext or path.name}'")


# --------------------------------------------------------------------------
# Column guessing
# --------------------------------------------------------------------------

def guess_label_column(df: pd.DataFrame, meta: dict | None = None) -> str:
    meta = meta or {}
    features = meta.get("features")
    if features:
        remaining = [c for c in df.columns if c not in features]
        if len(remaining) == 1:
            return remaining[0]
    for cand in COMMON_LABEL_NAMES:
        if cand in df.columns:
            return cand
    return df.columns[-1]


def guess_feature_columns(df: pd.DataFrame, label_col: str, meta: dict | None = None) -> list[str]:
    meta = meta or {}
    features = meta.get("features")
    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(
                f"The model expects columns {missing} that aren't in the uploaded dataset. "
                f"Dataset columns: {list(df.columns)}"
            )
        return list(features)
    return [c for c in df.columns if c != label_col]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    y_true: list
    y_pred: list
    classes: list
    metrics: dict
    classification_report_df: pd.DataFrame
    confusion_matrix: np.ndarray
    confusion_matrix_fig: Any
    roc_fig: Any = None
    pr_fig: Any = None
    misclassified_df: pd.DataFrame = None
    latency_ms_per_row: float = None
    n_rows: int = 0
    warnings: list = field(default_factory=list)


def _fig_confusion_matrix(cm, classes):
    cm = np.asarray(cm)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(classes) * 1.1 * 2), max(4, len(classes) * 0.55)))
    for ax, data, title, fmt in (
        (axes[0], cm, "Confusion matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion matrix (row-normalized)", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        short = [str(c)[:18] for c in classes]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        thresh = data.max() / 2.0 if data.max() else 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text = f"{val:{fmt}}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if val > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _fig_roc_binary(y_true_bin, scores):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, roc_auc


def _fig_pr_binary(y_true_bin, scores):
    precision, recall, _ = precision_recall_curve(y_true_bin, scores)
    ap = average_precision_score(y_true_bin, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig, ap


def _fig_roc_multiclass(y_true_bin, y_proba, classes):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    aucs = {}
    for i, c in enumerate(classes):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        a = auc(fpr, tpr)
        aucs[c] = a
        ax.plot(fpr, tpr, label=f"{str(c)[:22]} (AUC={a:.2f})", linewidth=1)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest, per class)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, aucs


def run_evaluation(wrapper: ModelWrapper, X: pd.DataFrame, y_true, pos_label=None,
                    top_k_misclassified: int = 25) -> EvalResult:
    y_true = pd.Series(y_true).reset_index(drop=True)
    warn_list = []

    start = time.perf_counter()
    y_pred = wrapper.predict(X)
    elapsed = time.perf_counter() - start
    latency_ms_per_row = (elapsed / max(len(X), 1)) * 1000

    y_pred = pd.Series(y_pred).reset_index(drop=True)
    classes = wrapper.classes(y_true=y_true.tolist(), y_pred=y_pred.tolist())
    classes = [c for c in classes if c in set(y_true.tolist()) | set(y_pred.tolist())] or classes

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_fig = _fig_confusion_matrix(cm, classes)

    metrics = {
        "n_rows": len(y_true),
        "n_classes": len(classes),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "latency_ms_per_row": latency_ms_per_row,
    }

    roc_fig = None
    pr_fig = None
    try:
        y_proba = wrapper.predict_proba(X)
    except Exception as e:
        y_proba = None
        warn_list.append(f"predict_proba failed ({e}); ROC/PR curves skipped.")

    if y_proba is not None:
        try:
            if len(classes) == 2:
                pos = pos_label if pos_label in classes else classes[-1]
                pos_idx = classes.index(pos)
                y_true_bin = (y_true == pos).astype(int).to_numpy()
                scores = np.asarray(y_proba)[:, pos_idx]
                roc_fig, roc_auc_val = _fig_roc_binary(y_true_bin, scores)
                pr_fig, ap_val = _fig_pr_binary(y_true_bin, scores)
                metrics["roc_auc"] = roc_auc_val
                metrics["average_precision"] = ap_val
                metrics["positive_label"] = pos
            elif len(classes) > 2:
                y_true_bin = label_binarize(y_true, classes=classes)
                roc_fig, per_class_auc = _fig_roc_multiclass(y_true_bin, np.asarray(y_proba), classes)
                try:
                    metrics["roc_auc_macro"] = roc_auc_score(
                        y_true_bin, y_proba, average="macro", multi_class="ovr")
                    metrics["roc_auc_weighted"] = roc_auc_score(
                        y_true_bin, y_proba, average="weighted", multi_class="ovr")
                except ValueError as e:
                    warn_list.append(f"Overall multiclass ROC-AUC unavailable: {e}")
        except Exception as e:
            warn_list.append(f"Could not compute ROC/PR curves: {e}")

    mis_mask = (y_true.values != y_pred.values)
    mis_df = X.loc[mis_mask].copy() if hasattr(X, "loc") else pd.DataFrame(X[mis_mask])
    mis_df.insert(0, "predicted", y_pred[mis_mask].values)
    mis_df.insert(0, "actual", y_true[mis_mask].values)
    if y_proba is not None:
        try:
            proba_arr = np.asarray(y_proba)[mis_mask]
            mis_df.insert(0, "confidence", proba_arr.max(axis=1))
            mis_df = mis_df.sort_values("confidence", ascending=False)
        except Exception:
            pass
    mis_df = mis_df.head(top_k_misclassified).reset_index(drop=True)

    return EvalResult(
        y_true=y_true.tolist(), y_pred=y_pred.tolist(), classes=classes, metrics=metrics,
        classification_report_df=report_df, confusion_matrix=cm, confusion_matrix_fig=cm_fig,
        roc_fig=roc_fig, pr_fig=pr_fig, misclassified_df=mis_df,
        latency_ms_per_row=latency_ms_per_row, n_rows=len(y_true), warnings=warn_list,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html_report(result: EvalResult, model_path: str, data_path: str) -> str:
    m = result.metrics
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float) else
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in m.items()
    )
    cm_img = _fig_to_base64(result.confusion_matrix_fig)
    roc_html = ""
    if result.roc_fig is not None:
        roc_html += f'<h2>ROC Curve</h2><img src="data:image/png;base64,{_fig_to_base64(result.roc_fig)}"/>'
    if result.pr_fig is not None:
        roc_html += f'<h2>Precision-Recall Curve</h2><img src="data:image/png;base64,{_fig_to_base64(result.pr_fig)}"/>'

    warn_html = ""
    if result.warnings:
        items = "".join(f"<li>{w}</li>" for w in result.warnings)
        warn_html = f'<div class="warn"><b>Warnings</b><ul>{items}</ul></div>'

    report_table = result.classification_report_df.round(3).to_html(classes="tbl")
    mis_table = result.misclassified_df.round(3).to_html(classes="tbl", index=False) \
        if result.misclassified_df is not None and len(result.misclassified_df) else "<p>None — every row was predicted correctly.</p>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Model Evaluation Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; color:#1a1a1a; background:#fafafa; }}
h1 {{ margin-bottom: 0; }}
.meta {{ color:#666; margin-top:0.25rem; }}
table.tbl {{ border-collapse: collapse; margin: 1rem 0; }}
table.tbl th, table.tbl td {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 0.9rem; }}
table.tbl th {{ background:#f0f0f0; }}
img {{ max-width: 100%; margin: 0.5rem 0 1.5rem 0; border:1px solid #eee; }}
.warn {{ background:#fff8e1; border:1px solid #ffe082; padding:0.75rem 1rem; border-radius:6px; margin:1rem 0; }}
.card {{ background:white; border:1px solid #e5e5e5; border-radius:8px; padding:1.25rem 1.5rem; margin-bottom:1.5rem; }}
</style></head>
<body>
<h1>Model Evaluation Report</h1>
<p class="meta">Model: <code>{model_path}</code> &nbsp;|&nbsp; Data: <code>{data_path}</code> &nbsp;|&nbsp; Rows evaluated: {result.n_rows}</p>
{warn_html}
<div class="card"><h2>Summary Metrics</h2><table class="tbl">{rows}</table></div>
<div class="card"><h2>Confusion Matrix</h2><img src="data:image/png;base64,{cm_img}"/></div>
<div class="card">{roc_html}</div>
<div class="card"><h2>Per-Class Report</h2>{report_table}</div>
<div class="card"><h2>Misclassified Examples (top {len(result.misclassified_df) if result.misclassified_df is not None else 0})</h2>{mis_table}</div>
</body></html>"""
