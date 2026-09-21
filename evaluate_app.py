"""
evaluate_app.py
===============
Browser upload UI for the model evaluator. Run:

    streamlit run evaluate_app.py

Then in the browser: upload a model file (.joblib/.pkl/.h5/.keras/.tflite/
.pt/.pth/.onnx) and a labeled CSV test set, pick the label column, and hit
"Run evaluation" to get metric scores, confusion matrix, ROC/PR curves,
per-class report, and misclassified examples -- with a one-click HTML report
download.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Model Evaluator", layout="wide")
st.title("Model Evaluator")
st.caption("Upload a trained model and a labeled test set to auto-evaluate it.")

MODEL_EXTS = ["joblib", "pkl", "pickle", "h5", "keras", "tflite", "pt", "pth", "onnx"]

col1, col2 = st.columns(2)
with col1:
    model_file = st.file_uploader("Model file", type=MODEL_EXTS)
with col2:
    data_file = st.file_uploader("Labeled test dataset (CSV)", type=["csv"])

model_key = st.text_input(
    "Model key (only needed if the file is a dict with multiple objects, e.g. 'focus_area')",
    value="",
)
support_files = st.file_uploader(
    "Optional: supporting .py files (only needed if the model pickle references custom code, "
    "e.g. this project's v3_features.py) — upload alongside the module(s) it needs",
    type=["py"], accept_multiple_files=True,
)

if "wrapper" not in st.session_state:
    st.session_state.wrapper = None
    st.session_state.meta = None
    st.session_state.model_sig = None

if model_file is not None:
    sig = (model_file.name, model_file.size, tuple(f.name for f in (support_files or [])))
    if sig != st.session_state.model_sig:
        work_dir = Path(tempfile.mkdtemp(prefix="model_eval_"))
        tmp_path = work_dir / model_file.name
        tmp_path.write_bytes(model_file.getbuffer())
        for f in support_files or []:
            (work_dir / f.name).write_bytes(f.getbuffer())
        try:
            wrapper, meta = core.load_model(tmp_path, model_key=model_key or None)
            st.session_state.wrapper = wrapper
            st.session_state.meta = meta
            st.session_state.model_sig = sig
            st.success(f"Loaded model ({wrapper.kind}) from {model_file.name}")
            if meta.get("dict_keys"):
                st.info(f"Pickle contained keys {meta['dict_keys']}; using estimator '{meta['chosen_key']}'.")
        except Exception as e:
            st.session_state.wrapper = None
            st.error(f"Could not load model: {e}")

if data_file is not None and st.session_state.wrapper is not None:
    df = pd.read_csv(data_file)
    st.subheader("Preview of uploaded dataset")
    st.dataframe(df.head(10), use_container_width=True)

    meta = st.session_state.meta or {}
    default_label = core.guess_label_column(df, meta)
    label_col = st.selectbox(
        "True-label column", options=list(df.columns),
        index=list(df.columns).index(default_label),
    )

    try:
        default_features = core.guess_feature_columns(df, label_col, meta)
    except ValueError as e:
        st.error(str(e))
        default_features = [c for c in df.columns if c != label_col]

    feature_cols = st.multiselect(
        "Feature columns fed to the model", options=[c for c in df.columns if c != label_col],
        default=[c for c in default_features if c != label_col],
    )

    classes_hint = sorted(df[label_col].dropna().unique().tolist())
    pos_label = None
    if len(classes_hint) == 2:
        pos_label = st.selectbox("Positive class (for ROC/PR curve)", options=classes_hint, index=len(classes_hint) - 1)

    sample_n = st.number_input(
        "Subsample rows (0 = use all)", min_value=0, value=0, step=100,
    )

    if st.button("Run evaluation", type="primary"):
        eval_df = df if not sample_n else df.sample(min(sample_n, len(df)), random_state=42)
        X = eval_df[feature_cols]
        y = eval_df[label_col]
        with st.spinner("Running predictions and computing metrics..."):
            try:
                result = core.run_evaluation(st.session_state.wrapper, X, y, pos_label=pos_label)
            except Exception as e:
                st.exception(e)
                result = None

        if result is not None:
            st.session_state.result = result
            st.session_state.model_name = model_file.name
            st.session_state.data_name = data_file.name

if st.session_state.get("result") is not None:
    result = st.session_state.result
    st.divider()
    st.header("Results")

    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    _VERDICT_ICON = {"Excellent": "🟢", "Great": "🟢", "Good": "🔵", "Fair": "🟠", "Poor": "🔴"}
    _VERDICT_ST_FN = {
        "Excellent": st.success, "Great": st.success, "Good": st.info,
        "Fair": st.warning, "Poor": st.error,
    }
    verdict_fn = _VERDICT_ST_FN.get(result.verdict_label, st.info)
    verdict_fn(
        f"**{_VERDICT_ICON.get(result.verdict_label, '')} {result.verdict_label}** "
        f"(score {result.verdict_score * 100:.1f}%) — {result.verdict_blurb}"
    )

    m = result.metrics
    disp = core.format_metrics_for_display(m)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", disp["accuracy"])
    metric_cols[1].metric("Balanced Accuracy", disp["balanced_accuracy"])
    metric_cols[2].metric("F1 (macro)", disp["f1_macro"])
    metric_cols[3].metric("F1 (weighted)", disp["f1_weighted"])

    metric_cols2 = st.columns(4)
    metric_cols2[0].metric("Precision (macro)", disp["precision_macro"])
    metric_cols2[1].metric("Recall (macro)", disp["recall_macro"])
    if "roc_auc" in m:
        metric_cols2[2].metric("ROC AUC", disp["roc_auc"])
    elif "roc_auc_macro" in m:
        metric_cols2[2].metric("ROC AUC (macro)", disp["roc_auc_macro"])
    metric_cols2[3].metric("Latency / row", disp["latency_ms_per_row"])

    st.subheader("Confusion Matrix")
    st.pyplot(result.confusion_matrix_fig, use_container_width=False)

    if result.roc_fig is not None:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.subheader("ROC Curve")
            st.pyplot(result.roc_fig, use_container_width=False)
        if result.pr_fig is not None:
            with rc2:
                st.subheader("Precision-Recall Curve")
                st.pyplot(result.pr_fig, use_container_width=False)

    st.subheader("Per-Class Report")
    st.dataframe(core.format_report_df_for_display(result.classification_report_df), use_container_width=True)

    st.subheader("False Positives / False Negatives by Class")
    st.caption("TP/FP/FN/TN computed one-vs-rest per class from the confusion matrix.")
    st.dataframe(core.format_error_table_for_display(result.error_table), use_container_width=True)

    st.subheader(f"Misclassified Examples (top {len(result.misclassified_df)})")
    if len(result.misclassified_df):
        st.dataframe(result.misclassified_df, use_container_width=True)
    else:
        st.write("None — every evaluated row was predicted correctly.")

    st.divider()
    html = core.render_html_report(result, st.session_state.model_name, st.session_state.data_name)
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download full HTML report", data=html, file_name="model_evaluation_report.html", mime="text/html")
    dl2.download_button("Download metrics.json", data=json.dumps(m, indent=2, default=str),
                         file_name="metrics.json", mime="application/json")
elif data_file is None or st.session_state.wrapper is None:
    st.info("Upload a model file and a labeled CSV to begin.")
