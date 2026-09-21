"""
evaluate_model.py
=================
CLI: point it at a model file and a labeled CSV, it auto-evaluates the model
and writes a self-contained HTML report (metrics, confusion matrix, ROC/PR
curves, per-class report, misclassified rows) plus a metrics.json.

Supports: .joblib / .pkl / .pickle (scikit-learn, including this project's
dict-wrapped models like sned_model_v3.joblib), .h5 / .keras (TensorFlow),
.tflite, .pt / .pth (a full pickled torch model), .onnx.

Examples
--------
    python evaluate_model.py --model ../v3/sned_model_v3.joblib \\
                              --data ../v3/sned_behavior_dataset_v3.csv

    python evaluate_model.py --model my_model.pkl --data test.csv \\
                              --label-col target --feature-cols f1,f2,f3

    python evaluate_model.py --model model.joblib --data test.csv --sample 2000 --no-open
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

import core


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to the model file")
    p.add_argument("--data", required=True, help="Path to a labeled CSV test set")
    p.add_argument("--label-col", default=None, help="Name of the true-label column (auto-detected if omitted)")
    p.add_argument("--feature-cols", default=None,
                   help="Comma-separated feature column names (default: model's own list, else all non-label columns)")
    p.add_argument("--model-key", default=None,
                   help="If the pickle is a dict of multiple objects, which key holds the estimator (e.g. 'focus_area')")
    p.add_argument("--pythonpath", default=None,
                   help="Extra comma-separated directories to add to sys.path before unpickling "
                        "(needed if the model references custom code that isn't next to the model file)")
    p.add_argument("--pos-label", default=None, help="Positive class label for binary ROC/PR (default: last class)")
    p.add_argument("--sample", type=int, default=None, help="Randomly subsample this many rows before evaluating")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None, help="Where to write the report (default: eval_report_<timestamp>)")
    p.add_argument("--no-open", action="store_true", help="Don't auto-open the HTML report in a browser")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")
    if not data_path.exists():
        sys.exit(f"Dataset file not found: {data_path}")

    print(f"Loading model: {model_path}")
    extra_paths = [p.strip() for p in args.pythonpath.split(",")] if args.pythonpath else None
    try:
        wrapper, meta = core.load_model(model_path, model_key=args.model_key, extra_sys_paths=extra_paths)
    except Exception as e:
        sys.exit(f"Could not load model: {e}")
    print(f"  -> detected format: {wrapper.kind}")
    if meta.get("dict_keys"):
        print(f"  -> pickle contained keys {meta['dict_keys']}; using estimator '{meta['chosen_key']}'")

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)

    label_col = args.label_col or core.guess_label_column(df, meta)
    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",")]
    else:
        feature_cols = core.guess_feature_columns(df, label_col, meta)

    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing columns: {missing}. Available: {list(df.columns)}")

    print(f"  -> label column : {label_col}")
    print(f"  -> feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df[label_col]

    print(f"Evaluating on {len(df):,} rows ...")
    result = core.run_evaluation(wrapper, X, y, pos_label=args.pos_label)

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"eval_report_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=str))
    result.classification_report_df.round(4).to_csv(out_dir / "classification_report.csv")
    if result.misclassified_df is not None:
        result.misclassified_df.to_csv(out_dir / "misclassified_examples.csv", index=False)

    html = core.render_html_report(result, str(model_path), str(data_path))
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    for k, v in core.format_metrics_for_display(result.metrics).items():
        print(f"  {k:<22}: {v}")
    print("-" * 70)
    print(f"  VERDICT: {result.verdict_label} (score {result.verdict_score * 100:.1f}%)")
    print(f"    {result.verdict_blurb}")
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print("=" * 70)
    print()
    print("False positives / false negatives by class:")
    print(core.format_error_table_for_display(result.error_table).to_string(index=False))
    print(f"Full report : {report_path.resolve()}")
    print(f"Raw metrics : {(out_dir / 'metrics.json').resolve()}")
    print(f"Misclassified rows: {(out_dir / 'misclassified_examples.csv').resolve()}")

    if not args.no_open:
        try:
            webbrowser.open(report_path.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
