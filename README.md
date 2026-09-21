# Model Evaluator

Generic evaluation tool: give it a trained model file and a labeled test CSV,
it auto-computes metric scores, a confusion matrix, ROC/PR curves, a
per-class report, and misclassified examples, and writes a self-contained
HTML report.

Supports `.joblib` / `.pkl` / `.pickle` (scikit-learn, including this
project's dict-wrapped models like `sned_model_v3.joblib`), `.h5` / `.keras`
/ `.tflite` (TensorFlow, optional dependency), `.pt` / `.pth` (a full pickled
torch model, optional dependency), `.onnx` (optional dependency).

## Setup

```bash
cd model_evaluator
pip install -r requirements.txt
```

## Option A: web upload UI

```bash
streamlit run evaluate_app.py
```

Opens a browser page. Upload the model file and a labeled CSV, pick the
label column (auto-guessed), and click "Run evaluation".

## Option B: CLI

```bash
python evaluate_model.py --model ../v3/sned_model_v3.joblib --data ../v3/sned_behavior_dataset_v3.csv
```

This auto-detects the label column (`focus_area`) and feature columns (from
the `features` list saved inside the joblib file), evaluates, and opens
`eval_report_<timestamp>/report.html` in your browser.

Useful flags:

- `--label-col NAME` — force the true-label column instead of auto-detecting.
- `--feature-cols a,b,c` — force which columns are fed to the model.
- `--model-key focus_area` — if the pickle is a dict with more than one
  estimator inside it, pick which one to evaluate.
- `--pos-label X` — which class counts as "positive" for a binary ROC/PR curve.
- `--sample 2000` — evaluate on a random subsample instead of the whole file (faster).
- `--output-dir path` — where to write the report (default: a timestamped folder).
- `--no-open` — don't auto-open the report in a browser.

Run `python evaluate_model.py --help` for the full list.

## What gets computed

- Accuracy, balanced accuracy, precision/recall/F1 (macro & weighted)
- Full per-class precision/recall/F1/support table
- Confusion matrix (raw counts + row-normalized), plotted as a heatmap
- ROC curve + AUC and Precision-Recall curve + AP (binary classification)
- One-vs-rest ROC curves per class + macro/weighted AUC (multiclass, when the
  model exposes `predict_proba`/probabilities)
- Top misclassified rows, sorted by the model's own confidence in its wrong answer
- Inference latency per row

## Notes

- A model file alone isn't enough to evaluate anything — you always need a
  labeled test set (features + the true answer) alongside it.
- For `.pt`/`.pth`, the file must be a full pickled model
  (`torch.save(model, path)`), not a bare `state_dict` — a state_dict has no
  architecture attached, so there's nothing to reconstruct it from
  automatically.
