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

## Skipping the supporting .py files (v1/v2/v3 models, or any custom pipeline)

This project's models (`sned_model.joblib`, `sned_model_v2.joblib`,
`sned_model_v3.joblib`) build their pipeline with a `FunctionTransformer`
wrapping a project-specific function (`add_derived` in `features.py` /
`v2_features.py` / `v3_features.py`). A plain pickle only stores *a reference*
to that function ("look up `add_derived` in module `v3_features`"), so
whatever unpickles it needs that module importable — which is why the app
asks for those `.py` files.

`make_standalone.py` fixes this once, permanently, per model: it re-saves the
model with `cloudpickle`, which bakes the function's actual code into the
file instead of a reference. Run it once, from inside the folder that already
has the `.py` files (so it can load the model normally the first time):

```bash
cd ../v3
python ../model_evaluator/make_standalone.py sned_model_v3.joblib sned_model_v3_standalone.joblib
```

(Same pattern for v1: `cd ../v1 && python ../model_evaluator/make_standalone.py sned_model.joblib sned_model_v1_standalone.joblib`,
and v2: `cd ../v2 && python ../model_evaluator/make_standalone.py sned_model_v2.joblib sned_model_v2_standalone.joblib`.)

From then on, upload only `sned_model_v3_standalone.joblib` (or v1/v2's) plus
the CSV — no supporting `.py` files needed, on this machine or anyone else's,
since the model no longer depends on where it was trained. This only requires
`cloudpickle` to be installed wherever the *converted* file is loaded
(already in `requirements.txt`); it's not needed for a plain, unconverted
`.joblib`/`.pkl` file.

Note `sned_model.joblib` (v1) is a dict with **two** estimators
(`focus_area` and `support_level`) — the tool picks `focus_area` by default;
pass `--model-key support_level` (CLI) or fill in the "Model key" field (app)
to evaluate the other one instead.
