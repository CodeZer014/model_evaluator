"""
make_standalone.py
===================
One-time conversion: takes a scikit-learn model pickle that references
custom project code (e.g. sned_model_v3.joblib's FunctionTransformer built
from v3_features.add_derived) and re-saves it with cloudpickle, which bakes
the actual function bytecode into the file instead of just a "look this up
in module X" reference.

Result: the output file loads on its own -- no supporting .py files needed,
anywhere, ever (only the `cloudpickle` package, a normal pip dependency,
needs to be installed wherever it's loaded -- already in requirements.txt).

This still needs the original supporting .py files ONCE, right now, to load
the model in the first place -- it can't retroactively fix a reference that
was never resolvable. Run it from inside the folder that already has those
files (e.g. v1/, v2/, v3/) so the plain import works, exactly like
v3_predict.py does.

Usage (from inside e.g. v3/):
    python ..\\model_evaluator\\make_standalone.py sned_model_v3.joblib sned_model_v3_standalone.joblib
"""
import sys
from pathlib import Path

import cloudpickle
import joblib


def _find_custom_modules(obj):
    """cloudpickle only inlines a function's actual code ('pickle by value')
    for modules it's told to -- by default it still just stores a reference
    to importable modules, same as plain pickle. Walk the object for
    FunctionTransformer steps (the pattern this project's pipelines use) and
    collect the modules their functions live in, so we know what to force."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    mods = set()

    def visit(o):
        if isinstance(o, Pipeline):
            for _, step in o.steps:
                visit(step)
        elif isinstance(o, FunctionTransformer) and o.func is not None:
            mod = getattr(o.func, "__module__", None)
            if mod and mod != "__main__":
                mods.add(mod)
        elif isinstance(o, dict):
            for v in o.values():
                visit(v)

    visit(obj)
    return mods


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(f"Usage: python {Path(__file__).name} <input.joblib> <output.joblib> [extra_module_name]")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    extra_module = sys.argv[3] if len(sys.argv) == 4 else None

    sys.path.insert(0, str(src.parent.resolve()))
    obj = joblib.load(src)

    custom_modules = _find_custom_modules(obj)
    if extra_module:
        custom_modules.add(extra_module)
    if not custom_modules:
        print("No FunctionTransformer-based custom modules detected; saving as-is "
              "(if this still needs a supporting file after conversion, pass its "
              "module name as a third argument).")
    for mod_name in custom_modules:
        mod = sys.modules.get(mod_name) or __import__(mod_name)
        cloudpickle.register_pickle_by_value(mod)
        print(f"Baking in code from module: {mod_name}")

    with open(dst, "wb") as fh:
        cloudpickle.dump(obj, fh)

    print(f"Wrote standalone copy: {dst.resolve()}")
    print("Verifying it loads with zero project files on the import path ...")

    import subprocess
    result = subprocess.run(
        [sys.executable, "-c",
         f"import cloudpickle; obj = cloudpickle.load(open(r'{dst.resolve()}', 'rb')); "
         f"print('OK, keys:', list(obj.keys()) if isinstance(obj, dict) else type(obj))"],
        capture_output=True, text=True, cwd=str(Path.home()),
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        sys.exit(f"Verification failed:\n{result.stderr}")


if __name__ == "__main__":
    main()
