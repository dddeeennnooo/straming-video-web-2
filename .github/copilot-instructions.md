# Copilot instructions
This repository is a small prediction prototype. The guidance below helps an AI coding agent to be immediately productive.

**Big Picture:**
- Data generation: `src/data.py` provides `make_dataset(n=1000, seed=42)` which creates a synthetic dataframe with `date`, `strength_diff`, `form_diff`, `home`, and `y`.
- Training: `src/train.py` is the intended training pipeline (currently a stub). Trained models should be persisted into `models/`.
- Prediction: `src/predict.py` should load serialized models from `models/` and expose a minimal API for inference.

**Key files and patterns**
- `src/data.py`: pure function `make_dataset(...)` — prefer deterministic RNG via `seed` and `np.random.default_rng`.
- `src/train.py` and `src/predict.py`: small, single-responsibility scripts. Keep side effects (disk I/O) isolated to `models/` and CLI entry points.
- `models/`: directory for serialized models (pickle/joblib). It is empty now; treat it as an artifact store.
- `main.py`, `test.py`: present but empty; they are OK to use as lightweight CLIs or integration tests.

**Developer workflows (explicit commands)**
- Create & activate venv (Windows):

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas
```

- Quick smoke test (generate dataset):

```powershell
& venv\Scripts\python -c "from src.data import make_dataset; print(make_dataset(5))"
```

**Project-specific conventions**
- Prefer pure functions that return pandas DataFrames (see `make_dataset`). Avoid top-level side effects when importing modules.
- Use `np.random.default_rng(seed)` for reproducible randomness (follow `src/data.py`).
- Persist models to `models/` with explicit filenames (e.g. `models/model_v1.pkl`) and include metadata (seed, training datetime).

**Integration points & external deps**
- Current code uses `numpy` and `pandas`. There are no external services or DB integrations in the codebase.
- If adding scikit-learn or similar, add to project-level `requirements.txt` and document training hyperparameters in `src/train.py`.

**When updating this file**
- If a `.github/copilot-instructions.md` already exists, preserve human-written sections and merge only missing, project-specific details.

If anything about the expected developer workflow or external integrations is missing or incorrect, tell me which commands or files to reference and I will iterate.
