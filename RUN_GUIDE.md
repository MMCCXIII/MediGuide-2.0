# Running MedGuide AI locally

MedGuide AI is an educational Streamlit application that maps reported symptoms to conditions in its bundled dataset. It is not a diagnostic or prescribing tool.

## Prerequisites

- Windows, macOS, or Linux
- Python 3.12 (recommended; see `runtime.txt`)
- Internet access on the first install and, optionally, first biomedical NLP model load

## Start the web app (Windows PowerShell)

From the project folder, create and activate a virtual environment if one does not already exist:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, run the app through the virtual environment directly; activation is not required:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Otherwise install and start it with:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the **Local URL** printed in the terminal, normally [http://localhost:8501](http://localhost:8501). Keep that terminal open while using the app, and press `Ctrl+C` there to stop the server.

To use a specific port or allow another device on the local network to connect:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## How to use the app

1. Open the local URL in a browser.
2. Enter a clear description of symptoms in the assessment chat.
3. Confirm the symptoms the app recognizes before viewing its educational result.
4. Treat emergency warnings as urgent: contact local emergency services or a qualified clinician as appropriate.

The app includes phrase-based emergency screening, but it must not be relied on as a replacement for medical care.

## Optional API server

The project also exposes a FastAPI service. With the same environment active, run:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Check it at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health), or use the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/assess -ContentType 'application/json' -Body '{"symptoms":["itching","skin rash"]}'
```

## Docker

With Docker Desktop running:

```powershell
docker build -t medguide-ai .
docker run --rm -p 8501:8501 medguide-ai
```

Then open [http://localhost:8501](http://localhost:8501).

## Verify the installation

```powershell
python -m pytest -q
python scripts/validate_dataset.py
python -m compileall -q app.py api.py src scripts
```

## Troubleshooting

- **`No module named streamlit`**: install dependencies using the same Python executable that starts the app: `python -m pip install -r requirements.txt`.
- **PowerShell activation policy error**: use the full `.venv\\Scripts\\python.exe` commands above instead of activating the environment.
- **Slow first startup**: optional biomedical NLP components may download model files on first use. Model training is deliberately not performed during app startup; use the offline training scripts and deploy validated artifacts.
- **Port already in use**: choose another port, for example `--server.port 8502`.


## Bundled expanded disease dataset

The project includes `models/diseases_and_symptoms.csv`. At startup, the app merges it with the original `models/Training.csv` in memory, deduplicating disease labels and symptom features. The current merged corpus has 647 unique conditions, 1,495 symptom features, and 11,977 rows. The application never trains a model during startup. Train and validate models offline, then deploy compatible, versioned artifacts. If TensorFlow or a compatible Random Forest artifact is unavailable, the app remains usable with the dataset-backed symptom-profile fallback and knowledge-base fallback.

To verify the data and run tests:

```bash
python scripts/verify_app.py
pytest -q
```

## Enable the TensorFlow primary model

The project uses TensorFlow first only when its compatible model and metadata artifacts are present. If `models/mediguide_tensorflow.keras` and `models/tensorflow_metadata.json` are absent, the app uses a compatible Random Forest artifact, then the support-filtered dataset-profile fallback, and finally the curated knowledge-base fallback. It does not train automatically.

From the project root, after installing dependencies from `requirements.txt`, train and save the TensorFlow artifacts with:

```bash
python -m src.tensorflow_predictor --train
```

Then restart the app and run `pytest -q`. Do not describe TensorFlow as active until the artifacts have been generated and loaded successfully.


## Dataset evaluation and saved history

Open **Method & data** to review the dataset audit and grouped holdout benchmark. To regenerate the reports:

```powershell
python scripts/audit_dataset.py
python scripts/audit_dataset.py --evaluate
```

The evaluation is a synthetic/imported-profile benchmark only. It is not an independent clinical validation set; the report lists conditions that have too few distinct patterns to evaluate.

To save a report, expand **Private saved history** in the sidebar, set a unique passphrase (at least 16 characters), and accept the storage notice. In Chat, choose **Save assessment & set follow-up**. Use **My history** to download the saved report, add follow-up progress notes, mark follow-up status, or delete records. Keep the passphrase safe: it is not stored or recoverable. Saved records are encrypted in the app host's local database, so this is not device-only storage on a hosted deployment.


## Secure API deployment

Before exposing the FastAPI assessment endpoint, set a long, randomly generated `MEDIGUIDE_API_KEY` in the deployment secret manager (never commit it to source control). Send the value in the `X-API-Key` request header. Without this setting, `/v1/assess` returns HTTP 503 by design. Keep `/health` free of personal data, terminate TLS at a trusted proxy, enforce request-body limits and distributed rate limiting at the gateway, and restrict access to the server-side SQLite databases and backups. The built-in process-local limiter is defense in depth, not a substitute for gateway controls.
