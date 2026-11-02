# MedGuide AI — Current Release & Deployment Guide

## Current release

**Application version:** `1.0.0`

MedGuide AI is an educational Streamlit symptom-pattern assistant. It accepts an English free-text description of symptoms, extracts supported symptom features, applies clinical/context filters, and ranks possible condition patterns from the bundled model/knowledge base.

It is **not a diagnostic device, medical advice service, prescribing system, or substitute for a licensed clinician**.

### Current bundled data

| Component | Current release |
|---|---:|
| Application version | 1.0.0 |
| Training rows | 11,977 |
| Condition classes | 647 |
| Normalized symptom features | 1,495 |
| Optional TensorFlow artifact | `models/mediguide_tensorflow.keras` (not bundled) |
| Packaged Random Forest artifact | `models/random_forest.joblib` |
| Dataset schema | v4 |
| Python target | 3.12 |

The training data is synthetic model-development data generated/cleaned for the project. It is not patient data, clinical-trial data, or evidence of diagnostic accuracy.


### UI/UX refresh

The main experience remains a **free-text chatbot**. Users describe symptoms in their own words and can continue the conversation; example prompts only start a chat message and do not open a symptom-selection form. The interface uses a responsive teal-and-navy medical theme, clearer conversational onboarding, and separate destinations for Chat, Appointments, My health, Health library, and Method & data.

- **Appointments:** create, review, and cancel appointments using date and time pickers. Appointment details are encrypted using the same opt-in private history key as saved assessments. Upcoming reminders are shown in the app while it is open; they are not push notifications or SMS alerts.
- **Health library:** search the bundled condition catalogue without interrupting the chatbot workflow.
- **Mobile layout:** wider desktop content, compact tab styling, responsive padding, larger touch-friendly buttons, and a simplified welcome state.

## What the current app does

### 1. English symptom chat

The main interface is a Streamlit chat experience. Users can describe symptoms in ordinary English, including duration and whether symptoms are worsening.

The app keeps the conversation's accumulated supported symptoms so later messages can add context to the current assessment.

### 2. Biomedical symptom extraction

The runtime uses a layered symptom-mapping pipeline:

- Exact symptom matching.
- Common-language aliases.
- Conservative fuzzy matching.
- Optional Hugging Face biomedical NER using `d4data/biomedical-ner-all`.
- Optional semantic symptom matching using `emilyalsentzer/Bio_ClinicalBERT`.

The NLP layers map text into the application's supported symptom vocabulary. They do **not** directly choose a disease. Age, sex, and pregnancy are retained as context only; they are not used to exclude conditions or adjust disease rankings in this release.

### 3. Current prediction/ranking path

The current application uses an ensemble-style architecture:

**English text → symptom extraction → TensorFlow artifact (if compatible) → Random Forest artifact (if compatible) → support-filtered dataset-profile fallback → curated knowledge-base fallback**

The knowledge-base layer applies contextual rules and can deliberately return a nonspecific result when the available evidence is weak.

The application also records which prediction route produced the displayed result. TensorFlow is only active when both `models/mediguide_tensorflow.keras` and `models/tensorflow_metadata.json` exist and pass compatibility checks. If those artifacts are absent or stale, the application does not train models at startup. If no compatible model is present, the dataset-backed symptom-profile fallback retains access to the expanded condition catalogue before the curated knowledge-base fallback is considered.

### 4. Clinical/context filters

Optional context can be supplied through the UI:

- Age group.
- Sex.
- Pregnancy status.
- Symptom duration in days.
- Severity.
- Whether symptoms are worsening.

These values are used to adjust eligibility/ranking. Pregnancy is not treated as a pregnancy test.

The system also recognizes emergency-pattern phrases and presents urgent-care guidance when appropriate.

### 5. Safety-oriented result presentation

Results are presented as **pattern rankings**, not as confirmed diagnoses.

The UI can show:

- A primary possible condition pattern.
- A confidence/pattern-strength indicator.
- Alternative patterns.
- The prediction route.
- Applied clinical/context badges.
- Triage or urgent-care guidance.
- An explanation of how the ranking was produced.

Medication, precaution, diet, and activity information comes only from the bundled educational support data and is not individualized prescribing advice.

### 6. Conversation controls

The current UI includes:

- New Chat.
- Multiple chat sessions in the sidebar.
- Delete chat.
- Edit the most recent user message and rerun the assessment.
- Optional context controls.
- Processing/status feedback while an assessment is running.
- Helpful / Not useful feedback.
- Privacy and safety information.
- Technical/model/data details outside the main chat flow.

### 7. Privacy behavior

Conversation state is kept in the current browser session and the application may maintain local runtime/analytics state.

Do **not** enter names, addresses, phone numbers, medical-record numbers, or other identifying information.

### 8. Optional FastAPI API

The project also contains an optional FastAPI interface.

Available endpoints:

- `GET /health`
- `POST /v1/assess`

The assessment endpoint accepts a list of symptoms and requires at least two supported symptoms before prediction.

Example:

```json
{
  "symptoms": ["itching", "skin rash"]
}
```

The API response includes the condition pattern, an uncalibrated `relative_pattern_score` (never a disease probability), alternatives, prediction route, recognized symptoms, and an `educational_only` flag. Emergency-level responses withhold automated disease predictions. The assessment endpoint is closed unless `MEDIGUIDE_API_KEY` is configured; callers must provide it in the `X-API-Key` header. A small in-process limit is included, but public deployments must additionally enforce rate limits at the API gateway or reverse proxy.

## Current model/data architecture

### Bundled runtime data

The `models/` directory contains the current runtime assets, including:

- `Training.csv` — canonical training matrix.
- `Symptom-severity.csv` — symptom weights.
- `description.csv` — condition descriptions.
- `medications.csv` — educational medication/support references.
- `precautions_df.csv` — precautions.
- `diets.csv` — general supportive guidance.
- `workout_df.csv` — conservative activity guidance.
- `best_model.joblib` — persisted classical model artifact.
- `random_forest.joblib` — Random Forest artifact used by the current ranking path.
- Dataset/training metadata and comparison files.

The bundled merged corpus contains **647 unique condition labels, 1,495 normalized symptom features, and 11,977 training rows**. It combines the original generated training matrix with the supplied `models/diseases_and_symptoms.csv` file, deduplicating overlapping disease labels and symptom features. `models/condition_evidence.csv` records which profile descriptions still need condition-specific medical-source review. The application does not train or rewrite model artifacts at startup. Train models offline, evaluate them, and deploy compatible artifacts as part of a controlled release. The 647 labels are a catalogue, not 647 equally supported prediction classes: 533 labels have only one training row and are withheld from specific ML/profile outputs. The current minimum is five training rows, leaving 114 labels eligible for dataset/profile model output. The curated knowledge-base fallback has its own separately maintained condition profiles. If a compatible trained model is unavailable, the application uses the support-filtered dataset-profile and curated knowledge-base fallback paths.

### Model compatibility

The application performs compatibility checks before using persisted model artifacts. A model trained against an incompatible dataset/schema must not silently be treated as current.

The bundled TensorFlow artifact is optional/compatibility-gated. If it cannot be loaded or does not pass the required compatibility checks, the application uses the available compatible classical/Random Forest and knowledge-base paths instead.

## Installation

Use **Python 3.12**.

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```


**Model setup:** Train and validate models offline, then deploy compatible versioned artifacts. The running application only loads existing artifacts and never trains or rewrites them at startup.

## Streamlit Community Cloud deployment

The repository's deployment log showed a dependency-installation failure before the application itself started.

### Root cause

The repository contains a `pyproject.toml`, but it originally contained only Ruff configuration:

```toml
[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = ["E9", "F"]
```

It did **not** contain the required PEP 621 `[project]` table.

The deployment installer detected `pyproject.toml` and attempted to process it as a Python project, resulting in:

```text
error: No `project` table found in: .../pyproject.toml
installer returned a non-zero exit code
```

This occurs during **dependency processing**, before `app.py` is executed. Therefore the log does not indicate a model-prediction or Streamlit runtime error.

### Deployment fix

The current `pyproject.toml` should contain a valid minimal project declaration, for example:

```toml
[project]
name = "medguide-ai"
version = "1.0.0"
description = "Educational symptom-pattern assistant"
requires-python = ">=3.12,<3.13"

[tool.ruff]
line-length = 120
exclude = ["models"]

[tool.ruff.lint]
select = ["E9", "F"]
```

The actual runtime dependencies remain defined in `requirements.txt`.

After committing/pushing this change, redeploy the application. If Streamlit Cloud has cached the failed environment, restart/redeploy after the repository update.

## Verification

Run these checks from the project root:

```bash
python -m pytest -q
python scripts/validate_dataset.py
python -m compileall -q app.py api.py src scripts
```

For a quick application start:

```bash
python -m streamlit run app.py
```

For the optional API:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Then check:

```text
http://127.0.0.1:8000/health
```

## Docker

```bash
docker build -t medguide-ai .
docker run --rm -p 8501:8501 medguide-ai
```

Then open:

```text
http://localhost:8501
```

## Troubleshooting

### `No project table found`

Check `pyproject.toml`. It must contain a `[project]` table if the deployment installer is processing it as a Python project.

### `No module named streamlit`

Install using the same Python interpreter that starts the application:

```bash
python -m pip install -r requirements.txt
```

### Slow first startup

The optional biomedical NLP models may download on first use. Deterministic symptom matching and compatible classical/knowledge-base paths remain available as fallbacks.

### TensorFlow/model compatibility warning

Do not bypass model compatibility checks just to force an incompatible artifact to load. A model trained against a different feature/class schema can produce misleading output.

## Medical safety

This application is an educational machine-learning/software demonstration.

- A ranked condition pattern is not a diagnosis.
- Scores are uncalibrated relative ranking signals, not clinical probabilities or diagnostic confidence. This release does not apply unsupported prevalence or demographic multipliers to alter disease rankings.
- Medication references are not prescriptions.
- The application does not determine an individualized dose or treatment plan.
- Emergency warnings should not be treated as a substitute for local emergency care.
- Seek qualified medical advice for personal symptoms, especially severe, rapidly worsening, or emergency symptoms.

## Release status

This README reflects the **1.0.0 application code and bundled data in this release**, rather than the older 32-condition/113-feature documentation that was previously present.

For reproducibility, keep the application code, `models/` artifacts, metadata, and dependency versions synchronized when making future model or dataset changes.


## Recent improvements: evaluation, explanations, saved history, and privacy

- **Dataset audit:** open **Method & data → Dataset audit & evaluation readiness** for duplicate symptom-vector groups, conflicting labels, and the number of conditions that cannot be held out because they have only one example.
- **Grouped holdout benchmark:** `python scripts/audit_dataset.py --evaluate` regenerates `models/dataset_audit_report.json` and `models/DATASET_AUDIT_REPORT.md`. Exact symptom vectors are kept out of both train and test sides to reduce leakage. The current benchmark covers only classes with enough unique patterns and is **not clinical diagnostic accuracy**; no independent patient-outcome cohort is included.
- **Prediction explanations and sources:** each suggested pattern shows reported symptoms that also occur in its stored profile. Where the evidence CSV includes a reference, the condition detail panel shows the source URL and its review/validation status. Missing references are disclosed rather than invented.
- **Saved assessments and follow-up:** saving is optional. Open **Private saved history** in the sidebar, enter a unique passphrase of at least 16 characters, and accept the storage notice. Then use **Save assessment & set follow-up** in the chat. The **My history** tab supports report downloads, due-date tracking, follow-up notes, status changes, and deletion.
- **Privacy:** saved report payloads are encrypted in the app's local SQLite database. The app does not save the private history key. If the key is lost, those records cannot be recovered. The database is stored on the app host (not necessarily the user's phone); public deployments still need HTTPS, access controls, retention rules, and a privacy notice. Unsaved chat state is not automatically copied to browser localStorage anymore.

To run checks:

```bash
python -m pytest -q
python scripts/audit_dataset.py
python scripts/audit_dataset.py --evaluate
python scripts/verify_app.py
```
