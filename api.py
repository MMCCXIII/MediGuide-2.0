"""FastAPI interface for MedGuide AI educational symptom-pattern ranking and safety prompts."""
from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict, deque

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "0")

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated

from src.config import APP_VERSION
from src.data_pipeline import load_data_bundle
from src.kb_predictor import KnowledgeBasePredictor
from src.profile_predictor import DatasetProfilePredictor
from src.nlp import SymptomMapper
from src.safety import triage_message

try:
    from fastapi import FastAPI, HTTPException, Header, Request
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install the API extras with: pip install fastapi uvicorn") from exc

app = FastAPI(
    title="MedGuide AI API",
    version=APP_VERSION,
    description="Experimental educational symptom-pattern API with conservative keyword safety prompts; no diagnosis or validated triage.",
)

data = load_data_bundle()
kb = KnowledgeBasePredictor()

rf = None
try:
    from src.rf_predictor import RandomForestPredictor
    rf = RandomForestPredictor.load()
    rf.validate_compatibility(data)
except Exception:
    rf = None

tf_model = None
try:
    from src.tensorflow_predictor import TensorFlowDiseasePredictor
    tf_model = TensorFlowDiseasePredictor.load(data)
except Exception:
    tf_model = None

from src.ensemble import EnsemblePredictor
profile = DatasetProfilePredictor(data)
ensemble = EnsemblePredictor(rf=rf, tensorflow=tf_model, kb=kb, profile=profile, class_support=data.targets.value_counts().to_dict())
mapper = SymptomMapper(data.feature_names)

# Assessment endpoints are closed until an operator configures a secret API key.
# The in-process limiter is a lightweight defense-in-depth measure; public
# deployments must also rate-limit at their reverse proxy/API gateway.
API_KEY = os.getenv("MEDIGUIDE_API_KEY", "").strip()
_RATE_WINDOW_SECONDS = 60
_RATE_LIMIT = 30
_request_times: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()


def _enforce_api_access(request: Request, supplied_key: str | None) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Assessment API is not configured. The operator must set MEDIGUIDE_API_KEY.")
    if not supplied_key or not secrets.compare_digest(supplied_key, API_KEY):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        events = _request_times[client]
        while events and now - events[0] > _RATE_WINDOW_SECONDS:
            events.popleft()
        if len(events) >= _RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        events.append(now)
        if len(_request_times) > 10_000:
            for address in list(_request_times):
                bucket = _request_times[address]
                while bucket and now - bucket[0] > _RATE_WINDOW_SECONDS:
                    bucket.popleft()
                if not bucket:
                    _request_times.pop(address, None)



SymptomText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: list[SymptomText] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Up to 20 free-text or recognized symptom strings; each is limited to 160 characters.",
    )
    age_group: Literal["Child", "Adult", "Older adult"] | None = None
    sex: Literal["Male", "Female"] | None = None
    pregnancy: Literal["Yes", "No", "Possibly"] | None = None
    duration_days: int | None = Field(default=None, ge=0, le=365, description="Duration in days")
    severity: Literal["Mild", "Moderate", "Severe", "Very severe"] | None = None
    worsening: bool = Field(default=False, description="Whether symptoms are actively worsening")


@app.get("/health")
@app.middleware("http")
async def enforce_request_body_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 8_192:
                return JSONResponse(status_code=413, content={"detail": "Request body exceeds the 8 KB limit."})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
    return await call_next(request)


def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "model": ensemble.model_name,
        "condition_catalogue_labels": data.class_count,
        "labels_meeting_minimum_training_support": profile.metadata["class_count"],
        "labels_below_minimum_training_support": profile.metadata["excluded_low_support_count"],
        "minimum_training_examples_per_class": profile.metadata["minimum_training_examples_per_class"],
        "symptoms_vocabulary": len(data.feature_names),
    }


@app.post("/v1/assess")
def assess(
    request: AssessmentRequest,
    http_request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _enforce_api_access(http_request, x_api_key)
    raw_text = "; ".join(request.symptoms)
    mapped, _ = mapper.map_text(raw_text)

    # Build context dict for triage and explanatory context
    context = {}
    if request.age_group:
        context["age_group"] = request.age_group
    if request.sex:
        context["sex"] = request.sex
    if request.pregnancy:
        context["pregnancy"] = request.pregnancy
    if request.severity:
        context["severity"] = request.severity
    if request.worsening:
        context["worsening"] = "worsening"

    triage = triage_message(raw_text, mapped, context)

    # Emergency guidance takes precedence over all automated disease ranking.
    if triage.level.value == "Emergency":
        return {
            "status": "emergency_guidance",
            "message": triage.message,
            "symptoms_recognized": sorted(mapped),
            "triage": {"level": triage.level.value, "reasons": triage.reasons, "message": triage.message},
            "educational_only": True,
            "prediction_withheld": True,
        }

    if len(mapped) < 2:
        return {
            "status": "needs_more_information",
            "message": f"Identified {len(mapped)} supported symptom. MedGuide requires at least two supported symptoms for pattern ranking.",
            "symptoms_recognized": sorted(mapped),
            "triage": {
                "level": triage.level.value,
                "reasons": triage.reasons,
                "message": triage.message,
            },
            "educational_only": True,
        }

    prediction = ensemble.predict(
        mapped,
        age_group=request.age_group,
        sex=request.sex,
        pregnancy=request.pregnancy,
        duration_days=request.duration_days,
        severity=request.severity,
        worsening=request.worsening,
        apply_clinical=True,
    )

    return {
        "status": "completed",
        "condition": data.canonical_disease(prediction.disease),
        "relative_pattern_score": None if getattr(prediction, "forced_nonspecific", False) else prediction.confidence,
        "score_interpretation": (
            "No specific pattern score: the available evidence was insufficient."
            if getattr(prediction, "forced_nonspecific", False)
            else "Uncalibrated relative model score; not a probability or diagnostic confidence."
        ),
        "alternatives": [
            {
                "condition": data.canonical_disease(name),
                "relative_score": round(float(score), 4),
            }
            for name, score in prediction.alternatives
        ],
        "route": getattr(prediction, "route", "ensemble"),
        "symptoms_used": sorted(mapped),
        "triage": {
            "level": triage.level.value,
            "reasons": triage.reasons,
            "message": triage.message,
        },
        "clinical_adjusted": getattr(prediction, "clinical_adjusted", False),
        "educational_only": True,
    }
