"""MedGuide AI — focused, clean Streamlit health education interface."""
from __future__ import annotations

# --- OpenMP runtime conflict guard (must be set BEFORE any other import) ---
# numpy>=2, PyTorch, and TensorFlow each ship their own copy of libiomp5md.dll.
# Loading more than one in the same process triggers OMP Error #15 and either
# crashes the app or silently degrades thread scheduling. Setting this env var
# tells Intel's OpenMP loader to ignore the duplicate-init check. This is the
# same workaround used by the PyTorch and TensorFlow CIs.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "0")

import base64
import html
import json
import uuid
from typing import Any
from datetime import datetime, date, timedelta
from pathlib import Path
from PIL import Image

FAVICON_PATH = Path(__file__).parent / "assets" / "favicon.png"
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

def get_logo_base64() -> str:
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""

# Streamlit's source watcher treats ``torch.classes`` as a Python package and
# probes ``__path__._path``.  That proxy belongs to Torch's custom-class
# registry, so the probe raises a harmless but noisy RuntimeError.  Supplying an
# empty package path prevents the invalid introspection without affecting model
# loading or inference.
try:
    import torch

    torch.classes.__path__ = []
except Exception:
    # Torch is optional: the app can still use its classical prediction fallback.
    pass

import pandas as pd
import streamlit as st

from src.cascade import ClassicalFallback, ModelCascade
from src.config import APP_VERSION, BEST_MODEL_PATH, CONFIDENCE_THRESHOLD, MIN_SYMPTOMS_FOR_PREDICTION, SHORT_DISCLAIMER
from src.conversation import ConversationEngine
from src.data_pipeline import display_symptom, load_data_bundle, normalize_key
from src.enhancements import FEEDBACK_STORE, GLOBAL_RATE_LIMITER
from src.nlp import HuggingFaceSymptomExtractor, SymptomMapper
from src.ops import export_doctor_summary, export_markdown, track_event
from src.safety import scrub_pii, triage_message
from src.tensorflow_predictor import TensorFlowDiseasePredictor
from src.predictor import DiseasePredictor
from src.kb_predictor import KnowledgeBasePredictor
from src.data_pipeline import load_kb_data_bundle
from src.monitoring import record_prediction
from src.database import record_prediction as persist_prediction
from src.dataset_audit import audit_bundle, audit_frame
from src.patient_history import (
    MIN_HISTORY_KEY_LENGTH,
    delete_all_assessments,
    delete_assessment,
    list_assessments,
    save_assessment,
    update_assessment_payload,
    update_follow_up,
)
from src.appointments import create_appointment, list_appointments, delete_appointment

st.set_page_config(
    page_title="MediGuide AI",
    page_icon=Image.open(FAVICON_PATH) if FAVICON_PATH.exists() else "🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _full_width(stretch: bool = True) -> dict[str, Any]:
    """Compatibility helper: uses width='stretch' on modern Streamlit (>=1.42)
    to prevent deprecation warnings, falling back to use_container_width=True on older versions.
    """
    try:
        ver_parts = [int(p) for p in st.__version__.split(".")[:2] if p.isdigit()]
        if len(ver_parts) >= 2 and tuple(ver_parts) >= (1, 42):
            return {"width": "stretch" if stretch else "content"}
    except Exception:
        pass
    return {"use_container_width": stretch}

# ---------------------------------------------------------------------------
# Onboarding content
# ---------------------------------------------------------------------------



def load_runtime():
    """Load KB data, Random Forest, optional TensorFlow, HF NLP extractor, ensemble."""
    data = load_data_bundle()
    kb = KnowledgeBasePredictor()
    rf = None
    try:
        from src.rf_predictor import RandomForestPredictor
        rf = RandomForestPredictor.load()
        rf.validate_compatibility(data)
    except Exception as exc:
        track_event("rf_unavailable", error_type=type(exc).__name__)
    tf_model = None
    try:
        from src.tensorflow_predictor import TensorFlowDiseasePredictor
        # Inference processes must never train or rewrite model artifacts at startup.
        # Train and validate models offline, then deploy immutable versioned artifacts.
        try:
            tf_model = TensorFlowDiseasePredictor.load(data)
        except Exception as load_exc:
            track_event("tensorflow_unavailable", error_type=type(load_exc).__name__)
            tf_model = None
    except Exception as exc:
        track_event("tensorflow_unavailable", error_type=type(exc).__name__)
        tf_model = None
    from src.ensemble import EnsemblePredictor
    from src.profile_predictor import DatasetProfilePredictor
    profile_predictor = DatasetProfilePredictor(data)
    class_support = data.targets.value_counts().to_dict()
    ensemble = EnsemblePredictor(rf=rf, tensorflow=tf_model, kb=kb, profile=profile_predictor, class_support=class_support)

    # The previous classical artifact is only valid for the exact same feature
    # and class space. Never display its old metrics as if they belonged to the
    # expanded merged corpus.
    ensemble.metadata = dict(getattr(rf, "meta", {}) or profile_predictor.metadata)
    ensemble.metadata.setdefault("class_count", data.class_count)
    ensemble.metadata.setdefault("feature_count", len(data.feature_names))
    ensemble.metadata["catalogue_label_count"] = data.class_count
    ensemble.metadata["labels_meeting_minimum_support"] = profile_predictor.metadata["class_count"]
    ensemble.metadata["labels_below_minimum_support"] = profile_predictor.metadata["excluded_low_support_count"]
    ensemble.metadata["minimum_training_examples_per_class"] = profile_predictor.metadata["minimum_training_examples_per_class"]
    try:
        import joblib as _jl
        _art = _jl.load(BEST_MODEL_PATH)
        artifact_matches = (
            _art.get("feature_names") == data.feature_names
            and sorted(_art.get("classes", [])) == sorted(data.targets.unique().tolist())
        )
        if artifact_matches:
            _comp = _art.get("metrics") or []
            _best_row = _comp[0] if _comp else {}
            ensemble.test_accuracy = _art.get("top_1_accuracy")
            ensemble.test_f1_weighted = _best_row.get("f1_weighted")
            ensemble.test_f1_macro = _best_row.get("f1_macro")
            ensemble.cv_f1_weighted_mean = _best_row.get("cv_f1_weighted_mean")
            ensemble.top_3_accuracy = _art.get("top_3_accuracy")
            ensemble.brier_score = _art.get("brier_score")
            ensemble.roc_auc_ovr_macro = _art.get("roc_auc_ovr_macro")
            ensemble.ece = _art.get("expected_calibration_error")
            ensemble.fnr_macro = _art.get("false_negative_rate_macro")
            ensemble.metadata.update({
                "framework": "scikit-learn (" + str(_art.get("model_name", "classical model")) + ")",
                "model_version": APP_VERSION,
                "schema_version": str(_art.get("schema_version", "—")),
                "test_accuracy": _art.get("top_1_accuracy"),
                "top_3_accuracy": _art.get("top_3_accuracy"),
                "brier_score": _art.get("brier_score"),
                "roc_auc_ovr_macro": _art.get("roc_auc_ovr_macro"),
                "ece": _art.get("expected_calibration_error"),
                "fnr_macro": _art.get("false_negative_rate_macro"),
                "test_f1_macro": _best_row.get("f1_macro"),
                "test_f1_weighted": _best_row.get("f1_weighted"),
                "train_samples": _art.get("training_samples"),
                "test_samples": _art.get("test_samples"),
                "artifact_status": "Trained at " + str(_art.get("trained_at_utc", "—"))[:19].replace("T", " ") + " UTC",
            })
    except Exception:
        # The profile fallback remains fully functional without optional metrics.
        pass

    extractor = HuggingFaceSymptomExtractor(SymptomMapper(data.feature_names))
    return data, ensemble, tf_model, extractor

@st.cache_resource(show_spinner=False)
def get_runtime():
    return load_runtime()

def new_chat() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "messages": [],
        "symptoms": set(),
        "symptoms_before_last": set(),
        "triage_level": None,
        "triage_message": None,
        "last_prediction": None,
        "feedback_given": False,
        "context": {
            "age": "Not provided",
            "sex": "Not provided",
            "pregnancy": "Not provided",
            "duration": 0,
            "severity": "Not provided",
            "worsening": False,
        },
        "created_at": datetime.now().strftime("%b %d, %H:%M"),
        "status": "Ready",
    }

def initialize_session() -> None:
    if "chats" not in st.session_state:
        st.session_state.chats = [new_chat()]
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = st.session_state.chats[0]["id"]
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def active_chat() -> dict:
    return next(item for item in st.session_state.chats if item["id"] == st.session_state.active_chat)

def render_styles() -> None:
    st.markdown(
        """
    <style>
    :root {
      --bg: #f5f8fa;
      --surface: #ffffff;
      --ink: #172b3a;
      --muted: #617483;
      --line: #dce6eb;
      --accent: #087f72;
      --accent-soft: #e3f4f0;
      --danger: #8b1e1e;
      --danger-bg: #fdecec;
      --warn: #8a5a00;
      --warn-bg: #fff6e5;
      --ok: #1e5c3a;
      --ok-bg: #e8f5ee;
    }
    .stApp { background: var(--bg); color: var(--ink); }
    [data-testid="stAppViewContainer"] { background: var(--bg); }
    section[data-testid="stSidebar"] { background: #edf4f5; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    div[data-testid="stTabs"] > div[role="tablist"] { gap: 0.35rem; border-bottom: 1px solid var(--line); }
    div[data-testid="stTabs"] button[role="tab"] { border-radius: 11px 11px 0 0; padding: 0.7rem 1rem; font-weight: 600; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 0.8rem 1rem; }
    div[data-testid="stExpander"] { background: rgba(255,255,255,0.82); border: 1px solid var(--line); border-radius: 12px; }
    div[data-testid="stButton"] > button { border-radius: 10px; font-weight: 550; transition: transform .14s ease, box-shadow .14s ease; }
    div[data-testid="stButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(15, 55, 70, .09); }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea, div[data-testid="stSelectbox"] div[role="combobox"] { border-radius: 10px; }
    /* Sidebar chat item styling: prominent active indicator vs inactive */
    div[data-testid="stSidebar"] div[class*="st-key-chat-"] button {
      text-align: left !important;
      justify-content: flex-start !important;
      border-radius: 8px !important;
      padding: 0.5rem 0.75rem !important;
      font-size: 0.86rem !important;
      transition: all 0.15s ease-in-out !important;
      margin-bottom: 2px !important;
    }
    div[data-testid="stSidebar"] div[class*="st-key-chat-"] button[kind="primary"] {
      background: linear-gradient(135deg, #087f72 0%, #11998b 100%) !important;
      color: #ffffff !important;
      border: 1px solid #087f72 !important;
      border-left: 5px solid #2ec4b6 !important;
      font-weight: 650 !important;
      box-shadow: 0 2px 8px rgba(8, 127, 114, 0.2) !important;
    }
    div[data-testid="stSidebar"] div[class*="st-key-chat-"] button[kind="secondary"] {
      background: #ffffff !important;
      color: #334155 !important;
      border: 1px solid #d8dee6 !important;
      border-left: 3px solid transparent !important;
      font-weight: 450 !important;
    }
    div[data-testid="stSidebar"] div[class*="st-key-chat-"] button[kind="secondary"]:hover {
      background: #f8fafc !important;
      border-color: #cbd5e1 !important;
      border-left-color: #94a3b8 !important;
      color: #087f72 !important;
    }
    div[data-testid="stSidebar"] div[class*="st-key-delete-chat-"] button {
      padding: 0.5rem 0.2rem !important;
      text-align: center !important;
      justify-content: center !important;
      border-radius: 8px !important;
      color: var(--muted) !important;
    }
    div[data-testid="stSidebar"] div[class*="st-key-delete-chat-"] button:hover {
      color: var(--danger) !important;
      background: var(--danger-bg) !important;
      border-color: #f5c6cb !important;
    }
    h1, h2, h3 { color: var(--ink) !important; letter-spacing: -0.01em; }
    .brand {
      display: flex; align-items: center; gap: 14px;
      padding: 12px 16px 15px; border: 1px solid var(--line); margin-bottom: 16px;
      margin-top: 0.25rem; background: rgba(255,255,255,.92); border-radius: 16px;
      box-shadow: 0 4px 18px rgba(22, 43, 58, .035);
    }
    .brand-mark {
      width: 36px; height: 36px; border-radius: 8px;
      background: var(--accent-soft); border: 1px solid #b6e0d8;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.25rem;
    }
    .brand-logo {
      width: 44px; height: 44px; border-radius: 12px;
      object-fit: cover;
      box-shadow: 0 2px 8px rgba(15, 76, 92, 0.15);
      border: 1px solid #cce3e6;
    }
    .brand-name { font-size: 1.28rem; font-weight: 750; color: var(--ink); letter-spacing: -.02em; }
    .brand-sub { font-size: 0.84rem; color: var(--muted); margin-top: 3px; }
    .trust-bar {
      background: var(--accent-soft); border: 1px solid #b6e0d8;
      border-radius: 8px; padding: 10px 12px; font-size: 0.82rem;
      color: var(--ink); margin: 0 0 14px;
    }
    .trust-bar strong { color: var(--accent); }
    .result {
      background: var(--surface); border: 1px solid var(--line);
      border-radius: 14px; padding: 18px 20px; margin: 12px 0 16px; box-shadow: 0 3px 14px rgba(20,45,60,.035);
    }
    .result-label {
      font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted); font-weight: 600; margin-bottom: 4px;
    }
    .result-title { font-size: 1.2rem; font-weight: 650; color: var(--ink); margin-bottom: 4px; }
    .result-meta { font-size: 0.82rem; color: var(--muted); }
    .result-badges { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
    .badge {
      font-size: 0.72rem; padding: 3px 8px; border-radius: 999px;
      background: var(--accent-soft); color: var(--accent); border: 1px solid #b7d0d6;
    }
    .triage {
      border-radius: 8px; padding: 12px 14px; margin: 10px 0;
      border: 1px solid var(--line); font-size: 0.88rem;
    }
    .triage.emergency { background: var(--danger-bg); border-color: #e3b4b4; color: var(--danger); }
    .triage.same-day { background: var(--warn-bg); border-color: #efd9a8; color: var(--warn); }
    .triage.routine, .triage.monitor { background: var(--ok-bg); border-color: #b9dcc8; color: var(--ok); }
    .safety {
      background: #f8f9fb; border: 1px solid var(--line); border-radius: 8px;
      padding: 10px 12px; font-size: 0.8rem; color: var(--muted); margin-top: 8px;
    }
    .footer {
      margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--line);
      font-size: 0.75rem; color: var(--muted); line-height: 1.45;
    }
    /* Prominent, Elevated Chat Input */
    div[data-testid="stChatInput"] {
      padding-bottom: 1.2rem;
    }
    div[data-testid="stChatInput"] > div {
      background: #ffffff !important;
      border: 1.5px solid #b6dcd6 !important;
      border-radius: 16px !important;
      box-shadow: 0 7px 26px rgba(8, 127, 114, 0.10), 0 2px 8px rgba(0, 0, 0, 0.035) !important;
      transition: all 0.2s ease-in-out !important;
      padding: 4px 6px !important;
    }
    div[data-testid="stChatInput"] > div:hover,
    div[data-testid="stChatInput"] > div:focus-within {
      border-color: #087f72 !important;
      box-shadow: 0 9px 30px rgba(8, 127, 114, 0.16), 0 3px 10px rgba(0, 0, 0, 0.06) !important;
    }
    div[data-testid="stChatInput"] textarea {
      font-size: 1.05rem !important;
      line-height: 1.5 !important;
      padding: 12px 14px !important;
      min-height: 56px !important;
      color: var(--ink) !important;
      border-radius: 10px !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
      font-size: 0.96rem !important;
      color: #64748b !important;
    }
    div[data-testid="stChatInput"] button {
      background-color: #087f72 !important;
      color: #ffffff !important;
      border-radius: 10px !important;
      width: 42px !important;
      height: 42px !important;
      align-self: center !important;
      margin-right: 4px !important;
      transition: transform 0.15s ease, background-color 0.15s ease !important;
    }
    div[data-testid="stChatInput"] button:hover {
      background-color: #06675e !important;
      transform: scale(1.05) !important;
    }
    div[data-testid="stChatInput"] button svg {
      fill: #ffffff !important;
    }
    .block-container { padding-top: 1.8rem !important; max-width: 1320px; padding-left: clamp(1rem, 3vw, 2.5rem); padding-right: clamp(1rem, 3vw, 2.5rem); }
    .welcome-card { background: linear-gradient(135deg, #e5f5f1 0%, #f7fbfc 76%); border: 1px solid #cce7e1; border-radius: 18px; padding: 24px 26px; margin: 8px 0 18px; }
    .welcome-kicker { color: #087f72; text-transform: uppercase; letter-spacing: .09em; font-size: .72rem; font-weight: 750; margin-bottom: 7px; }
    .welcome-title { color: #172b3a; font-size: clamp(1.7rem, 3vw, 2.35rem); line-height: 1.16; font-weight: 780; letter-spacing: -.04em; margin: 0 0 8px; }
    .welcome-copy { color: #536b78; font-size: 1rem; max-width: 760px; line-height: 1.6; margin: 0; }
    .section-eyebrow { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; margin: 1rem 0 .4rem; }
    .sidebar-overflow-note { color: var(--muted); font-size: .75rem; padding: .35rem .5rem; }
    .processing-pill { display: inline-flex; align-items: center; gap: .55rem; border: 1px solid #c5e5de; background: #e7f6f2; color: #086d62; border-radius: 999px; padding: .45rem .8rem; font-size: .83rem; font-weight: 650; margin: .4rem 0 .8rem; }
    .processing-dot { width: .48rem; height: .48rem; border-radius: 999px; background: #0a9b89; box-shadow: 0 0 0 4px rgba(10,155,137,.12); }
    @media (max-width: 720px) { .block-container { padding-top: .8rem !important; padding-left: .7rem; padding-right: .7rem; } .brand { padding: 10px 12px; } .welcome-card { padding: 18px 16px; border-radius: 14px; } .brand-sub { font-size: .75rem; } div[data-testid="stTabs"] button[role="tab"] { padding: .55rem .55rem; font-size: .82rem; } div[data-testid="stChatInput"] textarea { font-size: 1rem !important; } }
    .gauge {
      margin: -2px 0 14px;
      padding: 0 1px;
    }
    .gauge-track {
      position: relative;
      height: 6px;
      background: var(--line);
      border-radius: 999px;
      overflow: visible;
    }
    .gauge-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    .gauge-fill.low {
      background: #c27803;
    }
    .gauge-tick {
      position: absolute;
      top: -2.5px;
      width: 2px;
      height: 11px;
      background: #334155;
      border-radius: 1px;
      transform: translateX(-50%);
      z-index: 2;
    }
    .gauge-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.74rem;
      color: var(--muted);
      margin-top: 5px;
    }
    .gauge-meta b {
      color: var(--ink);
      font-weight: 600;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

def render_brand() -> None:
    logo_src = get_logo_base64()
    icon_html = f'<img src="{logo_src}" class="brand-logo" alt="MedGuide Logo">' if logo_src else '<div class="brand-mark">🩺</div>'
    st.markdown(
        f'<div class="brand">{icon_html}'
        f'<div><div class="brand-name">MediGuide AI</div>'
        f'<div class="brand-sub">A clearer way to explore symptoms and prepare for care</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="trust-bar"><strong>Health information, not a diagnosis.</strong> '
        'Describe what you are experiencing in your own words. MediGuide can explain possible symptom patterns, '
        'but it cannot confirm a condition or replace a clinician. For severe or rapidly worsening symptoms, '
        'seek urgent local medical care.</div>',
        unsafe_allow_html=True,
    )

def render_triage(level: str | None, message: str | None) -> None:
    if not level:
        return
    key = level.lower()
    if "emergency" in key:
        css = "emergency"
    elif "same" in key:
        css = "same-day"
    elif "routine" in key:
        css = "routine"
    else:
        css = "monitor"
    role = ' role="alert" aria-live="assertive"' if css == "emergency" else ""
    st.markdown(
        f'<div class="triage {css}"{role}>'
        f"<strong>{html.escape(level)}</strong> "
        f"{html.escape(message or '')}</div>",
        unsafe_allow_html=True,
    )

def render_sidebar(data, tensorflow) -> None:
    with st.sidebar:
        logo_src = get_logo_base64()
        if logo_src:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                f'<img src="{logo_src}" style="width:34px;height:34px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.08);" alt="MedGuide">'
                f'<span style="font-size:1.35rem;font-weight:700;color:var(--ink);">MedGuide AI</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("# 🩺 MedGuide AI Assistant")
        st.caption("A conversational symptom guide · not a diagnostic service")
        if st.button("＋  Start a new chat", **_full_width(), type="primary") :
            chat = new_chat()
            st.session_state.chats.insert(0, chat)
            st.session_state.active_chat = chat["id"]
            st.rerun()

        st.markdown("#### Your conversations")
        recent = list(st.session_state.chats[:6])
        hidden = max(0, len(st.session_state.chats) - len(recent))
        for chat in recent:
            active = chat["id"] == st.session_state.active_chat
            label = f"🩺 {chat['title'][:22]}  (Active)" if active else f"💬 {chat['title'][:26]}"
            chat_column, delete_column = st.columns([0.84, 0.16])
            with chat_column:
                if st.button(
                    label,
                    key=f"chat-{chat['id']}",
                    type="primary" if active else "secondary",
                    help="Currently active conversation" if active else "Switch to this conversation",
                    **_full_width(),
                ):
                    st.session_state.active_chat = chat["id"]
                    st.rerun()
            with delete_column:
                if st.button("×", key=f"delete-chat-{chat['id']}", help="Delete Chat", **_full_width()):
                    remaining = [item for item in st.session_state.chats if item["id"] != chat["id"]]
                    if not remaining:
                        remaining = [new_chat()]
                    st.session_state.chats = remaining
                    if st.session_state.active_chat == chat["id"]:
                        st.session_state.active_chat = remaining[0]["id"]
                    st.rerun()
        if hidden:
            st.markdown(
                f'<div class="sidebar-overflow-note">+{hidden} older Chat'
                f'{"s" if hidden != 1 else ""} not shown. Start a new one above to keep your recent list focused.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Privacy & data" ,expanded=True):
            st.caption("Chat messages stay in the active session unless you explicitly save an assessment. Avoid names, phone numbers, addresses, or medical-record numbers.")
            st.caption("Saved assessment payloads are encrypted in the app server's local database. The app does not store your history key; anyone with server/database access may still control the storage environment.")
        with st.expander("Private saved history", expanded=False):
            st.text_input(
                "Private history key",
                type="password",
                key="history_key_input",
                help=f"Use a unique passphrase of at least {MIN_HISTORY_KEY_LENGTH} characters. Keep it safe; it is not recoverable by the app.",
            )
            st.checkbox(
                "I understand saved health notes are stored encrypted on the app host.",
                key="history_storage_consent",
            )
            st.caption("History is opt-in. Your key is kept only in the active app session and is never written to the database.")

        with st.expander("About" ,expanded=False):
            st.caption(
                "MedGuide is an experimental educational symptom-pattern assistant. It uses biomedical NLP and symptom-profile matching to display possible patterns, with conservative keyword-based safety prompts. Those prompts are not a validated triage system, and model scores are not diagnostic probabilities. It does not diagnose, prescribe, or replace a licensed clinician."
            )
            st.caption(
                "Educational pattern assistant only. Not a diagnosis or prescribing service. "
                "Do not start, stop, or change medicine based on this app."
            )
            # Live counts -- always reflect the trained model artifact
            meta = getattr(tensorflow, "metadata", {}) or {}
            n_conditions = data.class_count
            n_symptoms = len(data.feature_names)
            supported = getattr(getattr(classical, "profile", None), "metadata", {}).get("class_count", 0)
            st.caption(f"{n_conditions} catalogue labels · {supported} meet the minimum dataset-support threshold · {n_symptoms} symptom features")

def context_controls(chat: dict) -> None:
    with st.expander("Add details (optional) — age / sex / pregnancy are context only", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ages = ["Not provided", "Child", "Adult", "Older adult"]
            chat["context"]["age"] = st.selectbox(
                "Age group", ages,
                index=ages.index(chat["context"].get("age", "Not provided")),
                key=f"age-{chat['id']}",
            )
            sevs = ["Not provided", "Mild", "Moderate", "Severe"]
            chat["context"]["severity"] = st.selectbox(
                "Severity", sevs,
                index=sevs.index(chat["context"].get("severity", "Not provided")),
                key=f"severity-{chat['id']}",
            )
        with c2:
            sexes = ["Not provided", "Male", "Female"]
            chat["context"]["sex"] = st.selectbox(
                "Sex", sexes,
                index=sexes.index(chat["context"].get("sex", "Not provided")),
                key=f"sex-{chat['id']}",
            )
            chat["context"]["duration"] = st.number_input(
                "Duration (days)", min_value=0, max_value=365,
                value=int(chat["context"].get("duration", 0) or 0),
                key=f"duration-{chat['id']}",
            )
        with c3:
            pregs = ["Not provided", "No", "Possibly", "Yes"]
            chat["context"]["pregnancy"] = st.selectbox(
                "Pregnant?", pregs,
                index=pregs.index(chat["context"].get("pregnancy", "Not provided")),
                key=f"preg-{chat['id']}",
                help="Used only to suppress incompatible ranks. Not a pregnancy test.",
            )
            chat["context"]["worsening"] = st.checkbox(
                "Getting worse", value=bool(chat["context"].get("worsening", False)),
                key=f"worsening-{chat['id']}",
            )

def render_feedback(chat: dict) -> None:
    prediction = chat.get("last_prediction")
    if not prediction or chat.get("feedback_given"):
        return
    st.caption("Was this result useful?")
    a, b = st.columns(2)
    with a:
        if st.button("Helpful", key=f"up-{chat['id']}", **_full_width()):
            FEEDBACK_STORE.add("up", prediction["disease"])
            chat["feedback_given"] = True
            st.toast("Thanks for the feedback.")
            st.rerun()
    with b:
        if st.button("Not useful", key=f"down-{chat['id']}", **_full_width()):
            FEEDBACK_STORE.add("down", prediction["disease"])
            chat["feedback_given"] = True
            st.toast("Thanks for the feedback.")
            st.rerun()

# ---------------------------------------------------------------------------
# New: confidence gauge, symptom chips, onboarding, explainability, processing pill
# ---------------------------------------------------------------------------

def render_confidence_gauge(confidence: float | int | None) -> None:
    if not isinstance(confidence, (float, int)):
        return
    pct = max(0.0, min(1.0, float(confidence)))
    threshold_pct = max(0.0, min(1.0, CONFIDENCE_THRESHOLD))
    is_low = pct < threshold_pct
    width = f"{pct * 100:.1f}%"
    tick_left = f"{threshold_pct * 100:.1f}%"
    fill_class = "gauge-fill low" if is_low else "gauge-fill"
    st.markdown(
        f'<div class="gauge" aria-label="Model confidence">'
        f'  <div class="gauge-track" role="progressbar" '
        f'       aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pct * 100:.0f}">'
        f'    <div class="{fill_class}" style="width:{width};"></div>'
        f'    <div class="gauge-tick" style="left:{tick_left};" title="Display threshold ({threshold_pct:.0%})"></div>'
        f'  </div>'
        f'  <div class="gauge-meta">'
        f'    <span><b>{pct:.0%}</b> dataset match strength</span>'
        f'    <span>Threshold <b>{threshold_pct:.0%}</b></span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_processing_pill(chat: dict) -> None:
    if chat.get("status") == "Analyzing symptoms":
        st.markdown(
            '<div class="processing-pill" role="status" aria-live="polite">'
            '<span class="processing-dot" aria-hidden="true"></span>'
            'Analyzing your symptoms…'
            '</div>',
            unsafe_allow_html=True,
        )

def render_onboarding_chips() -> None:
    """Conversational example prompts; each starts a chat turn, not a symptom form."""
    prompts = [
        ("I have had a headache since yesterday", "Headache"),
        ("I have a fever and a dry cough", "Fever & cough"),
        ("I feel tired and dizzy lately", "Fatigue & dizziness"),
    ]
    st.markdown('<div class="section-eyebrow">Not sure how to start? Try a message</div>', unsafe_allow_html=True)
    cols = st.columns(len(prompts))
    for column, (prompt_text, label) in zip(cols, prompts):
        with column:
            if st.button(label, key=f"onboard-{label}", **_full_width()):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

def render_explainability(prediction: dict, chat: dict, data) -> None:
    """Show which user-reported symptoms drove the top match."""
    symptoms: set[str] = set(chat.get("symptoms") or set())
    if not symptoms:
        st.info("No structured symptoms were retained for this turn, so a symptom-by-symptom explanation is unavailable.")
        return
    raw_disease = prediction.get("disease", "")
    if not raw_disease:
        st.info("No condition pattern is available to explain.")
        return
    canonical = data.canonical_disease(raw_disease)
    profile_key = next(
        (idx for idx in data.disease_profiles.index if normalize_key(idx) in {normalize_key(raw_disease), normalize_key(canonical)}),
        None,
    )
    profile = data.disease_profiles.loc[profile_key] if profile_key is not None else None
    if profile is None:
        st.info("No matching symptom profile is available for this label in the current dataset.")
        return

    contributions: list[tuple[str, float]] = []
    for feature in symptoms:
        if feature not in profile.index:
            continue
        prevalence = float(profile.get(feature, 0.0))
        weight = float(data.symptom_weights.get(feature, 1.0))
        score = prevalence * weight
        if score > 0:
            contributions.append((display_symptom(feature), score))
    contributions.sort(key=lambda item: item[1], reverse=True)
    if not contributions:
        st.info("None of the tracked symptoms directly matched this condition's stored symptom profile. This does not rule the condition in or out.")
        return

    items = []
    for label, score in contributions[:6]:
        items.append(
            f'<li><span>{html.escape(label)}</span>'
            f'<span class="explain-score">{score:.2f}</span></li>'
        )
    st.markdown(
        f'<ul class="explain-list" aria-label="Symptoms contributing to {html.escape(canonical)}">'
        + "".join(items)
        + "</ul>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="explain-note">'
        'Scores combine how often each symptom appears in the training rows for this '
        'condition with the symptom’s clinical weight. They are explainability hints, '
        'not causal evidence.'
        '</div>',
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def _load_condition_evidence_rows() -> list[dict[str, Any]]:
    path = Path(__file__).parent / "models" / "condition_evidence.csv"
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        return frame.to_dict(orient="records")
    except Exception:
        return []


def _condition_evidence(condition_name: str) -> dict[str, Any] | None:
    target = normalize_key(condition_name)
    for row in _load_condition_evidence_rows():
        if normalize_key(row.get("condition", "")) == target or normalize_key(row.get("condition_id", "")) == target:
            return row
    return None


def render_condition_details(condition_name: str, data) -> None:
    """Render transparent, dataset-backed details without inventing missing advice."""
    if data is None:
        return
    canonical = data.canonical_disease(condition_name) or condition_name
    support = data.get_support(canonical)
    with st.expander(f"Detailed information: {canonical}"):
        if support.description:
            st.markdown("**Overview**")
            st.write(support.description)
        else:
            st.info("A condition-specific, medically reviewed description is not available in this release.")

        profile = getattr(data, "disease_profiles", None)
        profile_key = next((idx for idx in profile.index if str(idx).casefold() == canonical.casefold()), None) if profile is not None and not profile.empty else None
        if profile_key is not None:
            series = profile.loc[profile_key].sort_values(ascending=False)
            associated = [(str(symptom).replace("_", " "), float(value)) for symptom, value in series.items() if float(value) > 0]
            if associated:
                st.markdown("**Symptoms associated with this label in the training dataset**")
                st.write(", ".join(f"{symptom} ({weight:.0%} of generated examples)" for symptom, weight in associated[:8]))
                st.caption("These are dataset associations, not validated diagnostic criteria or probabilities.")

        if support.precautions:
            st.markdown("**Precautions / when to seek care**")
            for item in support.precautions[:8]:
                st.markdown(f"- {item}")
        if support.medications:
            st.markdown("**Educational medication information**")
            for item in support.medications[:6]:
                st.markdown(f"- {item}")
            st.caption("This is general educational content, not a prescription. Confirm medicines with a qualified clinician or pharmacist.")
        if support.diet:
            st.markdown("**General supportive guidance**")
            for item in support.diet[:6]:
                st.markdown(f"- {item}")
        evidence = _condition_evidence(canonical)
        if evidence:
            st.markdown("**Evidence and review status**")
            review_status = str(evidence.get("evidence_review_status", "Not supplied"))
            validation_status = str(evidence.get("clinical_validation_status", "Not supplied"))
            st.caption(f"Source review: {review_status}. Clinical validation: {validation_status}.")
            reference_url = str(evidence.get("reference_url", "")).strip()
            if reference_url.startswith("https://"):
                st.markdown(f"[Open condition reference]({reference_url})")
            elif reference_url:
                st.caption(f"Reference URL supplied: {reference_url}")
        else:
            st.caption("No condition-specific source record is bundled for this label. The symptom profile is not clinically validated.")

        if not any((support.description, support.precautions, support.medications, support.diet)):
            st.warning("No condition-specific detail has been verified for this label. Do not infer a diagnosis from the ranking alone.")


def render_result(chat: dict, data) -> None:
    prediction = chat.get("last_prediction")
    if not prediction:
        return
    raw_name = str(prediction.get("disease", ""))
    confidence = prediction.get("confidence")
    forced = prediction.get("forced_nonspecific") or raw_name.startswith("Nonspecific")
    if forced or raw_name.startswith("Nonspecific"):
        label = "Insufficient specificity"
        display_name = html.escape("Nonspecific / common self-limiting pattern")
    elif "seek emergency" in raw_name.lower():
        label = "Urgent pattern — seek emergency care"
        display_name = html.escape(raw_name)
    else:
        label = "Possible educational pattern"
        canonical = data.canonical_disease(raw_name) if data else raw_name
        display_name = html.escape(canonical or raw_name)

    if isinstance(confidence, (float, int)):
        if confidence >= 0.7:
            strength = "Stronger pattern match"
        elif confidence >= 0.45:
            strength = "Moderate pattern match"
        else:
            strength = "Weak pattern match"
        conf_txt = f"{strength} · relative model score, not a probability"
    else:
        conf_txt = "Pattern ranking complete"

    badges = []
    if prediction.get("clinical_adjusted"):
        badges.append("clinical ranking applied")
    if prediction.get("route"):
        badges.append(str(prediction.get("route")).replace("+", "·"))
    if prediction.get("age_group"):
        badges.append(f"age: {prediction['age_group']}")
    if prediction.get("sex"):
        badges.append(f"sex: {prediction['sex']}")
    if prediction.get("pregnancy") and prediction.get("pregnancy") not in {"No", "Not provided", None}:
        badges.append(f"pregnancy: {prediction['pregnancy']}")
    badge_html = "".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges)

    st.markdown(
        f'<div class="result">'
        f'<div class="result-label">{label}</div>'
        f'<div class="result-title">{display_name}</div>'
        f'<div class="result-meta">{html.escape(conf_txt)}</div>'
        f'<div class="result-badges">{badge_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if isinstance(confidence, (float, int)) and not forced:
        st.caption("The model score is a relative ranking signal. It is not a calibrated probability, diagnostic confidence, or estimate of your chance of having this condition.")
    if not forced:
        with st.expander("Why this pattern was suggested", expanded=True):
            st.caption("The items below are symptoms you reported that also appear in this condition's training profile. They are descriptive matches, not proof of cause or diagnosis.")
            render_explainability(prediction, chat, data)

    alternatives = prediction.get("alternatives") or []
    if alternatives:
        with st.expander("Alternative patterns to consider", expanded=True):
            rows = []
            for name, prob in alternatives[:5]:
                rows.append({
                    "Possible pattern": data.canonical_disease(name) if data else name,
                    "Relative match score (not probability)": f"{float(prob):.0%}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, **_full_width())
            st.caption(
                "These are comparison scores from the available symptom profiles, not calibrated "
                "chances of having a disease. A clinician may need examination or testing to distinguish them."
            )

    # Make condition-specific detail available consistently for the leading result
    # and the main alternatives. Missing evidence is disclosed rather than fabricated.
    detail_names = []
    if raw_name and not forced and not raw_name.startswith("Nonspecific"):
        detail_names.append(raw_name)
    for name, _score in alternatives[:4]:
        if name and name not in detail_names and not str(name).startswith("Nonspecific"):
            detail_names.append(name)
    if detail_names:
        st.markdown("#### Detailed condition information")
        for condition_name in detail_names:
            render_condition_details(condition_name, data)

    with st.expander("How this ranking was produced"):
        st.markdown(
            "Symptoms are extracted with biomedical NLP (including optional Hugging Face NER / "
            "BioClinicalBERT), then compared against curated symptom profiles. Age, sex, and pregnancy "
            "are retained as context but are not used to rule out conditions or reweight disease scores. Machine-learning artifacts are "
            "fallbacks only. Scores reflect similarity to educational patterns — **not** diagnostic probability. "
            "This release's training examples are synthetic and have not been clinically validated."
        )

    render_triage(chat.get("triage_level"), chat.get("triage_message"))


def run_prediction(data, classical, tensorflow, extractor, chat: dict) -> None:
    engine = ConversationEngine(data, classical, extractor)
    prompt = chat["messages"][-1]["content"]
    ctx = chat.get("context") or {}
    age_group = ctx.get("age") if ctx.get("age") not in {None, "Not provided", ""} else None
    sex = ctx.get("sex") if ctx.get("sex") not in {None, "Not provided", ""} else None
    pregnancy = ctx.get("pregnancy") if ctx.get("pregnancy") not in {None, "Not provided", ""} else None
    duration = ctx.get("duration") or 0
    severity = ctx.get("severity") if ctx.get("severity") not in {None, "Not provided", ""} else None
    worsening = bool(ctx.get("worsening", False))
    turn = engine.respond(
        prompt,
        chat["symptoms"],
        age_group=age_group,
        sex=sex,
        pregnancy=pregnancy,
        duration_days=duration,
        severity=severity,
        worsening=worsening,
    )

    chat["symptoms"] = set(turn.accumulated_symptoms)
    # Store clarifying-question suggestions for the quick-reply chips
    chat["last_suggested_symptoms"] = list(turn.suggested_symptoms)
    triage = triage_message(prompt, set(turn.accumulated_symptoms), turn.extraction.context)
    chat["triage_level"] = triage.level.value
    chat["triage_message"] = triage.message
    if turn.prediction:
        chat["last_prediction"] = {
            "disease": turn.prediction.disease,
            "confidence": turn.prediction.confidence,
            "route": getattr(turn.prediction, "route", "classical + clinical"),
            "alternatives": turn.prediction.alternatives,
            "latency_ms": getattr(turn.prediction, "latency_ms", None),
            "clinical_adjusted": getattr(turn.prediction, "clinical_adjusted", False),
            "age_group": getattr(turn.prediction, "age_group", None),
            "sex": getattr(turn.prediction, "sex", None),
            "pregnancy": getattr(turn.prediction, "pregnancy", None),
            "forced_nonspecific": getattr(turn.prediction, "forced_nonspecific", False),
        }
        latency = getattr(turn.prediction, "latency_ms", None)
        record_prediction(getattr(turn.prediction, "route", "classical + clinical"), latency)
        persist_prediction(getattr(turn.prediction, "route", "classical + clinical"), turn.prediction.confidence, latency)
        chat["status"] = "Completed"
    else:
        chat["last_prediction"] = None
        chat["status"] = "Needs more information"
    if turn.reply:
        chat["messages"].append({"role": "assistant", "content": turn.reply})



def run_with_processing_state(data, classical, tensorflow, extractor, chat: dict) -> None:
    """Render immediate, user-facing feedback while the Chat is evaluated."""
    chat["status"] = "Analyzing symptoms"
    with st.status("Analyzing your symptoms", expanded=False) as processing:
        st.write("Reading your symptom description")
        st.write("Matching supported symptom patterns")
        run_prediction(data, classical, tensorflow, extractor, chat)
        st.write("Checking the model result and safety guidance")
        processing.update(label="Analysis complete", state="complete", expanded=False)

def render_chat(data, classical, tensorflow, extractor) -> None:
    chat = active_chat()

    render_processing_pill(chat)

    is_empty = not chat["messages"] and not chat.get("last_prediction")
    if is_empty:
        st.markdown(
            '<div class="welcome-card">'
            '<div class="welcome-kicker">Your health companion</div>'
            '<div class="welcome-title">Tell me what you are feeling.</div>'
            '<p class="welcome-copy">Use your own words, just as you would with a person. Mention when it started, how it feels, '
            'or whether it is getting worse. I may ask a few follow-up questions to understand your message.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        render_onboarding_chips()
        
    for index, message in enumerate(chat["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if (
                message["role"] == "user"
                and index == len(chat["messages"]) - 1
                and chat.get("status") != "Analyzing symptoms"
            ):
                if st.button(
                    "✏️ Edit last message",
                    key=f"edit-last-{chat['id']}",
                    help="Re-open this message in the input box so you can correct it",
                ):
                    _edit_last_user_message(chat)

    render_result(chat, data)

    # ── Removable symptom chips ───────────────────────────────────────────────
    symptoms_now = set(chat.get("symptoms") or set())
    if symptoms_now and chat.get("status") != "Analyzing symptoms":
        st.markdown("**Tracked symptoms:**")
        cols = st.columns(min(len(symptoms_now), 4))
        for i, sym in enumerate(sorted(symptoms_now)):
            col = cols[i % len(cols)]
            label = display_symptom(sym)
            if col.button(
                f"✕ {label}",
                key=f"chip-remove-{chat['id']}-{sym}",
                help=f"Remove '{label}' from tracked symptoms",
            ):
                chat["symptoms"] = symptoms_now - {sym}
                st.rerun()

    # ── Quick-reply suggestion buttons ────────────────────────────────────────
    suggestions = [s for s in chat.get("last_suggested_symptoms") or [] if s]
    if suggestions and not chat.get("last_prediction"):
        st.caption("💬 Clarifying suggestions:")
        qr_cols = st.columns(min(len(suggestions), 3))
        for j, sym in enumerate(suggestions[:3]):
            lbl = display_symptom(sym)
            if qr_cols[j % len(qr_cols)].button(
                f"+ {lbl}",
                key=f"quickreply-{chat['id']}-{j}",
                help="Add this symptom to conversation",
            ):
                st.session_state.pending_prompt = lbl
                st.rerun()

    if chat.get("symptoms") and not chat.get("last_prediction"):
        st.caption("Add one more clear symptom, or describe more detail below.")

    prompt = st.chat_input(
        "Describe your symptoms — e.g. “fever for 3 days, dry cough, getting worse”"
    )
    
    if prompt is None:
        prompt = st.session_state.pop("pending_prompt", None)
    if prompt:
        if not GLOBAL_RATE_LIMITER.allow(st.session_state.session_id):
            st.warning("Please wait a moment before sending another message.")
            return
        for key, field in [
            (f"age-{chat['id']}", "age"),
            (f"severity-{chat['id']}", "severity"),
            (f"sex-{chat['id']}", "sex"),
            (f"duration-{chat['id']}", "duration"),
            (f"preg-{chat['id']}", "pregnancy"),
            (f"worsening-{chat['id']}", "worsening"),
        ]:
            if key in st.session_state:
                chat["context"][field] = st.session_state[key]
        context = chat["context"]
        prompt = scrub_pii(prompt.strip())
        extras = []
        if context.get("age") not in {None, "Not provided"}:
            extras.append(f"age group {context['age'].lower()}")
        if context.get("sex") not in {None, "Not provided"}:
            extras.append(f"sex {context['sex'].lower()}")
        if context.get("pregnancy") not in {None, "Not provided", "No"}:
            extras.append(f"pregnancy status {context['pregnancy'].lower()}")
        if context.get("duration"):
            extras.append(f"for {context['duration']} days")
        if context.get("severity") not in {None, "Not provided"}:
            extras.append(context["severity"].lower())
        if context.get("worsening"):
            extras.append("getting worse")
        enriched = f"{prompt} ({', '.join(extras)})" if extras else prompt
        chat["title"] = " ".join(prompt.split())[:42] or "New Chat"
        # Clear quick-reply chips and snapshot symptoms for rollback
        chat["last_suggested_symptoms"] = []
        # Snapshot pre-turn symptoms so the user can roll back via "edit last message".
        chat["symptoms_before_last"] = set(chat["symptoms"])
        chat["messages"].append({"role": "user", "content": enriched})
        run_with_processing_state(data, classical, tensorflow, extractor, chat)
        st.rerun()

    context_controls(chat)

    # ── Doctor's Visit Summary Export & Conversation Export ───────────────────
    if chat.get("messages"):
        with st.expander("📄 Export Visit Summary / Transcript", expanded=False):
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                summary_md = export_doctor_summary(chat)
                st.download_button(
                    label="🩺 Download Doctor's Visit Summary",
                    data=summary_md,
                    file_name=f"medguide_doctor_summary_{chat.get('id', 'visit')}.md",
                    mime="text/markdown",
                    help="Download a clinical visit prep checklist formatted with symptoms, timeline, and questions to ask your doctor.",
                    **_full_width(),
                )
            with col_exp2:
                transcript_md = export_markdown([chat])
                st.download_button(
                    label="📋 Download Full Transcript",
                    data=transcript_md,
                    file_name=f"medguide_transcript_{chat.get('id', 'chat')}.md",
                    mime="text/markdown",
                    help="Download raw conversation transcript in markdown.",
                    **_full_width(),
                )

    # ── Opt-in encrypted assessment history ───────────────────────────────────
    if chat.get("last_prediction") and chat.get("messages"):
        with st.expander("🔐 Save assessment & set follow-up", expanded=False):
            history_key = str(st.session_state.get("history_key_input", "") or "").strip()
            consent = bool(st.session_state.get("history_storage_consent", False))
            st.caption("Saving is optional. The report includes your symptom notes and model output, encrypted in the app's local database.")
            schedule_follow_up = st.checkbox("Set a follow-up reminder", value=False, key=f"history-schedule-{chat['id']}")
            follow_up_date = None
            if schedule_follow_up:
                follow_up_date = st.date_input(
                    "Follow-up date",
                    value=date.today() + timedelta(days=7),
                    key=f"history-due-{chat['id']}",
                )
            if not consent:
                st.info("Open **Private saved history** in the sidebar and confirm the storage notice before saving.")
            if st.button("Save encrypted assessment", key=f"save-assessment-{chat['id']}", disabled=(not consent or len(history_key) < MIN_HISTORY_KEY_LENGTH)):
                snapshot = {
                    "title": chat.get("title", "Saved assessment"),
                    "messages": chat.get("messages", []),
                    "symptoms": sorted(chat.get("symptoms") or []),
                    "context": chat.get("context") or {},
                    "last_prediction": chat.get("last_prediction"),
                    "triage_level": chat.get("triage_level"),
                    "triage_message": chat.get("triage_message"),
                    "follow_up_entries": [],
                }
                try:
                    record_id = save_assessment(history_key, snapshot, follow_up_date.isoformat() if follow_up_date else None)
                    st.success(f"Assessment saved. Reference: {record_id[:8]}. Open the **My history** tab to review it.")
                except Exception as exc:
                    st.error(f"Could not save this assessment: {exc}")

    # Raw chats are no longer written automatically to browser localStorage.
    # Health-history persistence is opt-in and uses encrypted records instead.

def _edit_last_user_message(chat: dict) -> None:
    """Roll back the last user turn and re-open its text in the input box."""
    while chat["messages"] and chat["messages"][-1]["role"] != "user":
        chat["messages"].pop()
    if not chat["messages"]:
        return
    last_user = chat["messages"].pop()
    chat["symptoms"] = set(chat.get("symptoms_before_last") or set())
    chat["symptoms_before_last"] = set(chat["symptoms"])
    chat["last_prediction"] = None
    chat["triage_level"] = None
    chat["triage_message"] = None
    chat["status"] = "Ready"
    chat["feedback_given"] = False
    # Strip the synthetic context suffix added at submit time so the user sees
    # exactly what they typed before the app enriched it.
    raw = str(last_user.get("content", ""))
    if " (" in raw and raw.endswith(")"):
        raw = raw[: raw.rfind(" (")]
    st.session_state.pending_prompt = raw
    st.rerun()

def render_saved_history() -> None:
    st.subheader("Saved assessments & follow-up tracker")
    st.caption("Only records saved with your private history key appear here. The key is not stored; if you lose it, encrypted records cannot be recovered.")
    history_key = str(st.session_state.get("history_key_input", "") or "").strip()
    if len(history_key) < MIN_HISTORY_KEY_LENGTH:
        st.info(f"Enter your private history key (at least {MIN_HISTORY_KEY_LENGTH} characters) in the sidebar to unlock your saved assessments.")
        return
    try:
        records = list_assessments(history_key)
    except Exception as exc:
        st.error(f"Could not read saved history: {exc}")
        return
    if not records:
        st.info("No saved assessments were found for this key. Saving is always optional.")
        return

    today = date.today()
    for record in records:
        payload = record["payload"]
        due = record.get("follow_up_date") or "Not scheduled"
        overdue = bool(record.get("follow_up_date") and record.get("follow_up_status") == "Pending" and record["follow_up_date"] < today.isoformat())
        title = str(payload.get("title", "Saved assessment"))[:80]
        status = record.get("follow_up_status", "Pending")
        heading = f"{title} · {record['created_at'][:10]} · follow-up: {status}"
        with st.expander(("⚠️ " if overdue else "") + heading, expanded=overdue):
            st.write(f"**Follow-up date:** {due}" + (" — overdue" if overdue else ""))
            prediction = payload.get("last_prediction") or {}
            st.write(f"**Pattern noted:** {prediction.get('disease', 'Not available')}")
            symptoms = payload.get("symptoms") or []
            st.write("**Tracked symptoms:** " + (", ".join(str(item).replace("_", " ") for item in symptoms) if symptoms else "None recorded"))
            entries = payload.get("follow_up_entries") or []
            if entries:
                st.markdown("**Follow-up timeline**")
                for entry in entries:
                    st.markdown(f"- **{html.escape(str(entry.get('date', '')))}** — {html.escape(str(entry.get('note', '')))}")
            report_chat = dict(payload)
            report_chat["symptoms"] = set(payload.get("symptoms") or [])
            st.download_button(
                "Download saved visit report",
                data=export_doctor_summary(report_chat),
                file_name=f"mediguide_saved_report_{record['record_id'][:8]}.md",
                mime="text/markdown",
                key=f"download-history-{record['record_id']}",
            )
            with st.form(f"follow-up-form-{record['record_id']}"):
                follow_note = st.text_area("Follow-up symptoms / progress", placeholder="e.g. Fever improved; cough continues. Avoid names or identifying details.", key=f"follow-note-{record['record_id']}")
                new_status = st.selectbox("Follow-up status", ["Pending", "Completed", "Cancelled"], index=["Pending", "Completed", "Cancelled"].index(status) if status in {"Pending", "Completed", "Cancelled"} else 0, key=f"follow-status-{record['record_id']}")
                submitted = st.form_submit_button("Save follow-up update")
            if submitted:
                if follow_note.strip():
                    entries.append({"date": date.today().isoformat(), "note": scrub_pii(follow_note.strip())[:1000]})
                    payload["follow_up_entries"] = entries
                    if not update_assessment_payload(history_key, record["record_id"], payload):
                        st.error("The follow-up note could not be saved.")
                        continue
                update_follow_up(history_key, record["record_id"], new_status)
                st.success("Follow-up tracker updated.")
                st.rerun()
            if st.button("Delete this saved assessment", key=f"delete-history-{record['record_id']}"):
                if delete_assessment(history_key, record["record_id"]):
                    st.success("Saved assessment deleted.")
                    st.rerun()
    st.divider()
    if st.checkbox("Confirm deletion of all saved assessments for this history key", key="confirm-delete-all-history"):
        if st.button("Delete all my saved assessments", type="secondary"):
            deleted = delete_all_assessments(history_key)
            st.success(f"Deleted {deleted} saved record(s).")
            st.rerun()


def render_appointments() -> None:
    """Private, encrypted appointment planner with in-app upcoming reminders."""
    st.markdown("## Appointments")
    st.caption("Plan a visit and see reminders while MediGuide is open. This is an in-app reminder, not an SMS or push notification service.")
    history_key = str(st.session_state.get("history_key_input", "") or "").strip()
    if len(history_key) < MIN_HISTORY_KEY_LENGTH:
        st.info(f"To keep appointments private across sessions, enter your private history key (at least {MIN_HISTORY_KEY_LENGTH} characters) in the sidebar under **Private saved history**.")
        return

    with st.expander("＋ Add an appointment", expanded=True):
        with st.form("new-appointment-form", clear_on_submit=True):
            purpose = st.text_input("Appointment purpose", placeholder="e.g. Follow-up visit", max_chars=100)
            provider = st.text_input("Clinic or provider (optional)", placeholder="e.g. Family clinic", max_chars=100)
            c1, c2 = st.columns(2)
            with c1:
                appt_date = st.date_input("Date", value=date.today() + timedelta(days=1), min_value=date.today())
            with c2:
                appt_time = st.time_input("Time", value=datetime.now().replace(minute=0, second=0, microsecond=0).time())
            notes = st.text_area("Private note (optional)", placeholder="Only include details you need for the visit.", max_chars=1000)
            submitted = st.form_submit_button("Save appointment", type="primary", **_full_width())
        if submitted:
            if not purpose.strip():
                st.error("Enter a short appointment purpose first.")
            else:
                try:
                    scheduled = datetime.combine(appt_date, appt_time).isoformat(timespec="minutes")
                    create_appointment(history_key, {
                        "purpose": scrub_pii(purpose.strip())[:100],
                        "provider": scrub_pii(provider.strip())[:100],
                        "notes": scrub_pii(notes.strip())[:1000],
                        "scheduled_at": scheduled,
                    })
                    st.success("Appointment saved securely.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save this appointment: {exc}")

    try:
        appointments = list_appointments(history_key)
    except Exception as exc:
        st.error(f"Could not load appointments: {exc}")
        return
    now = datetime.now()
    future = []
    for appt in appointments:
        try:
            scheduled = datetime.fromisoformat(appt["payload"].get("scheduled_at", ""))
        except (TypeError, ValueError):
            continue
        appt["_scheduled"] = scheduled
        if scheduled >= now:
            future.append(appt)
    future.sort(key=lambda item: item["_scheduled"])

    soon = [item for item in future if item["_scheduled"] <= now + timedelta(hours=24)]
    if soon:
        next_appt = soon[0]
        st.info(f"**Upcoming reminder:** {next_appt['payload'].get('purpose', 'Appointment')} — {next_appt['_scheduled'].strftime('%a, %d %b at %I:%M %p')}. Keep MediGuide open to see this reminder.")
    st.markdown("### Upcoming visits")
    if not future:
        st.markdown("<div class='welcome-card'><div class='welcome-title' style='font-size:1.3rem'>No upcoming appointments</div><p class='welcome-copy'>Your visits will appear here after you add one.</p></div>", unsafe_allow_html=True)
        return
    for item in future:
        payload = item["payload"]
        scheduled = item["_scheduled"]
        with st.container(border=True):
            left, right = st.columns([0.8, 0.2])
            with left:
                st.markdown(f"**{html.escape(str(payload.get('purpose', 'Appointment')))}**")
                st.caption(scheduled.strftime("%A, %d %B %Y · %I:%M %p"))
                if payload.get("provider"):
                    st.write(f"Clinic/provider: {html.escape(str(payload['provider']))}")
                if payload.get("notes"):
                    st.caption(str(payload["notes"]))
            with right:
                if st.button("Cancel", key=f"delete-appt-{item['appointment_id']}"):
                    delete_appointment(history_key, item["appointment_id"])
                    st.rerun()


def render_health_library(data) -> None:
    """Searchable, lightweight condition library; it never replaces the chat flow."""
    st.markdown("## Health library")
    st.caption("Browse the bundled condition catalogue and see where supporting information is available. Entries are educational and may not be clinically reviewed.")
    counts = data.targets.value_counts()
    query = st.text_input("Search conditions", placeholder="Type a condition name…", key="health-library-search")
    rows = []
    for disease in sorted(data.targets.unique(), key=lambda item: data.canonical_disease(item).lower()):
        canonical = data.canonical_disease(disease)
        if query and query.casefold() not in str(canonical).casefold():
            continue
        support = data.get_support(disease)
        rows.append({
            "Condition": canonical,
            "Training examples": int(counts[disease]),
            "Description": "Available" if support.description else "Not supplied",
            "References": "Available" if getattr(support, "evidence", None) else "Check details",
        })
    st.caption(f"Showing {len(rows):,} of {data.class_count:,} catalogue labels. Catalogue presence does not mean a label is eligible for automated prediction.")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=520, **_full_width())
    else:
        st.info("No conditions match that search. Try a shorter name or a different spelling.")


def render_secondary(data, classical, tensorflow, extractor) -> None:
    """Detailed app, data, and model information kept out of the Chat flow."""
    st.subheader("Technical details")
    st.caption("How MedGuide AI processes symptom descriptions. These details describe the software, not clinical validation.")
    # `tensorflow` is the actual optional TF predictor, not the ensemble.
    # When no TF artifact is present, show the classical model's real metadata.
    tf_available = tensorflow is not None and not getattr(tensorflow, "is_fallback", False)
    meta = (getattr(tensorflow, "metadata", {}) or {}) if tf_available else (getattr(classical, "metadata", {}) or {})
    using_fallback = not tf_available
    if tf_available:
        model_label = getattr(tensorflow, "model_name", "TensorFlow Neural Network")
    elif getattr(classical, "rf", None) is not None:
        model_label = "Expanded-corpus Random Forest fallback (TensorFlow artifact unavailable)"
    elif getattr(classical, "profile", None) is not None:
        model_label = "Support-filtered dataset-profile fallback (TensorFlow artifact unavailable)"
    else:
        model_label = "Knowledge-base fallback (TensorFlow artifact unavailable)"

    # ── Top-level KPI row ────────────────────────────────────────────────────
    a, b, c, d = st.columns(4)
    a.metric("App version", APP_VERSION)
    b.metric("Catalogue labels", data.class_count)
    c.metric("Symptom features", len(data.feature_names))
    d.metric("Training rows", f"{data.sample_count:,}")

    # ── Accuracy metrics row ─────────────────────────────────────────────────
    top1  = meta.get("test_accuracy")
    top3  = meta.get("top_3_accuracy")
    brier = meta.get("brier_score")
    roc   = meta.get("roc_auc_ovr_macro")
    ece   = meta.get("ece")
    fnr   = meta.get("fnr_macro")

    if any(v is not None for v in [top1, top3, brier, roc]):
        st.markdown("#### Model performance metrics")
        m1, m2, m3, m4 = st.columns(4)
        if top1 is not None:
            m1.metric("Top-1 accuracy", f"{top1:.1%}", help="Fraction of test samples where the top prediction matches the true condition.")
        if top3 is not None:
            m2.metric("Top-3 accuracy", f"{top3:.1%}", help="Fraction where the correct condition appears in the top 3 ranked predictions.")
        if roc is not None:
            m3.metric("ROC-AUC (macro OvR)", f"{roc:.3f}", help="One-vs-rest macro-averaged area under the ROC curve. Closer to 1.0 is better.")
        if brier is not None:
            m4.metric("Brier score", f"{brier:.4f}", help="Mean probability calibration error. Lower is better (0 = perfect).")

        if ece is not None or fnr is not None:
            n1, n2, _, _ = st.columns(4)
            if ece is not None:
                n1.metric("Calibration error (ECE)", f"{ece:.4f}", help="Expected calibration error; how aligned probabilities are to true frequencies.")
            if fnr is not None:
                n2.metric("False-negative rate", f"{fnr:.3f}", help="Macro-averaged rate at which the correct condition is missed by the top prediction.")

    with st.expander("Method overview", expanded=True):
        st.markdown(
            "MedGuide is an educational chatbot that extracts symptoms from free text and ranks "
            "possible patterns. It is not a diagnostic device."
        )
        st.markdown(
            f"- **Stack:** Hugging Face biomedical NLP (NER / BioClinicalBERT when available) → "
            f"TensorFlow artifact (if compatible) → Random Forest artifact (if compatible) → dataset profile → knowledge-base fallback\n"
            f"- **Active ranker:** {model_label}\n"
            f"- **Minimum symptoms:** {MIN_SYMPTOMS_FOR_PREDICTION}\n"
            f"- **Catalogue labels:** {data.class_count}\n"
            f"- **Labels meeting ML support threshold:** {getattr(getattr(classical, 'profile', None), 'metadata', {}).get('class_count', 'not available')}\n"
            f"- **Symptom vocabulary:** {len(data.feature_names)}"
        )

    with st.expander("How a message is processed",expanded=True):
        st.markdown(
            "1. The app checks the message for urgent phrases and extracts optional context such as duration, "
            "severity, worsening symptoms, and age group.\n"
            "2. Exact matches, common-language aliases, and conservative fuzzy matching map text only to the "
            f"{len(data.feature_names)} symptoms supported by the merged dataset.\n"
            "3. The optional biomedical NER and semantic-matching layers can suggest supported symptom features; "
            "they never select a condition directly.\n"
            "4. The prediction model receives a binary feature vector, ranks candidate conditions, and displays "
            "an educational result with safety guidance."
        )

    with st.expander("Model and training artifact",expanded=True):
        st.markdown(
            f"- **Active framework:** {meta.get('framework', 'scikit-learn / knowledge-base fallback' if using_fallback else 'TensorFlow/Keras')}\n"
            f"- **Model version:** {meta.get('model_version', getattr(tensorflow, 'model_version', APP_VERSION) if tf_available else APP_VERSION)}\n"
            f"- **Schema version:** {meta.get('schema_version', 'Not supplied')}\n"
            f"- **Feature vector:** {meta.get('feature_count', len(data.feature_names))} binary symptom indicators\n"
            f"- **Artifact classes:** {meta.get('class_count', data.class_count)} labels; low-support labels are withheld from specific ML/profile outputs"
        )

        # Performance line
        f1_macro    = meta.get("test_f1_macro",    getattr(tensorflow, "test_f1_macro", None))
        f1_weighted = meta.get("test_f1_weighted", getattr(tensorflow, "test_f1_weighted", None))
        test_acc    = meta.get("test_accuracy",    getattr(tensorflow, "test_accuracy", None))
        top3_acc    = meta.get("top_3_accuracy",   getattr(tensorflow, "top_3_accuracy", None))

        performance = []
        if test_acc    is not None: performance.append(f"Top-1 accuracy: **{test_acc:.1%}**")
        if top3_acc    is not None: performance.append(f"Top-3 accuracy: **{top3_acc:.1%}**")
        if f1_macro    is not None: performance.append(f"Macro F1: **{f1_macro:.3f}**")
        if f1_weighted is not None: performance.append(f"Weighted F1: **{f1_weighted:.3f}**")
        if performance:
            st.markdown(" · ".join(performance))

        split_counts = [
            ("Training samples", meta.get("train_samples")),
            ("Test samples",     meta.get("test_samples")),
        ]
        if any(v is not None for _, v in split_counts):
            st.markdown(" · ".join(f"{label}: **{value:,}**" for label, value in split_counts if value is not None))
        if meta.get("artifact_status"):
            st.caption(f"Artifact status: {meta['artifact_status']}")

    with st.expander("Fallback model",expanded=True):
        fallback_metrics = getattr(classical, "metrics", [])
        fallback_accuracy = getattr(classical, "test_accuracy", None)
        fallback_f1 = getattr(classical, "test_f1_weighted", None)
        st.markdown(
            f"- **Role:** used only if the TensorFlow predictor cannot return a valid result\n"
            f"- **Model:** {getattr(classical, 'model_name', 'Not available')}\n"
            f"- **Feature compatibility:** {len(getattr(getattr(classical, 'rf', None), 'feature_names', getattr(getattr(classical, 'profile', None), 'feature_names', [])))} features"
        )
        if fallback_accuracy is not None:
            st.markdown(f"Held-out accuracy: **{fallback_accuracy:.1%}**")
        if fallback_f1 is not None:
            st.markdown(f"Held-out weighted F1: **{fallback_f1:.3f}**")
        if fallback_metrics:
            st.dataframe(pd.DataFrame(fallback_metrics), hide_index=True, **_full_width())

    with st.expander("Data, privacy, and safety limits", expanded=True):
        st.markdown(
            f"- **Dataset shape:** {data.sample_count:,} labelled rows, {data.class_count} catalogue labels, "
            f"and {len(data.feature_names)} normalised symptom features.\n"
            f"- **ML/profile support rule:** labels with fewer than 5 training rows are withheld from specific ML/profile outputs; {getattr(getattr(classical, 'profile', None), 'metadata', {}).get('excluded_low_support_count', 'unknown')} labels are currently below that threshold.\n"
            "- **Input scope:** only symptoms in the bundled vocabulary are used for prediction; unsupported terms "
            "are not turned into new medical claims.\n"
            "- **Privacy:** unsaved chat state is held in the active application session; opt-in saved history is encrypted on the app host. Do not enter identifying or medical-record information.\n"
            "- **Safety:** urgent symptoms trigger guidance to seek immediate local medical care.\n"
            "- **Limitation:** this tool is not a diagnosis, prescription, or substitute for a licensed clinician."
        )

    with st.expander("Dataset audit & evaluation readiness", expanded=True):
        audit = audit_bundle(data)
        audit_cols = st.columns(4)
        audit_cols[0].metric("Unique symptom vectors", f"{audit['unique_symptom_patterns']:,}")
        audit_cols[1].metric("Duplicate vector groups", f"{audit['duplicate_pattern_groups']:,}")
        audit_cols[2].metric("Conflicting label groups", f"{audit['conflicting_pattern_groups']:,}")
        audit_cols[3].metric("Single-example conditions", f"{audit['conditions_with_one_example']:,}")
        st.dataframe(audit_frame(audit), hide_index=True, **_full_width())
        audit_report_path = Path(__file__).parent / "models" / "dataset_audit_report.json"
        if audit_report_path.exists():
            try:
                saved_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
                evaluation = saved_report.get("grouped_holdout_evaluation") or {}
                if evaluation:
                    st.markdown("#### Grouped held-out benchmark")
                    st.caption("Generated/imported profile benchmark only — not clinical diagnostic accuracy.")
                    eval_cols = st.columns(4)
                    eval_cols[0].metric("Top-1", f"{evaluation['top_1_accuracy']:.1%}")
                    eval_cols[1].metric("Top-3", f"{evaluation['top_3_accuracy']:.1%}")
                    eval_cols[2].metric("Macro F1", f"{evaluation['macro_f1']:.3f}")
                    eval_cols[3].metric("Test patterns", f"{evaluation['test_unique_patterns']:,}")
                    st.caption(f"{evaluation['tested_classes']} classes had enough unique non-conflicting patterns for this split. The other conditions are not covered by these metrics.")
            except (OSError, ValueError, KeyError, TypeError):
                st.caption("The saved audit report could not be read. Run the audit script to regenerate it.")
        st.warning(
            "The bundled corpus combines generated training rows and imported symptom profiles. "
            "A condition with only one example cannot be fairly evaluated on a holdout split. "
            "These metrics are dataset-integrity checks, not independent clinical validation."
        )
        st.markdown("Run `python scripts/audit_dataset.py --evaluate` to produce a grouped holdout benchmark. "
                    "The script excludes exact symptom vectors from both sides of the split to reduce leakage.")
        if audit_report_path.exists():
            st.download_button("Download latest dataset audit JSON", data=audit_report_path.read_text(encoding="utf-8"), file_name="medguide_dataset_audit.json", mime="application/json")

    with st.expander("Deployment & privacy readiness", expanded=False):
        checks = [
            ("Required training/support CSVs present", all((Path(__file__).parent / "models" / name).exists() for name in ["Training.csv", "diseases_and_symptoms.csv", "Symptom-severity.csv", "condition_evidence.csv"])),
            ("Merged condition count is 500+", data.class_count >= 500),
            ("Merged symptom feature count is 1,400+", len(data.feature_names) >= 1400),
            ("Patient-history storage is opt-in", True),
            ("Saved payloads use encryption", True),
            ("Raw chats are not auto-saved to browser storage", True),
        ]
        for label, passed in checks:
            st.markdown(("✅ " if passed else "❌ ") + label)
        st.caption("For public deployment, also configure HTTPS, access controls, backups, retention rules, and a privacy notice. A local encrypted database is still hosted wherever the app is deployed.")

    with st.expander(f"Condition catalogue ({data.class_count} labels)", expanded=True):
        counts = data.targets.value_counts()
        rows = []
        for disease in sorted(data.targets.unique(), key=lambda item: data.canonical_disease(item).lower()):
            support = data.get_support(disease)
            rows.append({
                "Condition": data.canonical_disease(disease),
                "Training records": int(counts[disease]),
                "Description": "Available" if support.description else "Not supplied",
                "Medication references": "Available" if support.medications else "Not supplied",
                "Precautions / diet": "Available" if support.precautions or support.diet else "Not supplied",
            })
        st.caption("This is a reference catalogue, not a list of conditions the model can reliably predict. ML/profile predictions require at least five training rows per label; the curated knowledge base is a separate educational fallback. Condition-specific guidance appears only where supplied.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=520, **_full_width())

    with st.expander("Artifact metadata (advanced)", expanded=False):
        if meta:
            st.json({key: value for key, value in meta.items() if key not in {"feature_names", "class_names"}})
        else:
            st.info("The active fallback model does not expose artifact metadata.")


def main() -> None:
    initialize_session()
    render_styles()
    try:
        data, classical, tensorflow, extractor = get_runtime()
    except FileNotFoundError:
        st.error("The app is missing required dataset files. Restore the bundled models/data directory and reload.")
        return
    except Exception as exc:
        st.error("The app could not load a compatible model artifact.")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    render_sidebar(data, tensorflow)
    render_brand()

    chat_tab, appointments_tab, history_tab, library_tab, more_tab = st.tabs([
        "💬 Chat", "📅 Appointments", "🗂️ My health", "📚 Health library", "⚙️ Method & data"
    ])
    with chat_tab:
        render_chat(data, classical, tensorflow, extractor)
    with appointments_tab:
        render_appointments()
    with history_tab:
        render_saved_history()
    with library_tab:
        render_health_library(data)
    with more_tab:
        render_secondary(data, classical, tensorflow, extractor)

    st.markdown(f'<div class="footer">{html.escape(SHORT_DISCLAIMER)} · MediGuide offers educational information, not a medical diagnosis.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
