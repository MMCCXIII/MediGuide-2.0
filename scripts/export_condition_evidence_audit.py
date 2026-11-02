"""Export an auditable register of curated symptom profiles and evidence-review status."""
from __future__ import annotations
import csv, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.knowledge_base import CONDITIONS

# Only direct, relevant references are attached. Other rows remain explicitly pending review.
SOURCES = {
    "Common cold": "https://www.cdc.gov/common-cold/about/index.html",
    "Acute bronchitis": "https://www.cdc.gov/acute-bronchitis/about/index.html",
    "Diabetes": "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes",
}

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

rows=[]
for c in CONDITIONS:
    rows.append({
        "condition_id": slug(c.name),
        "condition": c.name,
        "category": c.category,
        "description": c.description,
        "symptom_profile_count": len(c.symptoms),
        "symptom_profile": "; ".join(f"{s}={w:.2f}" for s,w in c.symptoms.items()),
        "precautions": " | ".join(c.precautions),
        "educational_medication_notes": " | ".join(c.medications),
        "reference_url": SOURCES.get(c.name, ""),
        "evidence_review_status": "reference linked; verify exact profile" if c.name in SOURCES else "needs condition-specific medical-source review",
        "clinical_validation_status": "not clinically validated",
    })
with (ROOT / "models" / "condition_evidence.csv").open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} condition audit rows to models/condition_evidence.csv")
