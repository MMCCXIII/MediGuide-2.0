# 647-Label Catalog Coverage Audit

Generated from the current merged data pipeline.

- Catalog labels: **647**
- Total merged training rows: **11,977**
- Labels with exactly one training record: **533**
- Labels with two or fewer training records: **533**
- Labels with 100 training records: **70**
- Labels with 101 training records: **44**

## Current content coverage

| Field | Labels populated | Labels missing |
|---|---:|---:|
| Description | 114 | 533 |
| Precautions | 114 | 533 |
| Diet | 7 | 640 |
| Medication reference | 114 | 533 |

The full row-level export is in `models/catalog_coverage_audit.csv`. A single merged symptom profile is not equivalent to a clinically validated training set. Additional training rows should be sourced or reviewed rather than duplicated or generated as if they were observed cases. Medication and diet fields should remain educational and condition-specific; for many labels the correct value may be “no general diet recommendation” or “treatment must be individualized by a clinician.”
