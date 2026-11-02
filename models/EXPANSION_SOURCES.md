# Dataset expansion and evidence policy

## Public dataset option for the academic benchmark

**Disease and symptoms dataset 2023** — Mendeley Data, DOI `10.17632/2cxccsxydc.1`, https://data.mendeley.com/datasets/2cxccsxydc/1. The dataset description reports 773 unique diseases, 377 symptoms, and about 246,000 rows. It states the dataset was artificially generated. Its page lists CC BY 4.0. Keep attribution and the original dataset card in the project submission.

This dataset is suitable only as a generated-data benchmark. Do not present its labels or its held-out scores as clinically verified.

## Medical terminology reference

CDC/NCHS ICD-10-CM files provide official standardized diagnosis-code descriptions: https://www.cdc.gov/nchs/icd/icd-10-cm/files.html. ICD-10-CM is a coding/classification system; it does not by itself provide a symptom-to-disease model or validate a prediction.

## Evidence policy

Every added condition must retain its source, license, label count, symptom vocabulary, and review status. Descriptions inferred from a dataset must be labeled as data-derived. Do not add fabricated symptom profiles to reach a target count. Do not treat model scores as patient-level disease probabilities without external calibration and validation.
