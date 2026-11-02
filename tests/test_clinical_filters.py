"""Unit tests for the clinical ranking layer."""

from src.clinical_filters import (
    NONSPECIFIC_LABEL,
    CHILD_EXCLUDED,
    apply_clinical_ranking,
    should_show_medications,
    should_show_lifestyle,
    PREVALENCE_PRIOR,
)


NAMES = list(PREVALENCE_PRIOR.keys())


def _uniform_with_bumps(bumps: dict[str, float]):
    base = 0.02
    probs = [bumps.get(n, base) for n in NAMES]
    return probs


def test_child_demographics_do_not_zero_out_conditions():
    probs = _uniform_with_bumps({"AIDS": 0.25, "Alcoholic hepatitis": 0.2, "Common Cold": 0.1})
    result = apply_clinical_ranking(
        NAMES, probs, age_group="Child", symptoms={"fatigue", "headache", "nausea"}, top_k=5
    )
    assert result.disease == NONSPECIFIC_LABEL or result.disease not in CHILD_EXCLUDED
    ranked = [n for n, _ in result.alternatives]
    # Pregnancy status does not apply unsupported prevalence or exclusion rules.
    assert "AIDS" in ranked or result.forced_nonspecific
    assert "Alcoholic hepatitis" in ranked or result.forced_nonspecific


def test_nonspecific_bag_forces_catch_all():
    probs = _uniform_with_bumps({"AIDS": 0.2, "Dengue": 0.18, "Common Cold": 0.1})
    result = apply_clinical_ranking(
        NAMES, probs, age_group="Adult", symptoms={"fatigue", "headache", "nausea"}, top_k=5
    )
    assert result.forced_nonspecific is True
    assert result.disease == NONSPECIFIC_LABEL


def test_distinctive_symptoms_can_rank_concrete_disease():
    # Give Common Cold a clear lead in raw probs + prior-friendly symptoms
    probs = _uniform_with_bumps({"Common Cold": 0.55, "Allergy": 0.1})
    result = apply_clinical_ranking(
        NAMES,
        probs,
        age_group="Adult",
        symptoms={"continuous sneezing", "runny nose", "throat irritation", "cough"},
        top_k=3,
    )
    assert result.forced_nonspecific is False
    assert result.disease == "Common Cold"
    assert result.confidence >= 0.38


def test_pregnancy_does_not_suppress_serious_conditions():
    probs = _uniform_with_bumps({"AIDS": 0.3, "Alcoholic hepatitis": 0.25, "Common Cold": 0.15})
    result = apply_clinical_ranking(
        NAMES,
        probs,
        age_group="Adult",
        pregnancy="Yes",
        symptoms={"fatigue", "nausea", "vomiting", "headache"},
        top_k=5,
    )
    ranked = [n for n, _ in result.alternatives]
    # Pregnancy status does not apply unsupported prevalence or exclusion rules.
    assert "AIDS" in ranked or result.forced_nonspecific
    assert "Alcoholic hepatitis" in ranked or result.forced_nonspecific


def test_meds_suppressed_for_child():
    show, warn = should_show_medications("Common Cold", "Child")
    assert show is False
    assert warn is not None


def test_lifestyle_only_for_benign():
    assert should_show_lifestyle("Common Cold") is True
    assert should_show_lifestyle("Heart attack") is False
    assert should_show_lifestyle(NONSPECIFIC_LABEL) is True


def test_sex_incompatible_labels_are_downweighted_not_erased():
    test_names = ["Ectopic Pregnancy", "Prostate Cancer", "Common Cold"]
    probs = [0.45, 0.45, 0.10]
    # Anatomical context may affect interpretation, but must not act as a rule-out.
    male_result = apply_clinical_ranking(test_names, probs, sex="Male", top_k=3)
    male_scores = dict(male_result.alternatives)
    assert "Ectopic Pregnancy" in male_scores

    female_result = apply_clinical_ranking(test_names, probs, sex="Female", top_k=3)
    female_scores = dict(female_result.alternatives)
    assert "Prostate Cancer" in female_scores


def test_child_context_does_not_erase_expanded_conditions():
    test_names = ["Benign Prostatic Hyperplasia", "Common cold"]
    probs = [0.8, 0.2]
    result = apply_clinical_ranking(test_names, probs, age_group="Child", top_k=2)
    ranked = [n for n, _ in result.alternatives]
    assert "Benign Prostatic Hyperplasia" in ranked

