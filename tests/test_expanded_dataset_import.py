from src.data_pipeline import load_data_bundle, normalize_key


def test_bundled_dataset_adds_500_plus_conditions_and_1400_plus_symptoms():
    bundle = load_data_bundle()
    assert bundle.class_count >= 500
    assert len(bundle.feature_names) >= 1400
    assert bundle.sample_count >= 11_900


def test_overlapping_disease_labels_are_merged_not_duplicated():
    bundle = load_data_bundle()
    normalized = [normalize_key(label) for label in bundle.targets.unique()]
    assert len(normalized) == len(set(normalized))
    assert sum(value == "acute kidney injury" for value in normalized) == 1


def test_overlapping_symptom_names_share_a_single_feature():
    bundle = load_data_bundle()
    assert len(bundle.feature_names) == len(set(bundle.feature_names))
    assert "abdominal pain" in bundle.feature_names
    assert "visible bulge or swelling" in bundle.feature_names
