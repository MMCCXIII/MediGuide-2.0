from pathlib import Path

from src.config import MODELS_DIR
from src.data_pipeline import load_data_bundle, locate_data_dir, normalize_key, parse_list_cell


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_normalize_key_removes_dataset_spacing_noise():
    assert normalize_key(" dischromic _patches ") == "dischromic patches"


def test_parse_list_cell_preserves_dataset_items():
    assert parse_list_cell("['One', 'Two']") == ["One", "Two"]


def test_training_data_is_a_clean_binary_matrix():
    bundle = load_data_bundle(MODELS_DIR)
    assert bundle.features.shape[0] == bundle.targets.shape[0]
    assert bundle.features.shape[1] == len(bundle.feature_names)
    assert set(bundle.features.stack().unique()).issubset({0, 1})
    assert bundle.class_count >= 600
    assert len(bundle.feature_names) >= 1400


def test_models_directory_is_the_default_data_location():
    assert locate_data_dir() == MODELS_DIR
