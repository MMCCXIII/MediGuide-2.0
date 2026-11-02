from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import load_data_bundle
from src.quality import dataset_quality

if __name__ == "__main__":
    data = load_data_bundle()
    print(dataset_quality(data))
