from pathlib import Path

ROOT            = Path(__file__).resolve().parents[1]
DATA_RAW        = ROOT / "data" / "raw"          # <- expect CSVs + images here
DATA_PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models"

IMG_SIZE        = 224        # square resize for simplicity
RANDOM_STATE    = 42
