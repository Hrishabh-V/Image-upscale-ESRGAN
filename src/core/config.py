import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

""" Load .env"""
load_dotenv()

""" validate tensor """
VALIDATE_TENSOR = os.getenv(
    "VALIDATE_TENSOR",
    "INFO"
).upper()

MODEL_PATH = (
    PROJECT_ROOT /
    os.getenv(
        "MODEL_PATH",
        "src/models/RRDB_ESRGAN_x4.pth"
    )
)