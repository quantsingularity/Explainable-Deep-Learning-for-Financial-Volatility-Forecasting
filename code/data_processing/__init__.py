"""
Data processing package: synthetic data generation and real data download.
"""

from .data_generator import generate_synthetic_dataset, save_dataset
from .download_real_data import build_complete_dataset, download_multiple_assets

__all__ = [
    "generate_synthetic_dataset",
    "save_dataset",
    "build_complete_dataset",
    "download_multiple_assets",
]
