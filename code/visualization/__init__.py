"""
Visualization package: research paper figures and architecture diagrams.
"""

from .generate_paper_figures import generate_all_figures
from .visualize_architecture import create_architecture_diagram

__all__ = ["generate_all_figures", "create_architecture_diagram"]
