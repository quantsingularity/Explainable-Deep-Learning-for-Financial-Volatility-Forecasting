"""
Explainability package: SHAP values and attention weight analysis.
"""

from .explain import (
    aggregate_shap_across_time,
    compute_shap_values,
    create_interpretability_report,
    plot_attention_heatmap,
    plot_shap_bar,
    plot_shap_summary,
    prepare_shap_background,
    run_full_explainability_pipeline,
)

__all__ = [
    "prepare_shap_background",
    "compute_shap_values",
    "aggregate_shap_across_time",
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_attention_heatmap",
    "create_interpretability_report",
    "run_full_explainability_pipeline",
]
