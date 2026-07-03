"""
Explainability package: SHAP values and attention weight analysis.

Re-exports are LAZY (PEP 562) so that importing unrelated packages never
pulls in this module's plotting dependencies.
"""

_EXPORTS = {
    "prepare_shap_background": "explain",
    "compute_shap_values": "explain",
    "aggregate_shap_across_time": "explain",
    "plot_shap_summary": "explain",
    "plot_shap_bar": "explain",
    "plot_attention_heatmap": "explain",
    "create_interpretability_report": "explain",
    "run_full_explainability_pipeline": "explain",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
