"""
Evaluation package: metrics, ablation study, baselines, and trading backtest.
"""

from .ablation_study import (
    AblationStudy,
    LSTMAttentionModel,
    LSTMAttentionSHAPModel,
    LSTMOnlyModel,
)
from .baseline_models import (
    EGARCHModel,
    GARCHModel,
    HARRVModel,
    HistoricalVolatility,
    compare_baseline_models,
)
from .eval import (
    compare_with_baselines,
    evaluate_var_backtest,
    evaluate_volatility_forecast,
    generate_results_table,
    plot_var_backtest,
)
from .trading_backtest import (
    BacktestConfig,
    BacktestEngine,
    MeanReversionVolStrategy,
    TrendFollowingVolStrategy,
    VolatilityArbitrageStrategy,
    compare_strategies,
)

__all__ = [
    "LSTMOnlyModel",
    "LSTMAttentionModel",
    "LSTMAttentionSHAPModel",
    "AblationStudy",
    "GARCHModel",
    "EGARCHModel",
    "HARRVModel",
    "HistoricalVolatility",
    "compare_baseline_models",
    "evaluate_volatility_forecast",
    "evaluate_var_backtest",
    "compare_with_baselines",
    "plot_var_backtest",
    "generate_results_table",
    "BacktestConfig",
    "VolatilityArbitrageStrategy",
    "TrendFollowingVolStrategy",
    "MeanReversionVolStrategy",
    "BacktestEngine",
    "compare_strategies",
]
