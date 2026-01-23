"""
Trading Strategy Backtester with Volatility-Based Strategies
Implements realistic trading with transaction costs, slippage, and position limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""

    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 5.0  # 5 basis points
    slippage_bps: float = 2.0  # 2 basis points
    max_position_size: float = 0.2  # 20% of capital
    min_position_size: float = 0.01  # 1% of capital
    rebalance_frequency: int = 1  # Daily rebalancing
    risk_free_rate: float = 0.02  # 2% annual


class VolatilityTradingStrategy:
    """
    Base class for volatility-based trading strategies
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions = []
        self.trades = []
        self.portfolio_values = []

    def generate_signals(
        self, volatility_forecasts: np.ndarray, realized_volatility: np.ndarray
    ) -> np.ndarray:
        """
        Generate trading signals based on volatility forecasts.
        Must be implemented by subclasses.

        Returns:
        --------
        signals : np.ndarray
            Trading signals (-1: short, 0: neutral, 1: long)
        """
        raise NotImplementedError

    def calculate_position_size(
        self, signal: float, capital: float, predicted_vol: float
    ) -> float:
        """
        Calculate position size based on signal and risk (Kelly criterion inspired)
        """
        if signal == 0:
            return 0.0

        # Base position size
        base_size = capital * self.config.max_position_size

        # Adjust for volatility (inverse volatility weighting)
        vol_adjustment = 1.0 / (1.0 + predicted_vol * 50)  # Scale factor

        position_size = base_size * abs(signal) * vol_adjustment

        # Enforce limits
        position_size = np.clip(
            position_size,
            self.config.min_position_size * capital,
            self.config.max_position_size * capital,
        )

        return position_size if signal > 0 else -position_size

    def apply_transaction_costs(self, trade_value: float) -> float:
        """Apply transaction costs and slippage"""
        total_cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        return trade_value * (total_cost_bps / 10000)


class VolatilityArbitrageStrategy(VolatilityTradingStrategy):
    """
    Volatility Arbitrage Strategy
    - Long when predicted vol > realized vol (expect volatility increase)
    - Short when predicted vol < realized vol (expect volatility decrease)
    """

    def __init__(self, config: BacktestConfig, threshold: float = 0.15):
        super().__init__(config)
        self.threshold = threshold  # 15% difference threshold

    def generate_signals(
        self, volatility_forecasts: np.ndarray, realized_volatility: np.ndarray
    ) -> np.ndarray:
        """
        Generate signals based on forecast vs realized volatility spread
        """
        # Calculate spread (percentage difference)
        spread = (volatility_forecasts - realized_volatility) / realized_volatility

        signals = np.zeros_like(spread)

        # Long when forecast significantly higher than realized
        signals[spread > self.threshold] = 1.0

        # Short when forecast significantly lower than realized
        signals[spread < -self.threshold] = -1.0

        # Gradual signal strength based on spread magnitude
        signals = signals * np.clip(np.abs(spread) / self.threshold, 0, 1)

        return signals


class TrendFollowingVolStrategy(VolatilityTradingStrategy):
    """
    Trend Following on Volatility
    - Long when volatility is trending up
    - Short when volatility is trending down
    """

    def __init__(self, config: BacktestConfig, lookback: int = 10):
        super().__init__(config)
        self.lookback = lookback

    def generate_signals(
        self, volatility_forecasts: np.ndarray, realized_volatility: np.ndarray
    ) -> np.ndarray:
        """
        Generate signals based on volatility trend
        """
        signals = np.zeros_like(volatility_forecasts)

        for i in range(self.lookback, len(volatility_forecasts)):
            # Calculate trend (linear regression slope)
            window = realized_volatility[i - self.lookback : i]
            x = np.arange(len(window))
            slope, _ = np.polyfit(x, window, 1)

            # Normalize slope
            avg_vol = np.mean(window)
            normalized_slope = slope / avg_vol if avg_vol > 0 else 0

            # Signal based on trend strength
            signals[i] = np.clip(normalized_slope * 50, -1, 1)

        return signals


class MeanReversionVolStrategy(VolatilityTradingStrategy):
    """
    Mean Reversion Strategy
    - Long when volatility is below historical average
    - Short when volatility is above historical average
    """

    def __init__(
        self, config: BacktestConfig, lookback: int = 60, z_threshold: float = 1.5
    ):
        super().__init__(config)
        self.lookback = lookback
        self.z_threshold = z_threshold

    def generate_signals(
        self, volatility_forecasts: np.ndarray, realized_volatility: np.ndarray
    ) -> np.ndarray:
        """
        Generate signals based on mean reversion
        """
        signals = np.zeros_like(volatility_forecasts)

        for i in range(self.lookback, len(volatility_forecasts)):
            window = realized_volatility[i - self.lookback : i]
            mean_vol = np.mean(window)
            std_vol = np.std(window)

            if std_vol > 0:
                # Z-score
                z_score = (volatility_forecasts[i] - mean_vol) / std_vol

                # Mean reversion signal (inverse of z-score)
                if z_score > self.z_threshold:
                    signals[i] = -0.5  # Short (expect reversion down)
                elif z_score < -self.z_threshold:
                    signals[i] = 0.5  # Long (expect reversion up)

        return signals


class BacktestEngine:
    """
    Backtesting engine with realistic trading constraints
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_backtest(
        self,
        strategy: VolatilityTradingStrategy,
        returns: np.ndarray,
        volatility_forecasts: np.ndarray,
        realized_volatility: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> Dict:
        """
        Run backtest for a given strategy

        Parameters:
        -----------
        strategy : VolatilityTradingStrategy
            Trading strategy to backtest
        returns : np.ndarray
            Asset returns
        volatility_forecasts : np.ndarray
            Model volatility predictions
        realized_volatility : np.ndarray
            Realized volatility
        dates : pd.DatetimeIndex
            Trading dates

        Returns:
        --------
        results : dict
            Backtest results and performance metrics
        """

        # Generate trading signals
        signals = strategy.generate_signals(volatility_forecasts, realized_volatility)

        # Initialize portfolio
        capital = self.config.initial_capital
        position = 0.0
        portfolio_values = [capital]
        positions = [0.0]
        trades = []
        costs = []

        # Simulate trading
        for i in range(1, len(returns)):
            # Current signal
            signal = signals[i]
            predicted_vol = volatility_forecasts[i]

            # Calculate target position
            target_position = strategy.calculate_position_size(
                signal, capital, predicted_vol
            )

            # Trade execution
            trade_size = target_position - position

            if abs(trade_size) > self.config.min_position_size * capital:
                # Execute trade with costs
                trade_cost = strategy.apply_transaction_costs(abs(trade_size))
                capital -= trade_cost
                costs.append(trade_cost)

                trades.append(
                    {
                        "date": dates[i],
                        "signal": signal,
                        "trade_size": trade_size,
                        "cost": trade_cost,
                        "position_before": position,
                        "position_after": target_position,
                    }
                )

                position = target_position

            # Update portfolio value based on position and returns
            pnl = position * returns[i]
            capital += pnl

            portfolio_values.append(capital)
            positions.append(position)

        # Calculate performance metrics
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        metrics = self._calculate_performance_metrics(
            portfolio_returns, portfolio_values, trades, costs
        )

        return {
            "metrics": metrics,
            "portfolio_values": portfolio_values,
            "positions": positions,
            "trades": trades,
            "signals": signals,
            "dates": dates,
        }

    def _calculate_performance_metrics(
        self,
        returns: np.ndarray,
        portfolio_values: List[float],
        trades: List[Dict],
        costs: List[float],
    ) -> Dict:
        """Calculate comprehensive performance metrics"""

        # Basic metrics
        total_return = (
            portfolio_values[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Annualized metrics (assuming 252 trading days)
        n_days = len(returns)
        n_years = n_days / 252

        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        annualized_vol = np.std(returns) * np.sqrt(252)

        # Sharpe ratio
        excess_returns = returns - (self.config.risk_free_rate / 252)
        sharpe_ratio = (
            np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0
        )

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (
            np.mean(excess_returns) / downside_std * np.sqrt(252)
            if downside_std > 0
            else 0
        )

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        winning_trades = sum(
            1 for t in trades if t["trade_size"] * returns[trades.index(t)] > 0
        )
        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0

        # Total costs
        total_costs = sum(costs)
        cost_ratio = total_costs / self.config.initial_capital

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "n_trades": len(trades),
            "total_costs": total_costs,
            "cost_ratio": cost_ratio,
        }


def compare_strategies(results_dict: Dict[str, Dict], save_path: str = "../figures"):
    """
    Compare multiple trading strategies

    Parameters:
    -----------
    results_dict : dict
        Dictionary of strategy results {strategy_name: results}
    save_path : str
        Path to save comparison plots
    """

    # Create comparison dataframe
    comparison_data = []

    for strategy_name, results in results_dict.items():
        metrics = results["metrics"]
        comparison_data.append(
            {
                "Strategy": strategy_name,
                "Total Return (%)": metrics["total_return"] * 100,
                "Annual Return (%)": metrics["annualized_return"] * 100,
                "Annual Vol (%)": metrics["annualized_volatility"] * 100,
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Sortino Ratio": metrics["sortino_ratio"],
                "Max Drawdown (%)": metrics["max_drawdown"] * 100,
                "Calmar Ratio": metrics["calmar_ratio"],
                "Win Rate (%)": metrics["win_rate"] * 100,
                "N Trades": metrics["n_trades"],
                "Cost Ratio (%)": metrics["cost_ratio"] * 100,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    print("\n" + "=" * 100)
    print("TRADING STRATEGY COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))

    # Save to CSV
    os.makedirs(save_path.replace("/figures", "/tables"), exist_ok=True)
    comparison_df.to_csv(
        save_path.replace("/figures", "/tables") + "/strategy_comparison.csv",
        index=False,
    )

    # Plot comparisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Portfolio values over time
    ax = axes[0, 0]
    for strategy_name, results in results_dict.items():
        dates = results["dates"]
        values = results["portfolio_values"]
        ax.plot(dates, values, label=strategy_name, linewidth=2)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sharpe ratios
    ax = axes[0, 1]
    strategies = list(results_dict.keys())
    sharpe_ratios = [results_dict[s]["metrics"]["sharpe_ratio"] for s in strategies]
    ax.bar(strategies, sharpe_ratios, color="steelblue", alpha=0.7)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Max drawdown
    ax = axes[1, 0]
    max_drawdowns = [
        results_dict[s]["metrics"]["max_drawdown"] * 100 for s in strategies
    ]
    ax.bar(strategies, max_drawdowns, color="coral", alpha=0.7)
    ax.set_ylabel("Max Drawdown (%)", fontsize=12)
    ax.set_title("Maximum Drawdown Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Win rate vs # trades
    ax = axes[1, 1]
    win_rates = [results_dict[s]["metrics"]["win_rate"] * 100 for s in strategies]
    n_trades = [results_dict[s]["metrics"]["n_trades"] for s in strategies]
    scatter = ax.scatter(
        n_trades, win_rates, s=200, alpha=0.6, c=sharpe_ratios, cmap="viridis"
    )

    for i, strategy in enumerate(strategies):
        ax.annotate(
            strategy,
            (n_trades[i], win_rates[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Number of Trades", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Win Rate vs Trading Frequency", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", fontsize=10)

    plt.tight_layout()

    # Save figure
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/strategy_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}/strategy_comparison.png")

    plt.close()

    return comparison_df


if __name__ == "__main__":
    print("Trading strategy backtester loaded.")
    print("Use this module to backtest volatility-based trading strategies.")
