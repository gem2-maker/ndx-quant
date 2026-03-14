"""Strategy registry."""

from strategies.momentum import MomentumStrategy, PriceMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.rsi_macd import RsiMacdStrategy

STRATEGIES = {
    "momentum": MomentumStrategy,
    "price_momentum": PriceMomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "rsi_macd": RsiMacdStrategy,
}


def get_strategy(name: str, **kwargs):
    """Get a strategy instance by name."""
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return STRATEGIES[name](**kwargs)
