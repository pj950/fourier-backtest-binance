"""Parameter space definitions and utilities."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpace:
    """Define a parameter search space."""

    name: str
    param_type: str  # "int", "float", "categorical"
    min_val: float | None = None
    max_val: float | None = None
    categories: list[Any] | None = None
    log_scale: bool = False

    def __post_init__(self):
        """Validate parameter space."""
        if self.param_type in ["int", "float"]:
            if self.min_val is None or self.max_val is None:
                raise ValueError(f"min_val and max_val required for {self.param_type} type")
            if self.min_val >= self.max_val:
                raise ValueError(f"min_val must be less than max_val")
        elif self.param_type == "categorical":
            if not self.categories or len(self.categories) == 0:
                raise ValueError("categories required for categorical type")
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


@dataclass
class StrategyParams:
    """Strategy parameters for backtesting."""

    # Fourier/smoothing params
    min_trend_period_hours: float = 48.0
    cutoff_scale: float = 1.0
    
    # Stop loss params
    atr_period: int = 14
    k_stop: float = 2.0
    k_profit: float = 3.0
    
    # Signal params
    slope_threshold: float = 0.0
    slope_lookback: int = 1
    min_volatility: float = 0.0
    
    # Backtest config
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    
    # Additional params
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_trend_period_hours": self.min_trend_period_hours,
            "cutoff_scale": self.cutoff_scale,
            "atr_period": self.atr_period,
            "k_stop": self.k_stop,
            "k_profit": self.k_profit,
            "slope_threshold": self.slope_threshold,
            "slope_lookback": self.slope_lookback,
            "min_volatility": self.min_volatility,
            "initial_capital": self.initial_capital,
            "fee_rate": self.fee_rate,
            "slippage": self.slippage,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyParams":
        """Create from dictionary."""
        known_fields = {
            "min_trend_period_hours",
            "cutoff_scale",
            "atr_period",
            "k_stop",
            "k_profit",
            "slope_threshold",
            "slope_lookback",
            "min_volatility",
            "initial_capital",
            "fee_rate",
            "slippage",
        }
        
        main_params = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        
        return cls(**main_params, extra=extra)


def create_default_param_space() -> dict[str, ParamSpace]:
    """Create default parameter search space."""
    return {
        "min_trend_period_hours": ParamSpace(
            name="min_trend_period_hours",
            param_type="float",
            min_val=12.0,
            max_val=168.0,  # 1 week
            log_scale=False,
        ),
        "cutoff_scale": ParamSpace(
            name="cutoff_scale",
            param_type="float",
            min_val=0.5,
            max_val=2.0,
            log_scale=False,
        ),
        "atr_period": ParamSpace(
            name="atr_period",
            param_type="int",
            min_val=7,
            max_val=28,
            log_scale=False,
        ),
        "k_stop": ParamSpace(
            name="k_stop",
            param_type="float",
            min_val=1.0,
            max_val=4.0,
            log_scale=False,
        ),
        "k_profit": ParamSpace(
            name="k_profit",
            param_type="float",
            min_val=2.0,
            max_val=6.0,
            log_scale=False,
        ),
        "slope_threshold": ParamSpace(
            name="slope_threshold",
            param_type="float",
            min_val=0.0,
            max_val=0.001,
            log_scale=False,
        ),
        "slope_lookback": ParamSpace(
            name="slope_lookback",
            param_type="int",
            min_val=1,
            max_val=5,
            log_scale=False,
        ),
    }
