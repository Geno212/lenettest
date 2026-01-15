"""Scheduler configurations and defaults.

Follow the same style as optimizers.py: each scheduler has
description, required/optional parameters, defaults, param_ranges,
use_cases and notes.
"""

from typing import Dict, Any, Optional


SCHEDULER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "STEPLR": {
        "description": "Decays the learning rate of each parameter group by gamma every step_size epochs.",
        "required": [],
        "optional": ["step_size", "gamma"],
        "defaults": {
            "step_size": 30,
            "gamma": 0.1
        },
        "param_ranges": {
            "step_size": (1, 100),
            "gamma": (0.0, 1.0)
        },
        "use_cases": [
            "Simple constant drops at fixed intervals",
            "Works well for many classical experiments"
        ],
        "notes": "Good default for long training runs; simple to tune."
    },

    "MULTISTEPLR": {
        "description": "Decays the learning rate by gamma once the number of epoch reaches one of the milestones.",
        "optional": ["gamma"],
        "defaults": {
            "gamma": 0.1
        },
        "param_ranges": {
            "gamma": (0.0, 1.0)
        },
        "use_cases": [
            "When you want manual control for LR drops",
            "Common in classification training recipes"
        ],
        "notes": "Specify milestones as epoch indices."
    },

    "EXPONENTIALLR": {
        "description": "Decays the learning rate of each parameter group by gamma every epoch.",
        "required": [],
        "optional": ["gamma"],
        "defaults": {
            "gamma": 0.95
        },
        "param_ranges": {
            "gamma": (0.0, 1.0)
        },
        "use_cases": [
            "Smooth exponential decay",
            "When you prefer continuous decay instead of steps"
        ],
        "notes": "Small gamma values lead to quick decay."
    },

    "COSINEANNEALINGLR": {
        "description": "Set the learning rate of each parameter group using a cosine annealing schedule.",
        "required": ["T_max"],
        "optional": ["eta_min"],
        "defaults": {
            "T_max": 50,
            "eta_min": 0
        },
        "param_ranges": {
            "T_max": (1, 1000),
            "eta_min": (0.0, 1.0)
        },
        "use_cases": [
            "Cyclical reduction with warm restarts (pair with WarmRestarts variant)",
            "Works well with long training and restarts"
        ],
        "notes": "Often used with restarts to find sharper minima."
    },

    "COSINEANNEALINGWARMRESTARTS": {
        "description": "Cosine annealing with periodic restarts.",
        "required": ["T_0"],
        "optional": ["T_mult", "eta_min"],
        "defaults": {
            "T_0": 10,
            "T_mult": 2,
            "eta_min": 0
        },
        "param_ranges": {
            "T_0": (1, 500),
            "T_mult": (1, 10)
        },
        "use_cases": [
            "Training where periodic restarts help escape local minima",
            "Modern architectures and transfer learning"
        ],
        "notes": "Tune T_0 based on dataset size and epoch count."
    },

    "REDUCELRONPLATEAU": {
        "description": "Reduce learning rate when a metric has stopped improving.",
        "required": ["mode"],
        "optional": ["factor", "patience", "threshold", "min_lr"],
        "defaults": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 1e-4,
            "min_lr": 0
        },
        "param_ranges": {
            "factor": (0.0, 1.0),
            "patience": (0, 100)
        },
        "use_cases": [
            "Fine-tuning or training when validation metric plateaus",
            "Works with any architecture and objective"
        ],
        "notes": "Monitor validation loss or metric; good as a fallback scheduler."
    },

    "ONECYCLELR": {
        "description": "One-cycle learning rate policy: increase then decrease over the course of training.",
        "required": [],
        "optional": ["max_lr", "pct_start", "anneal_strategy"],
        "defaults": {
            "max_lr": 0.1,
            "pct_start": 0.3,
            "anneal_strategy": "cos"
        },
        "param_ranges": {
            "max_lr": (1e-6, 1.0),
            "pct_start": (0.0, 1.0)
        },
        "use_cases": [
            "Often used to rapidly converge to a better learning rate",
            "Works well with a learning rate range test"
        ],
        "notes": "Requires knowing total number of steps/epochs; commonly used for classification."
    },

    "CYCLICLR": {
        "description": "Cycles the learning rate between two boundaries with a constant frequency.",
        "required": [],
        "optional": ["base_lr", "max_lr", "step_size_up", "mode"],
        "defaults": {
            "base_lr": 0.001,
            "max_lr": 0.006,
            "step_size_up": 2000,
            "mode": "triangular"
        },
        "param_ranges": {
            "base_lr": (1e-6, 1.0),
            "max_lr": (1e-6, 1.0)
        },
        "use_cases": [
            "When you want cyclical exploration of learning rates",
            "Often used for small datasets and short training runs"
        ],
        "notes": "Cycle length and amplitude are key; try a range test."
    }
}


def get_scheduler_config(scheduler_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a scheduler."""
    return SCHEDULER_CONFIGS.get(scheduler_name.upper())


def list_available_schedulers() -> list:
    """List all available schedulers."""
    return list(SCHEDULER_CONFIGS.keys())


def get_scheduler_defaults(scheduler_name: str) -> Dict[str, Any]:
    """Get default parameters for a scheduler."""
    config = get_scheduler_config(scheduler_name)
    return config.get("defaults", {}) if config else {}


def get_recommended_initial_lr(scheduler_name: str, base_lr: float = 0.001) -> float:
    """
    Get recommended initial learning rate adjustments based on scheduler choice.

    Args:
        scheduler_name: Name of scheduler
        base_lr: Base learning rate provided by user or optimizer defaults

    Returns:
        Recommended initial learning rate (may be adjusted for scheduler usage)
    """
    config = get_scheduler_config(scheduler_name)
    if not config:
        return base_lr

    # Heuristic adjustments
    if scheduler_name == "OneCycleLR":
        # OneCycle expects max_lr; start with a slightly larger base
        return min(base_lr * 10, 1.0)

    if scheduler_name in {"CosineAnnealingLR", "CosineAnnealingWarmRestarts"}:
        # Cosine often pairs with slightly higher start LR
        return min(base_lr * 5, 1.0)

    if scheduler_name == "ReduceLROnPlateau":
        # Use base lr directly; reduce when plateaued
        return base_lr

    # default: no change
    return base_lr
