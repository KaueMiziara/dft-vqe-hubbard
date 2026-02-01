from dataclasses import dataclass


@dataclass
class VQEResult:
    """
    Standardized container for VQE optimization results.
    """

    energy: float
    optimal_params: list[float]
    history: list[float]
    converged: bool
