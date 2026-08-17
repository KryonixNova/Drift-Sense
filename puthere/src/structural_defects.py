import numpy as np


def maybe_collapse_gap(
    gap_nm: float,
    threshold_nm: float,
    rng: np.random.Generator,
    collapse_prob: float = 0.7,
) -> bool:
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)
