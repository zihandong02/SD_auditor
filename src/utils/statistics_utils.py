# ── standard library ───────────────────────────────────────────────────
from typing import Optional, Sequence, Tuple, Union

from torch.distributions import Normal

# ── third-party libraries ──────────────────────────────────────────────
import torch

NumberLike = Union[float, torch.Tensor]
def sample_split(
    n: int,
    k: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.long,
) -> Tuple[torch.Tensor, ...]:
    """
    Randomly partition the index set ``{0, …, n-1}`` into *k* nearly
    equal, non-overlapping subsets.

    Parameters
    ----------
    n : int
        Total number of items to split. Must be non-negative.
    k : int
        Number of subsets to return. Must be ≥ 1.
    device : torch.device or None, optional
        Device on which the tensors are allocated. Defaults to CPU.
    dtype : torch.dtype, optional
        Data type of the returned index tensors (default: ``torch.long``).

    Returns
    -------
    Tuple[torch.Tensor, ...]
        A tuple of length *k*. Each element is a 1-D tensor containing
        a distinct slice of shuffled indices.

    Example
    -------
    >>> train_idx, test_idx = sample_split(1000, k=2)
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if n < 0:
        raise ValueError("n must be non-negative")

    # Random permutation on the requested device
    permuted = torch.randperm(n, dtype=dtype, device=device)

    # Evenly split into k pieces (size difference ≤ 1)
    splits: Sequence[torch.Tensor] = torch.tensor_split(permuted, k)

    return tuple(splits)


# ----------------------------------------------------------------------
# Wald CI helper
# ----------------------------------------------------------------------
def wald_ci(
    mu_hat: NumberLike,
    se: NumberLike,
    alpha_level: float = 0.05,
    *,
    as_tensor: bool = False,
) -> Tuple[NumberLike, NumberLike]:
    """
    Two-sided Wald confidence interval for a (potentially batched) normal estimator.

    Parameters
    ----------
    mu_hat : float | Tensor
        Point estimate(s).  Shape: arbitrary.
    se     : float | Tensor
        Standard error(s).  Must be broadcast-compatible with `mu_hat`.
    alpha_level : float, default 0.05
        Desired type-I error rate (two-sided).
    as_tensor : bool, default False
        • False → return Python floats if both inputs are floats.  
        • True  → always return `torch.Tensor`s on the proper device/dtype.

    Returns
    -------
    (lower, upper) : same type/shape as inputs unless `as_tensor=False`
        Wald confidence limits.
    """
    # Promote inputs to tensors for unified math  ------------------------
    mu_t = torch.as_tensor(mu_hat)
    se_t = torch.as_tensor(se).to(dtype=mu_t.dtype, device=mu_t.device)

    # z-critical on the correct device/dtype -----------------------------
    z = Normal(0.0, 1.0).icdf(
        torch.tensor(1 - alpha_level / 2, dtype=mu_t.dtype, device=mu_t.device)
    )

    lower = mu_t - z * se_t
    upper = mu_t + z * se_t

    # Return policy ------------------------------------------------------
    if as_tensor:
        return lower, upper            # tensors
    if isinstance(mu_hat, torch.Tensor) or isinstance(se, torch.Tensor):
        return lower, upper            # tensors (mixed inputs)
    return float(lower), float(upper)  # pure Python floats