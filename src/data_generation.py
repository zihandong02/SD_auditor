# ── standard library ───────────────────────────────────────────────────
from typing import Optional, Callable, Tuple, Union

# ── third-party libraries ──────────────────────────────────────────────
import torch
from torch.distributions import MultivariateNormal

# ── intra-package utilities ────────────────────────────────────────────
from . import utils  # set_global_seed, get_device, …

# ======================================================================
# Private helpers
# ======================================================================


def _as_tensor(
    x,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cast *x* (NumPy array, list, scalar, or Tensor) to a tensor on *device*."""
    device = utils.get_device() if device is None else device
    return torch.as_tensor(x, dtype=dtype, device=device)


def _sample_mv(
    n: int,
    d: int,
    cov: Optional[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = utils.get_device() if device is None else device
    mean = torch.zeros(d, device=device)
    if cov is None:
        cov = torch.eye(d, device=device)
    else:
        cov = _as_tensor(cov, device=device)
    dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
    return dist.sample((n,))


# ======================================================================
# 1) Complete-data generator
# ======================================================================


def lm_generate_complete_data(
    n: int,
    d_x: int,
    d_u1: int,
    d_u2: int,
    theta_star,
    beta1_star,
    beta2_star,
    *,
    Sigma_X: Optional[torch.Tensor] = None,
    Sigma_U1: Optional[torch.Tensor] = None,
    Sigma_U2: Optional[torch.Tensor] = None,
    sigma_eps: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Draw complete data for the linear model with two AI-generated pseudo-outcomes

        X   ~ ��(0, Σ_X)                      shape (n, d_x)
        θ̃_1 = θ* + ��(0, 0.2² I),  θ̃_2 = θ* + ��(0, 0.1² I)

        Y  = X·θ*  + ε
        W1 = X·θ̃_1 + ε₂ + 0.075·ε
        W2 = X·θ̃_2 + ε₃ + 0.075·ε
        V  = ��{|W1 − Y| ≤ |W2 − Y|}

    The three noise terms are scaled from the single ``sigma_eps`` argument:
    sd(ε) = 8·sigma_eps, sd(ε₂) = 2·sigma_eps, sd(ε₃) = 3·sigma_eps. With the
    default ``sigma_eps=2.0`` used for the reported figures this gives variances
    256, 16 and 36.

    θ̃_1 and θ̃_2 are re-drawn on every call.

    ``d_u1``, ``d_u2``, ``beta1_star``, ``beta2_star``, ``Sigma_U1`` and ``Sigma_U2``
    are retained for compatibility with the earlier latent-variable variants kept in
    the commented-out lines below, and are not used.

    Returns
    -------
    (X, Y, W1, W2, V)
        ``X`` has shape ``(n, d_x)``; the rest are column vectors ``(n, 1)``.
    """
    device = utils.get_device() if device is None else device

    # Parameters → tensors on target device
    theta_star = _as_tensor(theta_star, device=device)
    beta1_star = _as_tensor(beta1_star, device=device)
    beta2_star = _as_tensor(beta2_star, device=device)
    theta_star_shape = theta_star.shape

    # Randomly generate theta1 and theta2 with the same shape as theta_star
    theta1 = torch.randn(theta_star_shape, device=device)  * 0.2  # Using torch.randn for random normal distribution
    theta2 = torch.randn(theta_star_shape, device=device) * 0.1

    # 1) Latent covariates
    X = _sample_mv(n, d_x, Sigma_X, device=device)
    # U1 = _sample_mv(n, d_u1, Sigma_U1, device=device)
    # U2 = _sample_mv(n, d_u2, Sigma_U2, device=device)

    # 2) Independent Gaussian noise terms
    eps = torch.randn(n, device=device) * sigma_eps * 8
    eps2 = torch.randn(n, device=device) * sigma_eps * 2
    eps3 = torch.randn(n, device=device) * sigma_eps * 3

    # 3) Core linear parts
    X_theta = X @ theta_star
    # U1_beta = U1 @ beta1_star
    # U1_beetae = X @ (beta1_star + theta1)
    # U2_beta = U2 @ beta2_star
    X_theta1 = X @ (theta_star + theta1)
    X_theta2 = X @ (theta_star + theta2)

    # Y  = X_theta + U1_beta + eps
    # W1 = X_theta2 + U1_beta + eps2
    # W2 = X_theta2 + U1_beetae + eps3

    Y =  X_theta + eps
    W1 = X_theta1 + eps2 + eps * 0.075
    W2 = X_theta2 + eps3 + eps * 0.075

    # 4) Preference indicator
    V = (torch.abs(W1 - Y) <= torch.abs(W2 - Y)).long()
    #X = torch.cat([X, U1], dim=1)
    # 5) Shape to column vectors
    return (
        X,
        Y.unsqueeze(1),
        W1.unsqueeze(1),
        W2.unsqueeze(1),
        V.unsqueeze(1),
    )

# ======================================================================
# 2) MCAR missingness
# ======================================================================


def general_generate_mcar(
    X: torch.Tensor,
    Y: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    V: torch.Tensor,
    *,
    alpha: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Impose MCAR missingness with probabilities (α₁, α₂, α₃):

        Pattern 1 (α₁):  no values missing
        Pattern 2 (α₂):  Y missing
        Pattern 3 (α₃):  Y and V missing
    """
    device = X.device if device is None else device
    if not torch.isclose(alpha.sum(), torch.tensor(1.0, device=device)):
        raise ValueError("alpha must sum to 1")

    n = Y.shape[0]

    # Draw missingness pattern R ∈ {1, 2, 3}
    R = torch.multinomial(alpha, num_samples=n, replacement=True) + 1  # (n,)
    R = R.unsqueeze(1)

    # Clone observed copies (Y, V need float to hold NaNs)
    X_obs  = X.clone()
    Y_obs  = Y.clone().float()
    W1_obs = W1.clone()
    W2_obs = W2.clone()
    V_obs  = V.clone().float()

    # Apply masks
    mask2 = R == 2
    mask3 = R == 3
    Y_obs[mask2] = torch.nan
    Y_obs[mask3] = torch.nan
    V_obs[mask3] = torch.nan

    return X_obs, Y_obs, W1_obs, W2_obs, V_obs, R

# ======================================================================
# 3) Wrapper: complete data + MCAR
# ======================================================================


def lm_generate_obs_data_mcar(
    n: int,
    d_x: int,
    d_u1: int,
    d_u2: int,
    theta_star,
    beta1_star,
    beta2_star,
    *,
    alpha: torch.Tensor,
    Sigma_X: Optional[torch.Tensor] = None,
    Sigma_U1: Optional[torch.Tensor] = None,
    Sigma_U2: Optional[torch.Tensor] = None,
    sigma_eps: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Convenience wrapper: draw complete data and immediately impose MCAR.
    """
    device = utils.get_device() if device is None else device

    # 1) Complete data
    X, Y, W1, W2, V = lm_generate_complete_data(
        n, d_x, d_u1, d_u2,
        theta_star, beta1_star, beta2_star,
        Sigma_X=Sigma_X,
        Sigma_U1=Sigma_U1,
        Sigma_U2=Sigma_U2,
        sigma_eps=sigma_eps,
        device=device,
    )

    # 2) Apply MCAR
    # Debug: print alpha details before missingness generation
    return general_generate_mcar(
        X, Y, W1, W2, V,
        alpha=alpha,
        device=device,
    )


# ======================================================================
# MAR
# ======================================================================
def general_generate_mar(
    X: torch.Tensor,
    Y: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    V: torch.Tensor,
    *,
    alpha_fn: Union[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], torch.nn.Module],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Impose MAR missingness.  alpha_fn(X, W1, W2) -> (n, 3) tensor of pattern
    probabilities per row.  Patterns are the same as in the MCAR version.

    Parameters
    ----------
    alpha_fn : callable or nn.Module
        A function or PyTorch module that takes (X, W1, W2) and returns a (n, 3)
        tensor of pattern probabilities (α₁, α₂, α₃), each row summing to 1.
    """
    device = X.device if device is None else device

    # 1. Compute per-row probabilities
    alpha = alpha_fn(X, W1, W2).to(device)           # (n, 3)
    if alpha.ndim != 2 or alpha.shape[1] != 3:
        raise ValueError("alpha_fn must return a (n, 3) tensor")
    if not torch.allclose(alpha.sum(dim=1), torch.ones_like(alpha[:, 0])):
        raise ValueError("each row of alpha must sum to 1")

    n = Y.shape[0]

    # 2. Sample missing-pattern indicator R ∈ {1,2,3}
    R = torch.multinomial(alpha, num_samples=1).squeeze(1) + 1  # (n,)
    R = R.unsqueeze(1)

    # 3. Clone observed copies (Y, V must be float to hold NaNs)
    X_obs  = X.clone()
    Y_obs  = Y.clone().float()
    W1_obs = W1.clone()
    W2_obs = W2.clone()
    V_obs  = V.clone().float()

    # 4. Mask according to R
    mask2 = R == 2
    mask3 = R == 3
    Y_obs[mask2] = torch.nan
    Y_obs[mask3] = torch.nan
    V_obs[mask3] = torch.nan

    return X_obs, Y_obs, W1_obs, W2_obs, V_obs, R

def lm_generate_obs_data_mar(
    n: int,
    d_x: int,
    d_u1: int,
    d_u2: int,
    theta_star,
    beta1_star,
    beta2_star,
    *,
    alpha_fn: Union[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], torch.nn.Module],
    Sigma_X: Optional[torch.Tensor] = None,
    Sigma_U1: Optional[torch.Tensor] = None,
    Sigma_U2: Optional[torch.Tensor] = None,
    sigma_eps: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Convenience wrapper: sample complete data from the linear-model generator
    and immediately impose MAR missingness.

    Parameters
    ----------
    n : int
        Sample size.
    d_x, d_u1, d_u2 : int
        Dimensions of X, U1, U2.
    theta_star, beta1_star, beta2_star
        Ground-truth coefficients for the linear model (passed straight through
        to `lm_generate_complete_data`).
    alpha_fn : Callable
        Function mapping (X, W1, W2) → (n, 3) tensor of row-wise pattern
        probabilities (α₁, α₂, α₃); each row must sum to 1.
    Sigma_X, Sigma_U1, Sigma_U2 : torch.Tensor or None
        Covariance matrices for X, U1, U2 (optional).
    sigma_eps : float
        Standard deviation of the model error ε.
    device : torch.device or None
        Target device; defaults to whatever `utils.get_device()` returns.

    Returns
    -------
    Tuple[torch.Tensor, ...]
        (X_obs, U1, U2, Y_obs, W1_obs, W2_obs, V_obs, R)
    """

    # 1) Resolve device
    device = utils.get_device() if device is None else device

    # 2) Generate complete data
    X, Y, W1, W2, V = lm_generate_complete_data(
        n, d_x, d_u1, d_u2,
        theta_star, beta1_star, beta2_star,
        Sigma_X=Sigma_X,
        Sigma_U1=Sigma_U1,
        Sigma_U2=Sigma_U2,
        sigma_eps=sigma_eps,
        device=device,
    )

    # 3) Impose MAR missingness
    X_obs, Y_obs, W1_obs, W2_obs, V_obs, R = general_generate_mar(
        X, Y, W1, W2, V,
        alpha_fn=alpha_fn,
        device=device,
    )

    return X_obs, Y_obs, W1_obs, W2_obs, V_obs, R