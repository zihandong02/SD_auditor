# =============================================================================
# estimators.py  –  Statistical objects for MCAR mono-debias (pure Torch)
#
# Table of Contents (only the functions shown in the screenshot, in order)
# -----------------------------------------------------------------------------
# 0) Ordinary Least Squares (OLS) & Weighted Least Squares (WLS)
#    lm_fit_ols(...)                            # OLS regression
#    lm_fit_wls(...)                            # WLS regression
# 1) ψ-models
#    lm_build_all_psi(...)                      # compute ψ-values (no gradients)
#
# 2) φ-functions / phi-functions
#    general_build_all_phi(...)                 # evaluate φ on a fixed alpha
#    general_build_all_phi_function(...)        # return callables φ1/φ2/φ3
#    general_build_all_phi_function_mar(...)  # return callables φ1/φ2/φ3 (MAR)
#
# 3) Empirical moments
#    general_estimate_moments_mcar(...)         # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j)
#    general_estimate_moments_mar(...)          # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j) (MAR)
#    general_estimate_moments_function_mcar(...)# return moment_fn(alpha_vec)
#    general_estimate_moments_function_mar(...) # return moment_fn(alpha_vec) (MAR)
#
# 4) M-matrix & variance
#    general_estimate_m_matrix_mcar(...)        # estimate M^(1)
#    general_estimate_m_matrix_mar(...)         # estimate M^(1) (MAR)
#    general_estimate_variance_mcar(...)        # estimate Cov(θ̂)
#    general_estimate_variance_mar(...)         # estimate Cov(θ̂) (MAR)
#
# 5) g(alpha1) under budget constraint
#    general_get_trace_variance_function_alpha_mcar(...)  # return g(α1)
#    general_get_cov00_function_alpha_mcar(...)  # return Cov(ψ,φ₁) for g(α1)
#    general_get_trace_variance_function_alpha_mar(...)  # return g(α1) (MAR)
#    general_get_cov00_function_alpha_mar(...)  # return Cov(ψ,φ₁) for g(α1) (MAR)
# 6) One-dimensional search for α1
#    _golden_section(...)                       # derivative-free golden section
#    _adam_section(...)                         # gradient-based Adam search
#    search_alpha_mcar(...)                     # public API: choose "golden"/"adam"
#    train_alpha_model(...)                  # train AlphaModel for MAR
#    train_alpha_with_lagrangian(...)  # train AlphaModel with Lagrangian
#
# 7) Efficient estimator (Algorithm 3, MCAR)
#    lm_mono_debias_estimate(...)          # 3-fold cross-fit efficient θ̂
# =============================================================================


from __future__ import annotations
from typing import Dict, Sequence, Tuple, List, Callable, Literal, Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from .models  import train_model, AlphaModel

# ============================================================
# Ordinary Least Squares (OLS) – Torch version
# ============================================================

def lm_fit_ols(X: Tensor, Y: Tensor) -> Tensor:
    """
    Fit an Ordinary Least Squares (OLS) regression model.

    Parameters
    ----------
    X : Tensor (n_samples, n_features)
        Design matrix.
    Y : Tensor (n_samples, 1)
        Response vector.

    Returns
    -------
    coeff : Tensor (n_features,)
        Estimated OLS coefficients.
    """
    # Use least-squares solver (stable on CPU / CUDA)
    #   θ̂ = (XᵀX)⁻¹ XᵀY   is equivalent to   lstsq(X, Y)
    sol = torch.linalg.lstsq(X, Y).solution   # (n_features, 1)
    return sol.squeeze(1)                                  # (n_features,)


# ============================================================
# Weighted Least Squares (WLS) – Torch version
# ============================================================

def lm_fit_wls(
    X: Tensor,
    Y: Tensor,
    w: Optional[Tensor] = None
) -> Tensor:
    """
    Fit a Weighted Least Squares (WLS) regression model.

    Parameters
    ----------
    X : Tensor (n_samples, n_features)
        Design matrix.
    Y : Tensor (n_samples, 1)
        Response vector (must be a column vector).
    w : Tensor (n_samples,), optional
        Sample weights. If None or all ones, falls back to OLS.

    Returns
    -------
    coeff : Tensor (n_features,)
        Estimated WLS coefficients.
    """
    if w is None or torch.allclose(w, torch.ones_like(Y.squeeze())):
        return lm_fit_ols(X, Y)

    # Re-scale rows by sqrt(weights):  X̃ = diag(√w) · X,  Ỹ = √w · Y
    sqrt_w = torch.sqrt(w).unsqueeze(1)        # (n_samples, 1)
    Xw = X * sqrt_w                            # (n_samples, n_features)
    Yw = Y * sqrt_w                            # (n_samples, 1)

    sol = torch.linalg.lstsq(Xw, Yw).solution  # (n_features, 1)
    return sol.squeeze(1)                                    # (n_features,)

# ----------------------------------------------------------------------
# 1) ψ-functions   ------------------------------------------------------
# ----------------------------------------------------------------------

def lm_build_all_psi(
    X:  Tensor,
    Y:  Tensor,
    W1: Tensor,
    W2: Tensor,
    V:  Tensor,
    theta: Tensor,
    *,
    method: str = "mlp",
) -> Tuple[
    Callable[[Tensor, Tensor],                   Tensor],   # ψ₁
    Callable[[Tensor, Tensor, Tensor, Tensor],   Tensor],   # ψ₂
    Callable[[Tensor, Tensor, Tensor],           Tensor],   # ψ₃
]:
    """
    Returns ψ₁, ψ₂, ψ₃ that accept batches and return (n_batch, d) corrections.
    All inputs must be float32 and on the same device.
    """
    # ------------------------------------------------------------------
    # Reshape inputs to (n, 1) where necessary
    # ------------------------------------------------------------------
    Y  = Y.view(-1, 1)
    W1 = W1.view(-1, 1)
    W2 = W2.view(-1, 1)
    V  = V.view(-1, 1)
    theta = theta.view(-1, 1)

    # ------------------------------------------------------------------
    # Dimensions and precomputed inverse
    # ------------------------------------------------------------------
    n, d = X.shape
    inv_exx = torch.inverse((X.T @ X) / n)  # (d, d)

    # ------------------------------------------------------------------
    # Train auxiliary models for ψ₂ and ψ₃
    # ------------------------------------------------------------------
    feats_psi2 = torch.cat([X, W1, W2, V], dim=1)
    feats_psi3 = torch.cat([X, W1, W2], dim=1)

    with torch.enable_grad():
        # ψ₂
        psi_model_2 = train_model(
            feats_psi2, Y,
            method=method,        # "mlp", "mlpclass", ...
            epochs=200,           # total training epochs
            lr=2e-3,              # initial learning rate
            scheduler_name="cosine",   # (default) cosine decay → 0
            # scheduler_kw={}     # extra kwargs if needed
        ).eval()

        # ψ₃
        psi_model_3 = train_model(
            feats_psi3, Y,
            method=method,
            epochs=200,
            lr=2e-3,
            scheduler_name="cosine"
        ).eval()

    # ------------------------------------------------------------------
    # Core helper: ψ_core for batch input
    # ------------------------------------------------------------------
    def _psi_core_batch(Xv: Tensor, Y_like: Tensor) -> Tensor:
        """
        Xv:      (n_val, d)
        Y_like:  (n_val, 1)
        Returns: (n_val, d)
        """
        resid = Y_like - Xv @ theta                   # (n_val, 1)
        psi_batch = (Xv.unsqueeze(2) * resid.unsqueeze(1))  # (n_val, d, 1)
        return torch.matmul(psi_batch.squeeze(2), inv_exx.T)  # (n_val, d)

    # ------------------------------------------------------------------
    # ψ functions that support batch input
    # ------------------------------------------------------------------
    def psi_1(X_val: Tensor, Y_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        Yv = Y_val.view(-1, 1)
        return _psi_core_batch(Xv, Yv)  # (n_val, d)

    def psi_2(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        feats = torch.cat([
            Xv,
            W1_val.view(-1, 1),
            W2_val.view(-1, 1),
            V_val.view(-1, 1)
        ], dim=1)
        Y_hat = psi_model_2(feats).view(-1, 1)   # (n_val, 1)
        return _psi_core_batch(Xv, Y_hat)        # (n_val, d)

    def psi_3(X_val: Tensor, W1_val: Tensor, W2_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        feats = torch.cat([
            Xv,
            W1_val.view(-1, 1),
            W2_val.view(-1, 1)
        ], dim=1)
        Y_hat = psi_model_3(feats).view(-1, 1)   # (n_val, 1)
        return _psi_core_batch(Xv, Y_hat)        # (n_val, d)

    return psi_1, psi_2, psi_3



def lm_build_all_psi_weighted(
    X: Tensor,                   # (n, d)
    Y: Tensor,                   # (n, 1)
    W1: Tensor,                  # (n, 1)
    W2: Tensor,                  # (n, 1)
    V: Tensor,                   # (n, 1)
    theta: Tensor,               # (d, 1) current estimate
    w: Optional[Tensor] = None,  # (n,) optional sample weights
    method: str = 'default'
) -> Tuple[Callable, Callable, Callable]:
    """
    Constructs psi_1, psi_2, psi_3 functions.  If `w` is provided,
    uses weights in computing inv_exx for psi_1 (and downstream psi).
    """
    # ------------------------------------------------------------------
    # Reshape inputs to (n, 1) where necessary
    # ------------------------------------------------------------------
    Y  = Y.view(-1, 1)
    W1 = W1.view(-1, 1)
    W2 = W2.view(-1, 1)
    V  = V.view(-1, 1)
    theta = theta.view(-1, 1)

    n, d = X.shape

    # ------------------------------------------------------------------
    # Precompute weighted or unweighted inverse X'X
    # ------------------------------------------------------------------
    if w is not None:
        w = w.view(-1, 1)
        weighted_X = X * w.sqrt()  # (n, d)
        inv_exx = torch.inverse((weighted_X.T @ weighted_X) / w.sum())  # weighted
    else:
        inv_exx = torch.inverse((X.T @ X) / n)  # unweighted

    # ------------------------------------------------------------------
    # Train auxiliary models for ψ₂ and ψ₃
    # ------------------------------------------------------------------
    feats_psi2 = torch.cat([X, W1, W2, V], dim=1)
    feats_psi3 = torch.cat([X, W1, W2], dim=1)

    with torch.enable_grad():
        # ψ₂
        psi_model_2 = train_model(
            feats_psi2, Y,
            method=method,        # "mlp", "mlpclass", ...
            epochs=300,           # total training epochs
            lr=2e-3,              # initial learning rate
            scheduler_name="cosine",   # (default) cosine decay → 0
            # scheduler_kw={}     # extra kwargs if needed
        ).eval()

        # ψ₃
        psi_model_3 = train_model(
            feats_psi3, Y,
            method=method,
            epochs=300,
            lr=2e-3,
            scheduler_name="cosine"
        ).eval()

    # ------------------------------------------------------------------
    # Core helper: ψ_core for batch input, uses captured inv_exx
    # ------------------------------------------------------------------
    def _psi_core_batch(Xv: Tensor, Y_like: Tensor) -> Tensor:
        """
        Xv:      (batch, d)
        Y_like:  (batch, 1)
        Returns: (batch, d)
        """
        resid = Y_like - Xv @ theta          # (batch,1)
        # compute X_i * resid_i outer product, then apply inv_exx
        psi_batch = Xv.unsqueeze(2) * resid.unsqueeze(1)  # (batch, d, d? actually d,1->d,d?)
        # squeeze extra dim and apply inverse
        return torch.matmul(psi_batch.squeeze(2), inv_exx.T)  # (batch, d)

    # ------------------------------------------------------------------
    # ψ functions
    # ------------------------------------------------------------------
    def psi_1(X_val: Tensor, Y_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        Yv = Y_val.view(-1, 1)
        return _psi_core_batch(Xv, Yv)

    def psi_2(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        feats = torch.cat([
            Xv,
            W1_val.view(-1, 1),
            W2_val.view(-1, 1),
            V_val.view(-1, 1)
        ], dim=1)
        Y_hat = psi_model_2(feats).view(-1, 1)
        return _psi_core_batch(Xv, Y_hat)

    def psi_3(X_val: Tensor, W1_val: Tensor, W2_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        feats = torch.cat([
            Xv,
            W1_val.view(-1, 1),
            W2_val.view(-1, 1)
        ], dim=1)
        Y_hat = psi_model_3(feats).view(-1, 1)
        return _psi_core_batch(Xv, Y_hat)

    return psi_1, psi_2, psi_3
# ----------------------------------------------------------------------
# 2) φ-functions / phi-functions
# ----------------------------------------------------------------------

def general_build_all_phi(
    psi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    psi_3: Callable[[Tensor, Tensor, Tensor], Tensor],
    alpha: Tensor,  # (3,) Tensor: [alpha1, alpha2, alpha3]
) -> Tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
]:
    """
    Construct φ₁, φ₂, φ₃ from ψ₂, ψ₃ and fixed scalar weights α = [α₁, α₂, α₃].

    Each φ_j has signature:
        phi_j(X_val, W1_val, W2_val, V_val) -> Tensor of shape (n, d)
    """

    def phi_1(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        n = X_val.shape[0]
        a1 = alpha[0].expand(n, 1)
        a2 = alpha[1].expand(n, 1)
        a3 = alpha[2].expand(n, 1)

        denom_sum = a1 + a2
        denom_prod = denom_sum * a1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        term1 = (a2 / denom_prod) * psi2_val
        term2 = (a3 / denom_sum) * psi3_val

        return -term1 - term2

    def phi_2(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        n = X_val.shape[0]
        a1 = alpha[0].expand(n, 1)
        a2 = alpha[1].expand(n, 1)
        a3 = alpha[2].expand(n, 1)

        denom_sum = a1 + a2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1.0 / denom_sum) * psi2_val - (a3 / denom_sum) * psi3_val

    def phi_3(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3


def general_build_all_phi_mar(
    psi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    psi_3: Callable[[Tensor, Tensor, Tensor], Tensor],
    alpha_model: Union[Callable[[Tensor, Tensor, Tensor], Tensor], nn.Module],
) -> Tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
]:
    """
    Construct φ₁, φ₂, φ₃ from ψ₂, ψ₃ and a trained alpha_model producing weights [α₁, α₂, α₃].

    Each φ_j has signature:
        phi_j(X_val, W1_val, W2_val, V_val) -> Tensor of shape (n, d)

    All inputs and outputs are torch.Tensor and assumed on the same device.
    """

    def phi_1(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        alpha = alpha_model(X_val, W1_val, W2_val)  # (n, 3)
        a1, a2, a3 = alpha[:, 0:1], alpha[:, 1:2], alpha[:, 2:3]  # shape (n, 1)

        denom_sum = a1 + a2
        denom_prod = denom_sum * a1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return - (a2 / denom_prod) * psi2_val - (a3 / denom_sum) * psi3_val

    def phi_2(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        alpha = alpha_model(X_val, W1_val, W2_val)
        a1, a2, a3 = alpha[:, 0:1], alpha[:, 1:2], alpha[:, 2:3]

        denom_sum = a1 + a2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1.0 / denom_sum) * psi2_val - (a3 / denom_sum) * psi3_val

    def phi_3(X_val: Tensor, W1_val: Tensor, W2_val: Tensor,  V_val: Optional[Tensor] = None) -> Tensor:
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3


def general_build_all_phi_function(
    psi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    psi_3: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
]:
    """
    Return three φ-functions for MCAR debiasing, fully in PyTorch.

    Each phi_j has signature
        phi_j(alpha, X_val, W1_val, W2_val, V_val) -> Tensor shape (d,)

    All inputs and outputs are torch.Tensor.
    """
    def phi_1(
        alpha: Tensor,
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor,
    ) -> Tensor:
        # alpha: (3,)
        alpha1, alpha2, alpha3 = alpha.unbind(0)
        denom_sum  = alpha1 + alpha2
        denom_prod = denom_sum * alpha1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return -(alpha2 / denom_prod) * psi2_val \
               - (alpha3 / denom_sum) * psi3_val

    def phi_2(
        alpha: Tensor,
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor,
    ) -> Tensor:
        alpha1, alpha2, alpha3 = alpha.unbind(0)
        denom_sum = alpha1 + alpha2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1.0 / denom_sum) * psi2_val \
               - (alpha3 / denom_sum) * psi3_val

    def phi_3(
        alpha: Tensor,
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor,
    ) -> Tensor:
        # phi_3 = ψ₃, independent of alpha and V
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3


def general_build_all_phi_function_mar(
    psi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    psi_3: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
]:
    """
    Build φ-functions for MAR debiasing using a function or trainable alpha_model.

    Each phi_j has signature:
        phi_j(
            X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor
        ) -> Tensor of shape (batch, d)
    """
    def phi_1(
        alpha_model: Union[Callable[[Tensor, Tensor, Tensor], Tensor], nn.Module],
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor
    ) -> Tensor:
        # α: (batch,3)
        alpha = alpha_model(X_val, W1_val, W2_val)
        alpha1 = alpha[:, 0:1]
        alpha2 = alpha[:, 1:2]
        alpha3 = alpha[:, 2:3]
        denom_sum = alpha1 + alpha2      # (batch,1)
        denom_prod = denom_sum * alpha1  # (batch,1)

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)  # (batch,d)
        psi3_val = psi_3(X_val, W1_val, W2_val)         # (batch,d)

        return - (alpha2 / denom_prod) * psi2_val \
               - (alpha3 / denom_sum) * psi3_val

    def phi_2(
        alpha_model: Union[Callable[[Tensor, Tensor, Tensor], Tensor], nn.Module],
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor
    ) -> Tensor:
        alpha = alpha_model(X_val, W1_val, W2_val)
        alpha1 = alpha[:, 0:1]
        alpha2 = alpha[:, 1:2]
        alpha3 = alpha[:, 2:3]
        denom_sum = alpha1 + alpha2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1.0 / denom_sum) * psi2_val \
               - (alpha3 / denom_sum) * psi3_val

    def phi_3(
        alpha_model: Union[Callable[[Tensor, Tensor, Tensor], Tensor], nn.Module],
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor
    ) -> Tensor:
        # phi_3 = ψ₃, independent of alpha_model and V
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3
# ----------------------------------------------------------------------
# 3) empirical moments
# ----------------------------------------------------------------------

def general_estimate_moments_mcar(
    psi_1: Callable[[Tensor, Tensor], Tensor],
    phi_1: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_3: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    W2: Tensor,
    V: Tensor,
    R: Tensor  # Assuming R is the pattern indicator tensor
) -> Dict[str, List[Tensor] | Tensor]:
    """
    Estimate second-order moments and cross-moments under MCAR.

    Assumptions
    -----------
    * All input tensors are on the same device and have the same dtype.
    * psi_1, phi_1, phi_2, phi_3 are batch functions:
        - psi_1(X, Y)             -> (n, d)
        - phi_j(X, W1, W2, V)     -> (n, d)

    Returns
    -------
    dict
        {
          "E[psi1 psi1^T]"   : (d, d) Tensor,
          "E[phi_j phi_j^T]" : list of three (d, d) Tensors,
          "Cov(psi1, phi1)"  : (d, d) Tensor   # only φ₁
        }
    """
    n, d = X.shape

    # 1. Create boolean masks for each pattern 
    R_vec = R.squeeze()                # (n,)
    mask1 = R_vec == 1
    mask2 = R_vec == 2
    mask3 = R_vec == 3

    # 2. Split data explicitly for each missing pattern
    X1, Y1, W1_1, W2_1, V1 = X[mask1], Y[mask1], W1[mask1], W2[mask1], V[mask1]
    X2, W1_2, W2_2, V2 = X[mask2], W1[mask2], W2[mask2], V[mask2]
    X3, W1_3, W2_3 = X[mask3], W1[mask3], W2[mask3]

    # 3. Evaluate psi and phi functions on separated datasets
    psi1_1 = psi_1(X1, Y1)                     # (n1, d)
    phi1_1 = phi_1(X1, W1_1, W2_1, V1)         # (n1, d)
    phi2_2 = phi_2(X2, W1_2, W2_2, V2)         # (n2, d)
    phi3_3 = phi_3(X3, W1_3, W2_3, None)       # (n3, d)

    # Estimate E[phi_j phi_j^T] for each pattern
    E_phi_phiT_list = [
        (phi1_1.T @ phi1_1) / X1.shape[0],
        (phi2_2.T @ phi2_2) / X2.shape[0],
        (phi3_3.T @ phi3_3) / X3.shape[0]
    ]
    # E_psi1_psi1T = (psi1_1.T @ psi1_1) / X1.shape[0]

    # # *** DEBUG print ***
    # for idx, E_phi in enumerate(E_phi_phiT_list, 1):
    #     print(f"\nE[phi{idx} phi{idx}^T] =\n", E_phi)
    # print("\nE[psi1 psi1^T] =\n", E_psi1_psi1T)
    # 4. Pack results
    return {
        "E[psi1 psi1^T]":   (psi1_1.T @ psi1_1) / X1.shape[0],
        "E[phi_j phi_j^T]": E_phi_phiT_list,
        "Cov(psi1, phi1)": (psi1_1.T @ phi1_1) / X1.shape[0],
    }


def general_estimate_moments_mcar_only_complete_data(
    psi_1: Callable[[Tensor, Tensor], Tensor],
    phi_1: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_3: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    W2: Tensor,
    V: Tensor,
    debug: bool = False,  # Optional: print matrices when True
) -> Dict[str, List[Tensor] | Tensor]:
    """
    Estimate second-order moments and selected cross-moments under MCAR.

    Returns
    -------
    dict
        {
          "E[psi1 psi1^T]": (d, d) Tensor,
          "E[phi_j phi_j^T]": list[(d, d)],  # [φ₁, φ₂, φ₃]
          "Cov(psi1, phi1)": (d, d) Tensor,
        }
    """
    n, d = X.shape

    # 1. Evaluate basis functions on the full batch
    psi1_vals = psi_1(X, Y)                     # (n, d)
    phi1_vals = phi_1(X, W1, W2, V)             # (n, d)
    phi2_vals = phi_2(X, W1, W2, V)             # (n, d)
    phi3_vals = phi_3(X, W1, W2, V)             # (n, d)

    # 2. E[ψ₁ ψ₁ᵀ]
    E_psi1_psi1T = (psi1_vals.T @ psi1_vals) / n

    # 3. E[φⱼ φⱼᵀ] and Cov(ψ₁, φ₁)
    E_phi_phiT_list = [
        (phi1_vals.T @ phi1_vals) / n,          # φ₁
        (phi2_vals.T @ phi2_vals) / n,          # φ₂
        (phi3_vals.T @ phi3_vals) / n,          # φ₃
    ]
    Cov_psi1_phi1 = (psi1_vals.T @ phi1_vals) / n

    # 4. Optional debug output
    if debug:
        torch.set_printoptions(precision=4, sci_mode=False)
        print("\nE[psi1 psi1^T] =\n", E_psi1_psi1T)
        for idx, E_phi in enumerate(E_phi_phiT_list, 1):
            print(f"\nE[phi{idx} phi{idx}^T] =\n", E_phi)
        print("\nCov(psi1, phi1) =\n", Cov_psi1_phi1)

    # 5. Pack results
    return {
        "E[psi1 psi1^T]":   E_psi1_psi1T,
        "E[phi_j phi_j^T]": E_phi_phiT_list,
        "Cov(psi1, phi1)":  Cov_psi1_phi1,
    }

def general_estimate_moments_mar( 
    psi_1: Callable[[Tensor, Tensor], Tensor],
    phi_1: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_3: Callable[[Tensor, Tensor, Tensor, Optional[Tensor]], Tensor],
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    W2: Tensor,
    V: Tensor,
    R: Tensor,                    # (n, 1) missing-pattern indicator
    alpha: Tensor                 # (n, 3) pattern probabilities
) -> Dict[str, Tensor]:
    """
    Compute MAR-based moment estimates by explicitly separating data based on R.
    """
    n, d = X.shape
    device = X.device

    # 1. Create boolean masks for each pattern
    R_vec = R.squeeze()                # (n,)
    mask1 = R_vec == 1
    mask2 = R_vec == 2
    mask3 = R_vec == 3

    # 2. Split data explicitly for each missing pattern
    X1, Y1, W1_1, W2_1, V1 = X[mask1], Y[mask1], W1[mask1], W2[mask1], V[mask1]
    X2, W1_2, W2_2, V2 = X[mask2], W1[mask2], W2[mask2], V[mask2]
    X3, W1_3, W2_3 = X[mask3], W1[mask3], W2[mask3]

    # 3. Evaluate psi and phi functions on separated datasets
    psi1_1 = psi_1(X1, Y1)                     # (n1, d)
    phi1_1 = phi_1(X1, W1_1, W2_1, V1)         # (n1, d)
    phi2_2 = phi_2(X2, W1_2, W2_2, V2)         # (n2, d)
    phi3_3 = phi_3(X3, W1_3, W2_3, None)       # (n3, d)

    alpha1 = alpha[mask1].clamp_min(1e-12)     # (n1,)

    # 4. Compute E[1{R=1} ψ₁ ψ₁ᵀ / α₁²]
    w_psi = (1.0 / alpha1**2).unsqueeze(1)                         # (n1, 1)
    E_psi1_scaled2 = ((psi1_1 * w_psi).T @ psi1_1) / n             # (d, d)

    # 5. Compute E[Σ_j 1{R=j} φⱼ φⱼᵀ] explicitly for each pattern
    E_phi_agg = (
        (phi1_1.T @ phi1_1) + (phi2_2.T @ phi2_2) + (phi3_3.T @ phi3_3)
    ) / n                                                          # (d, d)

    # 6. Compute E[1{R=1} ψ₁ φ₁ᵀ / α₁]
    w_cross = (1.0 / alpha1).unsqueeze(1)                          # (n1, 1)
    E_psi1_phi1_scaled = ((psi1_1 * w_cross).T @ phi1_1) / n       # (d, d)

    return {
        "E_psi1_scaled2": E_psi1_scaled2,
        "E_phi_agg": E_phi_agg,
        "E_psi1_phi1_scaled": E_psi1_phi1_scaled,
    }

def general_estimate_moments_function_mcar(
    psi_1: Callable[[Tensor, Tensor], Tensor],
    phi_1: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_2: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_3: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    W2: Tensor,
    V: Tensor,
) -> Callable[[Tensor], Dict[str, List[Tensor] | Tensor]]:
    """
    Construct a function moment_fn(alpha) that computes empirical moment matrices 
    under the MCAR setting, based on precomputed psi_1 and on-demand phi_j(alpha).

    Assumptions
    -----------
    - psi_1 is batch-aware:      psi_1(X, Y) -> (n, d)
    - phi_j is batch-aware:      phi_j(alpha, X, W1, W2, V) -> (n, d)
    - All tensors share the same device and dtype.
    """
    n, d = X.shape
    # --------------------------------------------------------
    # 1) Pre-compute ψ₁ values (independent of alpha) --------
    # --------------------------------------------------------
    with torch.no_grad():
        psi1_vals: Tensor       = psi_1(X, Y)                      # (n, d)
        E_psi1_psi1T: Tensor    = (psi1_vals.T @ psi1_vals) / n    # (d, d)

    # --------------------------------------------------------
    # 2) Build and return moment_fn --------------------------
    # --------------------------------------------------------
    def moment_fn(alpha: Tensor) -> Dict[str, List[Tensor] | Tensor]:
        # Compute phi_j(alpha) for j = 1,2,3
        phi_vals = [
            phi_func(alpha, X, W1, W2, V)
            for phi_func in (phi_1, phi_2, phi_3)
        ]

        # Compute E[phi_j phi_j^T] and Cov(psi1, phi_j)
        E_phi_phiT_list = [(phi.T @ phi) / n for phi in phi_vals]
        Cov_psi_phi_list = [(psi1_vals.T @ phi) / n for phi in phi_vals]

        return {
            "E[psi1 psi1^T]":   E_psi1_psi1T,
            "E[phi_j phi_j^T]": E_phi_phiT_list,
            "Cov(psi1, phi_j)": Cov_psi_phi_list,
        }

    return moment_fn


def general_estimate_moments_function_mar(
    psi_1: Callable[[Tensor, Tensor], Tensor],
    phi_1: Callable[[nn.Module, Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_2: Callable[[nn.Module, Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_3: Callable[[nn.Module, Tensor, Tensor, Tensor, Tensor], Tensor],
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    W2: Tensor,
    V: Tensor,
    c: float,
) -> Callable[[nn.Module], Dict[str, Union[Tensor, List[Tensor]]]]:
    """
    Build a moment_fn(alpha_model) that returns:
      - E[psi1 psi1^T / alpha1]
      - E[alpha_j * phi_j phi_j^T]  for j=1..3
      - Cov(psi1, phi_j)             for j=1..3
      - E[c*alpha1 + alpha2]

    All empirical expectations are taken over the sample of size n.
    """
    n, d = X.shape

    # 1) Precompute psi1 values once (no grad needed)
    with torch.no_grad():
        psi1_vals: Tensor = psi_1(X, Y)           # (n, d)

    def moment_fn(alpha_model: nn.Module) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Args:
            alpha_model:  nn.Module mapping (X,W1,W2) -> (n,3) α‐weights

        Returns:
            A dict containing weighted second‐moments, covariances, and the cost expectation.
        """
        # 2) Compute α‐weights
        alpha_vals: Tensor = alpha_model(X, W1, W2)  # (n, 3)
        alpha1 = alpha_vals[:, 0]                    # (n,)
        alpha2 = alpha_vals[:, 1]
        alpha3 = alpha_vals[:, 2]

        # 3) Compute phi_j(alpha) for j = 1,2,3
        phi_vals: List[Tensor] = [
            phi_func(alpha_model, X, W1, W2, V)
            for phi_func in (phi_1, phi_2, phi_3)
        ]  # each (n, d)

        # 4) Weighted E[psi1 psi1^T / alpha1]
        #    = (1/n) Σ_i [ psi1_i psi1_i^T / alpha1_i ]
        psi1_div = psi1_vals / alpha1.unsqueeze(1)  # (n, d)
        E_psi1_psi1_over_alpha1: Tensor = psi1_vals.T @ psi1_div / n  # (d, d)

        # 5) Weighted E[alpha_j * phi_j phi_j^T]
        E_alpha_phi_phiT_list: List[Tensor] = []
        for j, alpha_j in enumerate((alpha1, alpha2, alpha3)):
            phi = phi_vals[j]                         # (n, d)
            # φᵀ diag(α) φ = Σ_i α_i * φ_i φ_iᵀ
            E_alpha_phi_phiT_list.append((phi.T @ (phi * alpha_j.unsqueeze(1))) / n)

        # 6) Covariance Cov(psi1, φ_j) = E[psi1 φ_j^T]
        Cov_psi_phi_list: List[Tensor] = [
            (psi1_vals.T @ phi) / n
            for phi in phi_vals
        ]

        # 7) Cost expectation E[c*α1 + α2]
        cost_expectation: Tensor = torch.mean(c * alpha1 + alpha2)  # scalar

        return {
            "E[psi1 psi1^T / alpha1]":   E_psi1_psi1_over_alpha1,
            "E[alpha_j phi_j phi_j^T]":  E_alpha_phi_phiT_list,
            "Cov(psi1, phi_j)":          Cov_psi_phi_list,
            "E[c alpha1 + alpha2]":      cost_expectation,
        }

    return moment_fn
# ----------------------------------------------------------------------
# 4) M-matrix & variance
# ----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Diagnostic: largest spectral norm of the calibration matrix M seen so far.
#
# The bounded-calibration formulation constrains M to the spectral-norm ball
# {M : ||M||_sp <= C_M}. The estimators below apply the unconstrained projection
# coefficient, which coincides with the constrained minimiser whenever it lies inside
# the ball. Recording the realised maximum lets a run certify that a declared C_M was
# never binding.
# ---------------------------------------------------------------------------
_M_SPEC_MAX = {"value": 0.0}


def reset_m_spec_norm() -> None:
    """Reset the running maximum of ||M||_sp."""
    _M_SPEC_MAX["value"] = 0.0


def get_m_spec_norm() -> float:
    """Largest ||M||_sp recorded since the last reset."""
    return _M_SPEC_MAX["value"]


def _record_m_spec_norm(M: Tensor) -> None:
    try:
        v = torch.linalg.matrix_norm(M, ord=2).item()
    except Exception:
        return
    if v > _M_SPEC_MAX["value"]:
        _M_SPEC_MAX["value"] = v


def general_estimate_m_matrix_mcar(
    moment_dict: Dict[str, List[Tensor] | Tensor],
    alphas: Tensor,  # expects shape (3,)
) -> Tensor:
    """
    Estimate M^{(1)} under the MCAR assumption.

    Formula
    -------
        M = - Cov(ψ₁, φ₁)  @  [ Σⱼ αⱼ E[φⱼ φⱼᵀ] ]^{-1}

    Parameters
    ----------
    moment_dict : dict
        {
          "Cov(psi1, phi_j)" : list[(d, d)]  # we use only index 0 (φ₁)
          "E[phi_j phi_j^T]" : list[(d, d)]  # length = 3
        }
        All entries are already `torch.Tensor`s.
    alphas : length-3 sequence
        Scalar weights (Python floats, ints, or 0-D Tensors).

    Returns
    -------
    Tensor
        (d, d) matrix M on the same device / dtype as the inputs.
    """
    # 1. Extract required tensors
    cov_psi1_phi1: Tensor = moment_dict["Cov(psi1, phi1)"]   # (d, d)
    E_phi_phiT_list: List[Tensor] = moment_dict["E[phi_j phi_j^T]"]
    # 1. Filter out any second-moment matrices that contain NaN values
    valid_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for alpha, E_phi in zip(alphas, E_phi_phiT_list):
        if torch.isnan(E_phi).any():
            # Skip this matrix if it has any NaNs
            # print(f"Skipping phi second-moment matrix with NaNs (shape {E_phi.shape})")
            continue
        else:
            # Convert alpha to a 0-D tensor with matching dtype and device
            alpha_tensor = torch.as_tensor(
                alpha,
                dtype=cov_psi1_phi1.dtype,
                device=cov_psi1_phi1.device
            )
            valid_pairs.append((alpha_tensor, E_phi))

    # 2. Ensure there is at least one valid matrix to sum
    if not valid_pairs:
        raise ValueError("All phi second-moment matrices contain NaNs; cannot compute weighted sum.")

    # 3. Compute the weighted sum of the valid matrices
    weighted_sum = sum(alpha * E_phi for alpha, E_phi in valid_pairs)
    # 3. Compute M   (torch.linalg.inv handles CPU / CUDA transparently)
    M = -cov_psi1_phi1 @ torch.linalg.inv(weighted_sum)  # (d, d)
    _record_m_spec_norm(M)
    return M

def general_estimate_m_matrix_mar(
    moment_dict: Dict[str, Tensor]
) -> Tensor:
    r"""
    Estimate M under MAR using the revised moments:

        M = - E[1{R=1} ψ₁ φ₁ᵀ / α₁(X, W1, W2)]
              · [E[Σ_{j=1}^3 1{R=j} φⱼ φⱼᵀ]]⁻¹

    Expects moment_dict to contain:
        "E_psi1_phi1_scaled" : (d, d)
        "E_phi_agg"         : (d, d)
    """
    # 1. Extract the scaled cross-moment and aggregated phi moment
    psi1_phi1 = moment_dict["E_psi1_phi1_scaled"]  # (d, d)
    phi_agg   = moment_dict["E_phi_agg"]           # (d, d)

    # 2. Compute M
    M = - psi1_phi1 @ torch.linalg.inv(phi_agg)      # (d, d)
    _record_m_spec_norm(M)
    return M

def general_estimate_variance_mcar(
    moment_dict: Dict[str, List[Tensor] | Tensor],
    alphas: Tensor,  # expects shape (3,)
) -> Tensor:
    """
    Estimate  Cov(θ̂)  for the MCAR efficient estimator.

    Formula
    -------
        Cov̂(θ̂) =  Ê[ψ₁ ψ₁ᵀ] / α₁
                   – Cov̂(ψ₁, φ̂₁) · (Σⱼ αⱼ Ê[φ̂ⱼ φ̂ⱼᵀ])⁻¹ · Cov̂(ψ₁, φ̂₁)ᵀ

    Parameters
    ----------
    moment_dict : dict
        Output of `general_estimate_moments_mcar_torch`, containing
          "E[psi1 psi1^T]"   : (d, d) Tensor
          "E[phi_j phi_j^T]" : list[Tensor]  length-3, each (d, d)
          "Cov(psi1, phi_j)" : list[Tensor]  length-3, each (d, d)
    alphas : length-3 sequence
        (alpha1, alpha2, alpha3) – Python scalars or 0-D Tensors.

    Returns
    -------
    Tensor
        (d, d) covariance estimate on the same device / dtype as the inputs.
    """
    # ---------------- 1) unpack moments ---------------------
    E_psi1_psi1T: Tensor        = moment_dict["E[psi1 psi1^T]"]      # (d, d)
    E_phi_phiT_list: List[Tensor] = moment_dict["E[phi_j phi_j^T]"]   # 3 × (d, d)
    cov_psi_phi1 = moment_dict["Cov(psi1, phi1)"]      # (d, d)

    # ---------------- 2) prepare alphas ---------------------
    # Cast each alpha to a 0-D tensor on the same device / dtype
    device, dtype = E_psi1_psi1T.device, E_psi1_psi1T.dtype
    alpha1, alpha2, alpha3 = [
        torch.as_tensor(a, device=device, dtype=dtype) for a in alphas
    ]

    alpha_list = [alpha1, alpha2, alpha3]

    # 3) filter out any φ second‑moment matrices containing NaNs
    valid_pairs: List[Tuple[Tensor, Tensor]] = []
    for alpha, E_phi in zip(alpha_list, E_phi_phiT_list):
        if torch.isnan(E_phi).any():
            # print(f"Skipping phi second-moment matrix with NaNs (shape {E_phi.shape})")
            continue
        else:
            valid_pairs.append((alpha, E_phi))

    if not valid_pairs:
        raise ValueError("All phi second-moment matrices contain NaNs; cannot compute weighted sum.")

    # 4) build Σ valid αⱼ E[φⱼ φⱼᵀ] and invert
    weighted_phi_cov = sum(alpha * E_phi for alpha, E_phi in valid_pairs)  # (d, d)
    inv_weighted_phi_cov = torch.linalg.inv(weighted_phi_cov)               # (d, d)


    # ---------------- 4) covariance estimator ---------------
    correction = cov_psi_phi1 @ inv_weighted_phi_cov @ cov_psi_phi1.T  # (d, d)
    cov_theta  = E_psi1_psi1T / alpha1 - correction                    # (d, d)
    # # *** DEBUG print ***
    # print("\ninv_weighted_phi_cov =\n", inv_weighted_phi_cov)
    # print("\ncov_psi_phi1 =\n", cov_psi_phi1)
    # print("\ncorrection =\n", correction)
    # print("\nE_psi1_psi1T / alpha1 =\n", E_psi1_psi1T / alpha1)
    # print("\ncov_theta =\n", cov_theta)
    torch.set_printoptions(precision=4, linewidth=120, sci_mode=False)

    def r(t, dec=4):
        factor = 10 ** dec
        return (t * factor).round() / factor
    return cov_theta


def general_estimate_variance_mar(
    moment_dict: Dict[str, Tensor]
) -> Tensor:
    r"""
    Estimate Cov(θ̂) under MAR using the revised moments:

        Cov̂(θ̂) = E[1{R=1} ψ₁ ψ₁ᵀ / α₁²(X, W1, W2)]
                  - E[1{R=1} ψ₁ φ₁ᵀ / α₁(X, W1, W2)]
                    · [E[Σ_{j=1}^3 1{R=j} φⱼ φⱼᵀ]]⁻¹
                    · E[1{R=1} φ₁ ψ₁ᵀ / α₁(X, W1, W2)]

    Expects moment_dict to contain:
        "E_psi1_scaled2"       : (d, d)
        "E_phi_agg"            : (d, d)
        "E_psi1_phi1_scaled"   : (d, d)
    """
    E_psi1_scaled2       = moment_dict["E_psi1_scaled2"]      # (d, d)
    E_phi_agg            = moment_dict["E_phi_agg"]           # (d, d)
    E_psi1_phi1_scaled   = moment_dict["E_psi1_phi1_scaled"]  # (d, d)

    inv_phi_agg = torch.linalg.inv(E_phi_agg)                  # (d, d)

    correction = (
        E_psi1_phi1_scaled
        @ inv_phi_agg
        @ E_psi1_phi1_scaled.T 
    )                                                           # (d, d)

    cov_theta = E_psi1_scaled2 - correction                     # (d, d)
    return cov_theta
# ----------------------------------------------------------------------
# 5) g(alpha1)  under budget constraint
# ----------------------------------------------------------------------
def general_get_trace_variance_function_alpha_mcar(
    moment_fn: Callable[[Tensor], Dict[str, List[Tensor] | Tensor]],
    tau: float,
    c: float,
    return_full: bool = False,
) -> Callable[[Tensor], Union[Tensor, Tensor]]:
    """
    Return a function g(alpha1) that produces either
        • Tr(  Cov̂(θ̂) )   (default) or
        •       Cov̂(θ̂)    (if return_full=True)
    under the budget constraint (2.3):

        alpha2 = tau - c * alpha1
        alpha3 = 1 + (c - 1) * alpha1 - tau

    Parameters
    ----------
    moment_fn : callable
        Output of `general_estimate_moments_function_mcar` (Torch version).
        It maps a length-3 alpha vector to the required moment dictionary.
    tau : float
        Constraint alpha1 + alpha2 = tau
    c : float
        Constraint parameter in Eq. (2.3)
    return_full : bool, default False
        If True, g(alpha1) returns the full covariance matrix; otherwise its trace.

    Returns
    -------
    g : callable
        g(alpha1) -> scalar (trace) or (d, d) Tensor, depending on `return_full`.
        Accepts a 0-D Tensor for alpha1.
    """

    def g(alpha1: Tensor) -> Tensor:
        """
        Evaluate the covariance (or its trace) at a given alpha1.
        """

        # -------- 1) build full alpha vector under constraint -------------
        alpha_vec = torch.stack([
            alpha1,
            tau - c * alpha1,
            1.0 + (c - 1.0) * alpha1 - tau
        ]).to(alpha1)  # Keep same device and dtype

        # -------- 2) retrieve empirical moments ---------------------------
        moment_dict = moment_fn(alpha_vec)
        E_psi1_psi1T = moment_dict["E[psi1 psi1^T]"]
        E_phi_phiT_lst = moment_dict["E[phi_j phi_j^T]"]
        Cov_psi_phi1 = moment_dict["Cov(psi1, phi_j)"][0]

        # -------- 3) compute covariance estimator -------------------------
        weighted_phi_cov = sum(a * E for a, E in zip(alpha_vec, E_phi_phiT_lst))
        inv_weighted = torch.linalg.inv(weighted_phi_cov)

        cov_theta = E_psi1_psi1T / alpha_vec[0] - Cov_psi_phi1 @ inv_weighted @ Cov_psi_phi1.T
        out = cov_theta if return_full else torch.trace(cov_theta)

        return out
    return g

def general_get_cov00_function_alpha_mcar(
    moment_fn: Callable[[Tensor], Dict[str, List[Tensor] | Tensor]],
    tau: float,
    c: float,
) -> Callable[[Tensor], Tensor]:
    """
    Return a function g(alpha1) that evaluates Cov(θ̂)[0, 0]
    under the budget constraint (2.3).

    This isolates the variance of the first component of the parameter vector.
    """

    def g(alpha1: Tensor) -> Tensor:
        # -------- 1) build full alpha vector -----------------------------
        alpha_vec = torch.stack([
            alpha1,
            tau - c * alpha1,
            1.0 + (c - 1.0) * alpha1 - tau
        ]).to(alpha1)

        # -------- 2) retrieve empirical moments --------------------------
        moment_dict = moment_fn(alpha_vec)
        E_psi1_psi1T = moment_dict["E[psi1 psi1^T]"]
        E_phi_phiT_lst = moment_dict["E[phi_j phi_j^T]"]
        Cov_psi_phi1 = moment_dict["Cov(psi1, phi_j)"][0]

        # -------- 3) compute covariance estimator ------------------------
        weighted_phi_cov = sum(a * E for a, E in zip(alpha_vec, E_phi_phiT_lst))
        inv_weighted = torch.linalg.inv(weighted_phi_cov)

        cov_theta = E_psi1_psi1T / alpha_vec[0] - Cov_psi_phi1 @ inv_weighted @ Cov_psi_phi1.T
        return cov_theta[0, 0]  # return variance of θ̂₀

    return g


def general_get_trace_variance_function_alpha_mar(
    moment_fn: Callable[[nn.Module], Dict[str, Union[Tensor, List[Tensor]]]],
    return_full: bool = False,
) -> Callable[[nn.Module], Union[Tensor, Tensor]]:
    """
    Return a function g(alpha_model) that computes either
      • Tr(Cov(θ̂))         (default), or
      • Cov(θ̂)            (if return_full=True)
    under the MCAR weighting and budget constraint baked into moment_fn.

    Args:
        moment_fn:   A callable mapping an nn.Module (alpha_model) to a dict with keys:
                        - "E[psi1 psi1^T / alpha1]":   Tensor (d, d)
                        - "E[alpha_j phi_j phi_j^T]":  List of three Tensors (d, d)
                        - "Cov(psi1, phi_j)":          List of three Tensors (d, d)
                        - "E[c alpha1 + alpha2]":      Tensor scalar (optional)
        return_full: If True, g returns the full covariance matrix; otherwise its trace.

    Returns:
        A callable g that takes alpha_model and returns either:
          - scalar Tensor = trace of Cov(θ̂), or
          - Tensor of shape (d, d) = full Cov(θ̂).
    """
    def g(alpha_model: nn.Module) -> Union[Tensor, Tensor]:
        # 1) Retrieve empirical moments for this alpha_model
        moments = moment_fn(alpha_model)
        E_psi1_psi1_over_alpha1 = moments["E[psi1 psi1^T / alpha1]"]
        E_alpha_phi_phiT_list  = moments["E[alpha_j phi_j phi_j^T]"]
        Cov_psi_phi1           = moments["Cov(psi1, phi_j)"][0]

        # 2) Build weighted phi covariance: sum_j E[alpha_j * phi_j phi_j^T]
        weighted_phi_cov = sum(E_alpha_phi_phiT_list)

        # 3) Invert the weighted phi covariance
        inv_weighted = torch.linalg.inv(weighted_phi_cov)

        # 4) Compute the sandwich covariance estimator:
        #    Cov(θ̂) = E[ψ₁ψ₁ᵀ / α₁] − Cov(ψ₁, φ₁) @ [E[α φφᵀ]]⁻¹ @ Cov(ψ₁, φ₁)ᵀ
        cov_theta = (
            E_psi1_psi1_over_alpha1
            - Cov_psi_phi1 @ inv_weighted @ Cov_psi_phi1.T
        )

        # 5) Return trace or full matrix
        return cov_theta if return_full else torch.trace(cov_theta)

    return g

def general_get_cov00_function_alpha_mar(
    moment_fn: Callable[[nn.Module], Dict[str, Union[Tensor, List[Tensor]]]]
) -> Callable[[nn.Module], Tensor]:
    """
    Return a function g(alpha_model) that computes the variance of the first
    component of theta_hat, i.e. Cov(theta_hat)[0,0], under the MCAR weighting.

    Args:
        moment_fn: A callable that takes an alpha_model and returns a dict with keys:
            - "E[psi1 psi1^T / alpha1]":   Tensor (d, d)
            - "E[alpha_j phi_j phi_j^T]":  List[3] of Tensors (d, d)
            - "Cov(psi1, phi_j)":          List[3] of Tensors (d, d)
            - "E[c alpha1 + alpha2]":      Tensor scalar (not used here)

    Returns:
        A function g that takes alpha_model: nn.Module and returns a scalar Tensor
        equal to Cov(theta_hat)[0,0].
    """
    def g(alpha_model: nn.Module) -> Tensor:
        # 1) Retrieve all moment matrices via the current alpha_model
        moments = moment_fn(alpha_model)
        E_psi1_psi1_over_alpha1 = moments["E[psi1 psi1^T / alpha1]"]
        E_alpha_phi_phiT_list  = moments["E[alpha_j phi_j phi_j^T]"]
        Cov_psi_phi1           = moments["Cov(psi1, phi_j)"][0]  # only j=1 for covariance term

        # 2) Build the weighted phi covariance: sum_j E[alpha_j * phi_j phi_j^T]
        weighted_phi_cov = sum(E_alpha_phi_phiT_list)  # shape (d, d)
        # 3) Invert the weighted phi covariance
        inv_weighted = torch.linalg.inv(weighted_phi_cov)

        # 4) Compute the sandwich form: 
        #    Cov(theta) = E[psi1 psi1^T / alpha1] - Cov(psi1, phi1) * inv(E[alpha phi phi^T]) * Cov(psi1, phi1)^T
        cov_theta = E_psi1_psi1_over_alpha1 - Cov_psi_phi1 @ inv_weighted @ Cov_psi_phi1.T

        # 5) Return the (0,0) entry as the variance of the first component
        return cov_theta[0, 0]

    return g
# ---------------------------------------------------------------------
# 6) Derivative-free: Golden-section search  (kept unchanged)
# ---------------------------------------------------------------------
def _golden_section(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Unimodal 1-D minimisation via golden-section."""
    phi = 0.5 * (3.0 - 5 ** 0.5)          # 1 − Φ
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1, f2 = f(x1), f(x2)

    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        if f1 > f2:                       # keep [lo, x2]
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = f(x2)
        else:                             # keep [x1, hi]
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = f(x1)

    return (lo + hi) / 2.0


# ---------------------------------------------------------------------
# 6) Gradient-based: Adam search (1-D, closed interval)
# ---------------------------------------------------------------------
def _adam_section(
    f: Callable[[Tensor], Tensor],
    lo: float,
    hi: float,
    *,
    lr: float = 0.05,
    iters: int = 200,
    device: Optional[torch.device] = None,
    scheduler_name: str = "none",          # {"cosine", "linear", "step", "none"}
    scheduler_kw: Optional[Dict] = None,   # extra args for the scheduler
    log_interval: int = 10,                # NEW: print every n iterations
) -> Tensor:
    """
    One-dimensional Adam optimisation with optional LR scheduling.

    We optimise an unconstrained scalar z and map it to
        α₁ = lo + sigmoid(z) · (hi − lo)  ∈ (lo, hi).

    Parameters
    ----------
    lr : float
        Initial learning rate for Adam.
    scheduler_name / scheduler_kw
        Choose and configure LR scheduler ("cosine", "linear", "step", "none").
    log_interval : int
        Print loss & current LR every *log_interval* iterations.
    Returns
    -------
    Tensor
        Optimised α₁ (shape: scalar tensor, detach()ed).
    """
    # ---------- init ----------
    device  = torch.device("cpu") if device is None else device
    z       = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    optim   = torch.optim.Adam([z], lr=lr)

    # ---------- scheduler ----------
    kw = {} if scheduler_kw is None else scheduler_kw
    name = scheduler_name.lower()
    if name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=iters, **kw)
    elif name == "linear":
        sched = torch.optim.lr_scheduler.LinearLR(
            optim,
            start_factor=kw.get("start_factor", 1.0),
            end_factor  =kw.get("end_factor",   0.0),
            total_iters =iters
        )
    elif name == "step":
        sched = torch.optim.lr_scheduler.StepLR(optim, **kw)   # e.g. {"step_size":50,"gamma":0.5}
    elif name == "none":
        sched = None
    else:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'")

    lo_t, hi_t = torch.tensor(lo, device=device), torch.tensor(hi, device=device)

    def _to_alpha(z_val: Tensor) -> Tensor:
        """Map ℝ → (lo, hi) via sigmoid."""
        return lo_t + torch.sigmoid(z_val) * (hi_t - lo_t)

    # ---------- optimisation loop ----------
    with torch.enable_grad():
        for t in range(1, iters + 1):
            optim.zero_grad(set_to_none=True)
            alpha1 = _to_alpha(z)
            loss   = f(alpha1)         # scalar tensor
            loss.backward()
            optim.step()
            if sched is not None:
                sched.step()

            # # ----- logging -----
            # if t == 1 or t % log_interval == 0 or t == iters:
            #     current_lr = optim.param_groups[0]["lr"]
            #     print(f"[iter {t:3d}/{iters}] lr={current_lr:.2e} | loss={loss.item():.6f}")

    return _to_alpha(z).detach()


def _aug_lagrange_section(
    f: Callable[[Tensor], Tensor],          #   objective f(alpha) -> Tensor(1,)
    lo: float,
    hi: float,
    *,
    # ----- AL parameters -----
    g: Callable[[Tensor], Tensor] | None = None,   # constraint g(alpha) <= 0; may be None
    tau: float = 0.0,                              # tau; when g is given the constraint is g(alpha) - tau
    lambda_init: float = 0.0,
    rho_init: float = 10.0,
    # ----- optimisation -----
    lr: float = 5e-3,
    iters: int = 500,
    scheduler_name: str = "linear",
    scheduler_kw: Optional[Dict] = None,
    log_interval: int = 20,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    1-D augmented-Lagrange minimiser with bound (lo, hi).

    If *g* is None we just minimise f inside (lo, hi).
    Otherwise we minimise f(α) subject to  g(α) ≤ τ.

    Returns
    -------
    Tensor  # scalar tensor (detached)
    """
    # -------- 0. helpers --------
    def _make_scheduler(opt, name: str, kw: dict):
        name = name.lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, **kw)
        elif name == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=kw.get("start_factor", 1.0),
                end_factor  =kw.get("end_factor",   0.0),
                total_iters =iters
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(opt, **kw)
        elif name == "none":
            return None
        raise ValueError(name)

    device  = torch.device("cpu") if device is None else device
    z       = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    opt     = torch.optim.Adam([z], lr=lr)
    sched   = _make_scheduler(opt, scheduler_name, scheduler_kw or {})

    lam, rho = lambda_init, rho_init
    lo_t, hi_t = torch.tensor(lo, device=device), torch.tensor(hi, device=device)

    def to_alpha(z_val: Tensor) -> Tensor:
        """ℝ → (lo, hi) via sigmoid mapping."""
        return lo_t + torch.sigmoid(z_val) * (hi_t - lo_t)

    for t in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)

        a = to_alpha(z)               # α ∈ (lo, hi)
        main_loss = f(a)              # scalar tensor

        if g is None:
            lagrangian = main_loss
            diff = torch.zeros_like(main_loss)
        else:
            diff = g(a) - tau         # constraint residual  (want ≤ 0)
            # Augmented L:  L = f + λ·diff + ½ρ·ReLU(diff)²
            proxy = torch.clamp(diff, min=0.0)       # only if violating
            lagrangian = main_loss + lam * proxy + 0.5 * rho * proxy.pow(2)

        lagrangian.backward()
        opt.step()
        if sched is not None:
            sched.step()

        # --- λ update ---
        if g is not None:
            lam = max(0.0, lam + rho * diff.item())

        # --- ρ adaptation (optional, simple) ---
        if g is not None and (t % 100 == 0):
            if diff.item() > 0.05:    # still bad → raise ρ
                rho *= 1.5
            else:                     # close enough → soften ρ
                rho *= 0.9

        # --- logging ---
        if t == 1 or t % log_interval == 0 or t == iters:
            lr_cur = opt.param_groups[0]["lr"]
            if g is None:
                print(f"[{t:3d}/{iters}] lr={lr_cur:.2e}  f={main_loss.item():.6f}")
            else:
                print(f"[{t:3d}/{iters}] lr={lr_cur:.2e}  f={main_loss.item():.6f} "
                      f"d={diff.item():.4f} λ={lam:.2f} ρ={rho:.2f}")

    return to_alpha(z).detach()

# ---------------------------------------------------------------------
# 6) Public API: search_alpha_mcar  (choose "golden" | "adam")
# ---------------------------------------------------------------------
def search_alpha_mcar(
    trace_funcs: Sequence[Callable[[Tensor], Tensor]],
    tau: float,
    c: float,
    *,
    eps: float = 1e-6,
    method: str = "aug",              # "golden" | "adam"
    device: torch.device | None = None,
) -> float:
    """
    Minimise the averaged trace of covariance matrices over alpha1.

    The feasible set for alpha1 is an interval [lo, hi] derived from
    the budget and linear constraints:
        alpha2 = tau − c * alpha1
        alpha3 = 1 + (c − 1) * alpha1 − tau
    """
    up1  = tau / c
    low1 = 0.0 if c == 1 else (tau - 1) / (c - 1)
    lo, hi = max(eps, low1), min(1.0 - eps, up1)

    # Callable that returns mean trace( Σ_hat ) for a given alpha1
    def mean_trace(a1: Tensor) -> Tensor:
        if not isinstance(a1, Tensor):
            a1 = torch.tensor(a1, dtype=torch.float32, device=device)

        val = sum(f(a1) for f in trace_funcs) / len(trace_funcs)
        return val
    if method == "adam":
        alpha_opt = _adam_section(
            mean_trace, lo, hi,
            lr=1e-1, iters=500,
            scheduler_name="linear",
            scheduler_kw={"start_factor":1.0, "end_factor":0.0},
            log_interval=10,
            device=device
        )
        return alpha_opt
    
    elif method == "aug":                # <-- NEW: 1-D augmented-Lagrange search
        # Only the box constraint (lo, hi) matters here, no extra inequality
        return _aug_lagrange_section(
            mean_trace, lo, hi,
            lr=5e-3, iters=500,          # optimisation hyper-params
            scheduler_name="linear",     # linear LR decay
            scheduler_kw={"start_factor": 1.0, "end_factor": 0.0},
            log_interval=10,             # print every 10 iterations
            device=device
        )
    elif method == "golden":
        # _golden_section expects a function that takes a float
        def mean_trace_float(a1_float: float) -> float:
            return float(mean_trace(torch.tensor(a1_float)).detach())
        return _golden_section(mean_trace_float, lo, hi)
    else:
        raise ValueError(f"unknown method '{method}', choose 'golden' or 'adam'")

def search_alpha_mcar_trace(
    trace_funcs: Sequence[Callable[[Tensor], Tensor]],
    tau: float,
    c: float,
    *,
    eps: float = 1e-6,
    method: str = "aug",              # "golden" | "adam"
    device: torch.device | None = None,
) -> float:
    """
    Minimise the averaged trace of covariance matrices over alpha1.

    The feasible set for alpha1 is an interval [lo, hi] derived from
    the budget and linear constraints:
        alpha2 = tau − c * alpha1
        alpha3 = 1 + (c − 1) * alpha1 − tau
    """
    up1  = tau / c
    low1 = 0.0 if c == 1 else (tau - 1) / (c - 1)
    lo, hi = max(eps, low1), min(1.0 - eps, up1)

    # Callable that returns mean trace( Σ_hat ) for a given alpha1
    def mean_trace(a1: Tensor) -> Tensor:
        if not isinstance(a1, Tensor):
            a1 = torch.tensor(a1, dtype=torch.float32, device=device)

        val = sum(f(a1) for f in trace_funcs) / len(trace_funcs)
        return val
    if method == "adam":
        alpha_opt = _adam_section(
            mean_trace, lo, hi,
            lr=5e-1, iters=800,
            scheduler_name="linear",
            scheduler_kw={"start_factor":1.0, "end_factor":0.01},
            log_interval=10,
            device=device
        )
        return alpha_opt
    
    elif method == "aug":                # <-- NEW: 1-D augmented-Lagrange search
        # Only the box constraint (lo, hi) matters here, no extra inequality
        return _aug_lagrange_section(
            mean_trace, lo, hi,
            lr=5e-3, iters=500,          # optimisation hyper-params
            scheduler_name="linear",     # linear LR decay
            scheduler_kw={"start_factor": 1.0, "end_factor": 0.0},
            log_interval=10,             # print every 10 iterations
            device=device
        )
    elif method == "golden":
        # _golden_section expects a function that takes a float
        def mean_trace_float(a1_float: float) -> float:
            return float(mean_trace(torch.tensor(a1_float)).detach())
        return _golden_section(mean_trace_float, lo, hi)
    else:
        raise ValueError(f"unknown method '{method}', choose 'golden' or 'adam'")
    

def train_alpha_model(
    alpha_model: nn.Module,
    loss_fn: Callable[[nn.Module], Tensor],
    alpha_epochs: int,
    alpha_lr: float,
) -> nn.Module:
    """
    Train the alpha_model to minimize the scalar loss returned by loss_fn.

    Args:
        alpha_model:    the model to be trained
        loss_fn:        a function that takes alpha_model and returns a scalar loss Tensor
        alpha_epochs:   number of training epochs
        alpha_lr:       learning rate for the optimizer

    Returns:
        The trained alpha_model in eval mode.
    """
    # set up Adam optimizer on model parameters
    optimizer = torch.optim.Adam(alpha_model.parameters(), lr=alpha_lr)
    alpha_model.train()  # switch to training mode

    for epoch in range(alpha_epochs):
        optimizer.zero_grad(set_to_none=True)     # clear previous gradients
        lossalpha = loss_fn(alpha_model)          # compute scalar loss from model
        lossalpha.backward()                      # backpropagate
        optimizer.step()                          # update parameters

        # optionally print progress every 10 epochs (and on the first epoch)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{alpha_epochs}] loss = {lossalpha.item():.6f}")

    alpha_model.eval()  # switch to evaluation mode
    return alpha_model

def train_alpha_with_lagrangian(
    alpha_model: nn.Module,
    moment_fn: Callable[[nn.Module], Dict[str, Union[Tensor, List[Tensor]]]],
    tau: float,
    lambda_init: float = 0.0,
    lr_alpha: float = 1e-4,
    lr_lambda: float = 1e-2,
    alpha_epochs: int = 200,
    log_interval: int = 10,
) -> nn.Module:
    """
    Lagrange‐multiplier method for training `alpha_model`:
      L(α, λ) = Cov00(α) + λ * (E[c*α1 + α2] − τ),
    where we update:
      α ← α − lr_alpha ∇_α L,
      λ ← max(0, λ + lr_lambda * (E[c*α1 + α2] − τ)).
    """
    # Optimizer for α
    opt_alpha = torch.optim.AdamW(alpha_model.parameters(), lr=lr_alpha)
    cov00_fn = general_get_cov00_function_alpha_mar(moment_fn)

    # Initialize multiplier λ (must remain ≥ 0)
    lambda_var = lambda_init

    for epoch in range(1, alpha_epochs + 1):
        # 1) Compute Cov00(α) and the constraint value g(α) = E[c*α1 + α2]
        cov00 = cov00_fn(alpha_model)                            # Cov00(α)
        cost  = moment_fn(alpha_model)["E[c alpha1 + alpha2]"]   # g(α)

        # 2) Form the Lagrangian L = Cov00 + λ * (g(α) − τ)
        violation = cost - tau
        lagrangian = cov00 + lambda_var * violation

        # 3) Gradient‐step on α to minimize L
        opt_alpha.zero_grad()
        lagrangian.backward()
        opt_alpha.step()

        # 4) Gradient‐ascent on λ (projected at zero)
        lambda_var = max(0.0, lambda_var + lr_lambda * violation.item())

        # 5) Logging every `log_interval` epochs
        if epoch == 1 or epoch % log_interval == 0:
            print(
                f"Epoch {epoch:3d} | Lagrangian={lagrangian.item():.4f} | "
                f"Cov00={cov00.item():.4f} | cost={cost.item():.4f} | "
                f"violation={max(0.0, violation.item()):.4f} | "
                f"lambda={lambda_var:.4f}"
            )

    return alpha_model

def train_alpha_aug_lagrange(
    alpha_model: nn.Module,
    moment_fn: Callable[[nn.Module], Dict[str, Union[Tensor, List[Tensor]]]],
    tau: float,
    lambda_init: float = 0.0,
    rho_init: float = 10.0,
    *,
    lr_alpha: float = 2e-3,
    alpha_epochs: int = 500,
    scheduler_name: str = "linear",       # choose LR scheduler
    scheduler_kw: Optional[dict] = None,  # optional kwargs for the scheduler
    log_interval: int = 50,
    clip_grad_norm: float = 1.0,
) -> nn.Module:
    """
    Augmented-Lagrangian training for ``alpha_model`` with an inequality constraint.

    Objective
    ---------
    Minimize:  L(α, λ) = Cov00(α) + λ · [g(α) − τ]_+ + (ρ / 2) · [g(α) − τ]_+^2
    where     g(α) = E[c·α₁ + α₂]  and  [x]_+ = max(x, 0).

    Per-epoch updates
    -----------------
    • α  ←  α − lr_alpha · ∇_α L                  (descent on α)
    • λ  ←  max(0, λ + η_λ · ρ · [g(α) − τ]_+)    (projected ascent, only if violated)
    • ρ  is adapted: relax in feasible region, tighten if persistently infeasible.

    Parameters
    ----------
    alpha_model : nn.Module
        Model that outputs α-parameters.
    moment_fn : callable
        Function returning required moments given the current model.
        Must include key "E[c alpha1 + alpha2]".
    tau : float
        Feasibility threshold for g(α).
    lambda_init : float, default 0.0
        Initial value for the Lagrange multiplier λ (nonnegative).
    rho_init : float, default 10.0
        Initial penalty coefficient ρ (>0).
    lr_alpha : float, default 2e-3
        Learning rate for α parameters.
    alpha_epochs : int, default 500
        Number of training epochs.
    scheduler_name : {"linear","cosine","step","exp","none"}, default "linear"
        Learning-rate scheduler type.
    scheduler_kw : dict or None
        Extra kwargs for the scheduler (e.g., {"eta_min": 1e-5} for cosine).
    log_interval : int, default 50
        Print metrics every *log_interval* epochs (block is commented below).
    clip_grad_norm : float or None, default 1.0
        Gradient-norm clipping threshold; set None to disable.

    Returns
    -------
    nn.Module
        The trained ``alpha_model`` (returned in eval mode).
    """
    alpha_model.train()

    # ---------- optimizer ----------
    opt = torch.optim.AdamW(alpha_model.parameters(), lr=lr_alpha)

    # ---------- scheduler ----------
    kw = {} if scheduler_kw is None else dict(scheduler_kw)
    name = scheduler_name.lower()

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=alpha_epochs, **kw
        )
    elif name == "step":
        # e.g. {"step_size": 100, "gamma": 0.5}
        scheduler = torch.optim.lr_scheduler.StepLR(opt, **kw)
    elif name == "exp":
        # e.g. {"gamma": 0.98}
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, **kw)
    elif name == "linear":
        # {"start_factor": 1.0, "end_factor": 0.0} over alpha_epochs steps
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=kw.get("start_factor", 1.0),
            end_factor=kw.get("end_factor", 0.0),
            total_iters=alpha_epochs,
        )
    elif name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'")

    # Convenience helpers from your codebase
    cov00_fn = general_get_cov00_function_alpha_mar(moment_fn)

    # Dual/penalty variables
    lam: float = max(0.0, float(lambda_init))
    rho: float = max(1e-8, float(rho_init))
    lam_lr: float = 0.5     # damping for lambda updates (0,1]
    eps: float = 0.0        # optional slack: penalize only when g > tau + eps
    tol: float = 1e-3       # treat <= tol as feasible (for rho adaptation)
    rho_min, rho_max = 1e-4, 1e6

    for epoch in range(1, alpha_epochs + 1):
        # ---- forward ----
        cov00 = cov00_fn(alpha_model)                                # scalar tensor
        cost  = moment_fn(alpha_model)["E[c alpha1 + alpha2]"]       # scalar tensor
        diff  = cost - tau
        viol  = torch.relu(diff - eps)                               # [g - tau - eps]_+

        # Augmented Lagrangian with squared hinge -> no gradient when feasible
        lagrangian = cov00 + lam * viol + 0.5 * rho * viol.pow(2)

        # ---- backward / optimizer ----
        opt.zero_grad(set_to_none=True)
        lagrangian.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(alpha_model.parameters(), clip_grad_norm)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        # ---- lambda update (only when constraint is violated) ----
        lam = max(0.0, lam + lam_lr * rho * float(viol.item()))

        # ---- rho adaptation: relax when feasible; tighten if persistently infeasible ----
        if float(viol.item()) <= tol:
            rho = max(rho_min, rho * 0.8)       # relax penalty
        else:
            if epoch % 20 == 0:                 # tighten slowly to avoid oscillation
                rho = min(rho_max, rho * 1.7)

        # # ---- logging (optional) ----
        # if epoch == 1 or epoch % log_interval == 0:
        #     current_lr = opt.param_groups[0]["lr"]
        #     print(
        #         f"Ep {epoch:4d} | lr={current_lr:.2e} | "
        #         f"L={lagrangian.item():.4f} | Cov00={cov00.item():.4f} | "
        #         f"cost={cost.item():.4f} | viol={viol.item():.4f} | "
        #         f"lambda={lam:.3f} | rho={rho:.3f}"
        #     )

    alpha_model.eval()
    return alpha_model

def train_alpha_with_penalty(
    alpha_model: nn.Module,
    moment_fn: Callable[[nn.Module], Dict[str, Union[Tensor, List[Tensor]]]],
    tau: float,
    lambda_pen: float = 200,
    lr_alpha: float = 1e-4,
    alpha_epochs: int = 500,
    log_interval: int = 50,
) -> nn.Module:
    """
    Penalty method for training alpha_model with a linear penalty:
      loss = Cov00(α) + λ_pen * max(0, E[c*α1 + α2] - τ)
    """
    # 1) set up optimizer and covariance function
    opt = torch.optim.AdamW(alpha_model.parameters(), lr=lr_alpha)
    cov00_fn = general_get_cov00_function_alpha_mar(moment_fn)

    for epoch in range(1, alpha_epochs + 1):
        # 2) compute Cov00(α) and the constraint expectation g(α)
        cov00 = cov00_fn(alpha_model)                            # Cov00
        cost  = moment_fn(alpha_model)["E[c alpha1 + alpha2]"]   # g(α)

        # 3) linear penalty: only penalize if cost exceeds tau
        violation = torch.clamp(cost - tau, min=0.0)
        loss = cov00 + lambda_pen * violation

        # 4) gradient update step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # # 5) logging every `log_interval` epochs
        # if epoch == 1 or epoch % log_interval == 0:
        #     print(
        #         f"Epoch {epoch:3d} | loss={loss.item():.4f} | "
        #         f"Cov00={cov00.item():.4f} | cost={cost.item():.4f} | "
        #         f"violation={violation.item():.4f}"
        #     )

    return alpha_model


def train_alpha_aug_lagrange_trace(
    alpha_model: nn.Module,
    moment_fn: Callable[[nn.Module], Dict[str, 'Tensor | List[Tensor]']],
    tau: float,
    lambda_init: float = 0.0,
    rho_init: float = 10.0,
    *,
    lr_alpha: float = 2e-3,
    alpha_epochs: int = 500,
    scheduler_name: str = "linear",
    scheduler_kw: Optional[dict] = None,
    log_interval: int = 50,
    clip_grad_norm: float = 1.0,
) -> nn.Module:
    """
    Augmented-Lagrangian training specialized for minimizing Tr(Cov(θ̂)) under
    the inequality constraint g(α) = E[c·α₁ + α₂] <= τ.

    Objective:
        J(α) = Tr(Cov(θ̂)(α))
    Lagrangian:
        L(α, λ) = J(α) + λ·[g(α)−τ−eps]_+ + (ρ/2)·[g(α)−τ−eps]_+^2
    """
    alpha_model.train()

    # ----- optimizer -----
    opt = torch.optim.AdamW(alpha_model.parameters(), lr=lr_alpha)

    # ----- scheduler -----
    kw = {} if scheduler_kw is None else dict(scheduler_kw)
    name = scheduler_name.lower()
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=alpha_epochs, **kw)
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, **kw)          # e.g. {"step_size": 100, "gamma": 0.5}
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, **kw)   # e.g. {"gamma": 0.98}
    elif name == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=kw.get("start_factor", 1.0),
            end_factor=kw.get("end_factor", 0.0),
            total_iters=alpha_epochs,
        )
    elif name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'")

    # ----- trace objective -----
    trace_fn = general_get_trace_variance_function_alpha_mar(moment_fn, return_full=False)

    # ----- dual/penalty vars -----
    lam: float = max(0.0, float(lambda_init))
    rho: float = max(1e-8, float(rho_init))
    lam_lr: float = 1.0       # was 0.5 → faster dual response
    eps: float = 1e-3         # was 0.0 → ignore tiny numeric violations
    tol: float = 1e-3
    rho_min, rho_max = 1e-4, 1e6

    for epoch in range(1, alpha_epochs + 1):
        # forward
        obj  = trace_fn(alpha_model)                                      # scalar tensor: Tr(Cov)
        cost = moment_fn(alpha_model)["E[c alpha1 + alpha2]"]             # scalar tensor
        viol = torch.relu(cost - tau - eps)                               # [g - tau - eps]_+

        lagrangian = obj + lam * viol + 0.5 * rho * viol.pow(2)

        # backward/step
        opt.zero_grad(set_to_none=True)
        lagrangian.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(alpha_model.parameters(), clip_grad_norm)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        # lambda update (projected)
        lam = max(0.0, lam + lam_lr * rho * float(viol.item()))

        # rho adaptation (slightly stronger/faster)
        if float(viol.item()) <= tol:
            rho = max(rho_min, rho * 0.7)           # relax a bit faster
        else:
            if epoch % 10 == 0:                     # tighten a bit sooner
                rho = min(rho_max, rho * 2.0)

        # # optional logging
        # if epoch == 1 or epoch % log_interval == 0:
        #     lr_now = opt.param_groups[0]["lr"]
        #     print(
        #         f"Ep {epoch:4d} | lr={lr_now:.2e} | "
        #         f"L={lagrangian.item():.4f} | Tr(Cov)={obj.item():.4f} | "
        #         f"cost={cost.item():.4f} | viol={viol.item():.4f} | "
        #         f"lambda={lam:.3f} | rho={rho:.3f}"
        #     )

    alpha_model.eval()
    return alpha_model
# ----------------------------------------------------------------------
def lm_mono_debias_estimate(
    X: Tensor,                   # (n, d)
    Y: Tensor,                   # (n,)
    W1: Tensor,                  # (n,)
    W2: Tensor,                  # (n,)
    V: Tensor,                   # (n,)
    R: Tensor,                   # (n,)  int64 values in {1,2,3}
    phi_list: List[Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]],
    M_hat: Tensor,               # (d, d)
    w_init: Optional[Tensor] = None  # (n,) or None
) -> Tensor:
    """
    Efficient estimator θ̂^(1) under MCAR (mono-debiasing), Torch version.

    Each φ_j in `phi_list` must accept (X_batch, W1_batch, W2_batch, V_batch)
    and return a (batch_size, d) tensor on the same device / dtype.
    """
    n, d = X.shape
    device, dtype = X.device, X.dtype

    # ------------------------------------------------------------------
    # Step 1: initial estimator θ̃ via (weighted) least squares
    # ------------------------------------------------------------------
    Y_flat = Y.view(-1)
    mask_obs = ~torch.isnan(Y_flat)                 # (n,)
    X_obs = X[mask_obs]                             # (n_obs, d)
    Y_obs = Y_flat[mask_obs]                        # (n_obs,)

    if w_init is not None:
        w_obs = w_init[mask_obs]                    # (n_obs,)
        sqrt_w = torch.sqrt(w_obs).unsqueeze(1)     # (n_obs, 1)
        Xw = X_obs * sqrt_w                         # weighted X
        Yw = Y_obs * sqrt_w.squeeze()               # (n_obs,)
    else:
        Xw, Yw = X_obs, Y_obs

    theta_tilde = torch.linalg.lstsq(Xw, Yw).solution.view(-1)  # (d,)

    # ------------------------------------------------------------------
    # Step 2: correction term  (1/n) Σ_i φ_{r_i}
    # ------------------------------------------------------------------
    mask_r1 = (R == 1).view(-1)
    mask_r2 = (R == 2).view(-1)
    mask_r3 = (R == 3).view(-1)

    phi1_vals = phi_list[0](X[mask_r1], W1[mask_r1], W2[mask_r1], V[mask_r1]) if mask_r1.any() else torch.zeros((0, d), device=device)
    phi2_vals = phi_list[1](X[mask_r2], W1[mask_r2], W2[mask_r2], V[mask_r2]) if mask_r2.any() else torch.zeros((0, d), device=device)
    phi3_vals = phi_list[2](X[mask_r3], W1[mask_r3], W2[mask_r3], V[mask_r3]) if mask_r3.any() else torch.zeros((0, d), device=device)
    correction = torch.cat([phi1_vals, phi2_vals, phi3_vals], dim=0).mean(dim=0)  # (d,)

    # ------------------------------------------------------------------
    # Step 3: final debiased estimator
    # ------------------------------------------------------------------
    theta_hat = theta_tilde + M_hat @ correction  # (d,)

    return theta_hat
