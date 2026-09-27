"""
mono_bias.py
============
High-level MCAR mono-debias pipeline

Public API
----------
1) lm_ols_estimate                                – OLS baseline (Torch)
2) lm_mono_debias_estimate_mcar_crossfit          – 3-fold efficient θ̂ (given alpha)
3) lm_mono_debias_budget_constrained_obtain_alpha_mcar
                                                 – 2-fold search for alpha* (Alg-4)
4) lm_mcar                                        – Full Alg-1 (Stage-1 + Stage-2)
"""

from __future__ import annotations
from typing import Callable, Literal, Dict, List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm
import math

from typeguard import value

from .utils import (
    get_device, set_global_seed, sample_split, wald_ci         # <-- our helpers
)
from .models import AlphaModel
from .data_generation import lm_generate_obs_data_mcar, lm_generate_obs_data_mar, lm_generate_complete_data, general_generate_mcar, general_generate_mar
from .estimators import (
    lm_fit_ols,                                    # OLS regression
    lm_fit_wls,                                    # WLS regression
    lm_build_all_psi,                             # compute ψ-values (no grads)
    lm_build_all_psi_weighted,                     # compute ψ-values (with weights)
    general_build_all_phi,                        # evaluate φ on a fixed alpha
    general_build_all_phi_mar,                    # evaluate φ on a fixed alpha (MAR)
    general_build_all_phi_function,               # return callables φ1/φ2/φ3
    general_build_all_phi_function_mar,           # return callables φ1/φ2/φ3 (MAR)
    general_estimate_moments_mcar,                # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j)
    general_estimate_moments_mar,                 # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j) (MAR)
    general_estimate_moments_function_mcar,       # return moment_fn(alpha_vec)
    general_estimate_moments_function_mar,        # return moment_fn(alpha_vec) (MAR)
    general_estimate_m_matrix_mcar,               # estimate M^(1)
    general_estimate_m_matrix_mar,                # estimate M^(1) (MAR)
    general_estimate_variance_mcar,               # estimate Cov(θ̂)
    general_estimate_variance_mar,                # estimate Cov(θ̂) (MAR)
    general_get_trace_variance_function_alpha_mcar,# return g(alpha1)
    general_get_cov00_function_alpha_mcar,       # return g00(alpha1)
    general_get_trace_variance_function_alpha_mar,# return g(alpha1) (MAR)
    general_get_cov00_function_alpha_mar,        # return g00(alpha1) (MAR)
    search_alpha_mcar,                            # public API: "golden" | "adam"
    search_alpha_mcar_trace,                            # public API: "golden" | "adam"
    train_alpha_with_lagrangian,                # train a neural network to predict α₁
    train_alpha_aug_lagrange,                   # train a neural network to predict α₁
    train_alpha_with_penalty,                # train a neural network to predict α₁
    train_alpha_aug_lagrange_trace,          # train a neural network to predict α₁
    lm_mono_debias_estimate,                 # 3-fold cross-fit efficient θ̂
    reset_m_spec_norm,                       # diagnostic: reset max ||M||_sp
    get_m_spec_norm,                         # diagnostic: read  max ||M||_sp
)

# Relative tolerance on the expected acquisition cost. The augmented-Lagrangian solve
# enforces E[c*alpha1 + alpha2] <= tau only approximately, so a small positive excess is
# numerical noise rather than an infeasible design.
BUDGET_RTOL = 1e-2


# ======================================================================
# 1) 3-fold cross-fit efficient estimator  (Algorithm 3)
# ======================================================================
@torch.no_grad()  # cross-fit is purely empirical – gradients not needed
def lm_mono_debias_estimate_mcar_crossfit(
    X: Tensor, Y: Tensor,
    W1: Tensor, W2: Tensor, V: Tensor,
    R: Tensor,                      # (n,) int64 values {1,2,3}
    alpha: Tensor,
    method: str = "mlp"
) -> Tuple[Tensor, Tensor]:
    """
    End-to-end 3-fold cross-fit pipeline (Steps 1–6) – Torch version.

    Returns
    -------
    theta_final : (d,)  Tensor   — cross-fitted efficient estimator
    cov_final   : (d,d) Tensor   — cross-fitted covariance estimate
    """

    # ---------- Step 1 : deterministic 3-way split ----------
    device, dtype = X.device, X.dtype
    n, d = X.shape
    idx1, idx2, idx3 = sample_split(n, 3, device=device)        # assumes Torch-compatible
    folds = [
        (idx1, idx2, idx3),   # rotation 1  (D1, D2, D3)
        (idx2, idx3, idx1),   # rotation 2
        (idx3, idx1, idx2)    # rotation 3
    ]

    def subset(idxs: Tensor):
        """Return tensors restricted to `idxs` (1-D LongTensor)."""
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    theta_list: List[Tensor] = []
    cov_list:   List[Tensor] = []
    size_list:  List[int]    = []

    # ---------- 3-fold rotation loop ----------
    for D1, D2, D3 in folds:
        # Convert index arrays to Torch tensors for slicing
        D1_t = torch.as_tensor(D1, device=device, dtype=torch.long)
        D2_t = torch.as_tensor(D2, device=device, dtype=torch.long)
        D3_t = torch.as_tensor(D3, device=device, dtype=torch.long)

        # Fetch data subsets
        X1, Y1, W11, W21, V1, R1 = subset(D1_t)
        X2, Y2, W12, W22, V2, R2 = subset(D2_t)
        X3, Y3, W13, W23, V3, R3 = subset(D3_t)

        # ----- Step 2 : build ψ and φ on D3 ∩ {R = 1} -----
        mask_r1_D3 = (R3 == 1).squeeze(1)
        # print(alpha)
        theta_pre = lm_fit_ols(
            X3[mask_r1_D3],
            Y3[mask_r1_D3])
        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X3[mask_r1_D3], Y3[mask_r1_D3],
            W13[mask_r1_D3], W23[mask_r1_D3], V3[mask_r1_D3],
            theta_pre, method=method)

        phi_1, phi_2, phi_3 = general_build_all_phi(
            psi_2, psi_3,
            alpha)

        # ----- Step 3 : estimate covariance matrices on D2
        moments = general_estimate_moments_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X2, Y2,
            W12, W22, V2,
            R2
        )
        # Estimate M matrix
        M_hat = general_estimate_m_matrix_mcar(moments, alpha)

        # ----- Step 4 : debias on D1 -----
        theta_k = lm_mono_debias_estimate(
            X1, Y1, W11, W21, V1, R1,
            [phi_1, phi_2, phi_3], M_hat)
        cov_k = general_estimate_variance_mcar(moments, alpha)

        theta_list.append(theta_k)
        cov_list.append(cov_k)
        size_list.append(len(D1))

    # ---------- Step 6 : weighted average ----------
    weights = torch.tensor(size_list, device=device, dtype=dtype) / n
    theta_final = sum(w * th for w, th in zip(weights, theta_list))      # (d,)
    cov_final   = sum(w * cv for w, cv in zip(weights, cov_list))        # (d,d)

    return theta_final, cov_final


# ======================================================================
# 2) 2-fold search for alpha*  (Algorithm 2)
# ======================================================================


def lm_mono_debias_budget_constrained_obtain_alpha_mcar_trace(
    X: Tensor, Y: Tensor,
    W1: Tensor, W2: Tensor, V: Tensor,
    R: Tensor,                    # (n,) int64 in {1,2,3}
    tau: float,
    c: float,
    method: str = "mlp",
    eps: float = 1e-6
) -> Tuple[Tensor, float, Tensor]:
    """
    Two-fold cross-fitting (Alg. 2, MCAR) using your Adam optimiser
    `search_alpha_mcar(..., method="adam")` to find the optimal α₁.
    Returns α* = (α₁, α₂, α₃) that minimises Tr(Cov̂(θ̂)).
    """
    # ---------- helper: subset by indices -----------------
    def subset(idxs: Tensor):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    # ---------- deterministic 50/50 split -----------------
    n = X.shape[0]
    device, dtype = X.device, X.dtype
    idx_fold = sample_split(n, 2, device=device)           # returns [idxA, idxB]
    folds = [(idx_fold[0], idx_fold[1]), (idx_fold[1], idx_fold[0])]

    trace_funcs: List = []
    cov_funcs:   List = []

    # ============== two-fold cross-fit loop ===============
    for D_moment, D_build in folds:
        # ----- data partitions ----------------------------
        X1, Y1, W11, W21, V1, R1 = subset(D_moment)   # moment set
        X2, Y2, W12, W22, V2, R2 = subset(D_build)    # ψ/φ set
        # ----- build ψ / φ on D_build ∩ {R=1} ------------
        mask2 = (R2 == 1).squeeze(1)
        theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X2[mask2], Y2[mask2],
            W12[mask2], W22[mask2], V2[mask2],
            theta_pre, method=method
        )

        phi_1, phi_2, phi_3 = general_build_all_phi_function(psi_2, psi_3)

        # ----- moment function on D_moment ∩ {R=1} -------
        mask1 = (R1 == 1).squeeze(1)
        moment_fn = general_estimate_moments_function_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X1[mask1], Y1[mask1],
            W11[mask1], W21[mask1], V1[mask1]
        )

        # ----- trace / covariance evaluators --------------
        trace_f = general_get_trace_variance_function_alpha_mcar(
            moment_fn, tau, c, return_full=False)
        cov_f   = general_get_trace_variance_function_alpha_mcar(
            moment_fn, tau, c, return_full=True)
        trace_funcs.append(trace_f)
        cov_funcs.append(cov_f)

    # ---------- optimise α₁ with your Adam search ---------
    alpha1_opt = search_alpha_mcar_trace(
        trace_funcs, tau, c,
        eps=eps,
        method="adam",
        device=device
    )
    

    # ---------- construct full α vector ------------------
    alpha2_opt = tau - c * alpha1_opt
    alpha3_opt = 1.0 + (c - 1.0) * alpha1_opt - tau
    alpha_opt = torch.tensor(
        [alpha1_opt, alpha2_opt, alpha3_opt],
        device=device, dtype=dtype
    )                               # (3,)

    # ---------- final covariance & trace -----------------
    cov_opt = sum(f(alpha1_opt) for f in cov_funcs) / len(cov_funcs)
    trace_opt = float(torch.trace(cov_opt).item())

    return alpha_opt, trace_opt, cov_opt

def lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
    X: Tensor, Y: Tensor,
    W1: Tensor, W2: Tensor, V: Tensor,
    R: Tensor,                    # (n,) int64 in {1,2,3}
    tau: float,
    c: float,
    method: str = "mlp",
    eps: float = 1e-6
) -> Tuple[Tensor, float, Tensor]:
    """
    Two-fold cross-fitting (Alg. 2, MCAR) using your Adam optimiser
    `search_alpha_mcar(..., method="adam")` to find the optimal α₁.
    Returns α* = (α₁, α₂, α₃) that minimises Tr(Cov̂(θ̂)).
    """
    # ---------- helper: subset by indices -----------------
    def subset(idxs: Tensor):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    # ---------- deterministic 50/50 split -----------------
    n = X.shape[0]
    device, dtype = X.device, X.dtype
    idx_fold = sample_split(n, 2, device=device)           # returns [idxA, idxB]
    folds = [(idx_fold[0], idx_fold[1]), (idx_fold[1], idx_fold[0])]

    cov00_funcs: List = []

    # ============== two-fold cross-fit loop ===============
    for D_moment, D_build in folds:
        # ----- data partitions ----------------------------
        X1, Y1, W11, W21, V1, R1 = subset(D_moment)   # moment set
        X2, Y2, W12, W22, V2, R2 = subset(D_build)    # ψ/φ set
        # ----- build ψ / φ on D_build ∩ {R=1} ------------
        mask2 = (R2 == 1).squeeze(1)
        theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X2[mask2], Y2[mask2],
            W12[mask2], W22[mask2], V2[mask2],
            theta_pre, method=method
        )

        phi_1, phi_2, phi_3 = general_build_all_phi_function(psi_2, psi_3)

        # ----- moment function on D_moment ∩ {R=1} -------
        mask1 = (R1 == 1).squeeze(1)
        moment_fn = general_estimate_moments_function_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X1[mask1], Y1[mask1],
            W11[mask1], W21[mask1], V1[mask1]
        )

        # ----- trace / covariance evaluators --------------
        cov00_f = general_get_cov00_function_alpha_mcar(moment_fn, tau, c)
        cov00_funcs.append(cov00_f)

    # ---------- optimise α₁ with your Adam search ---------
    alpha1_opt = search_alpha_mcar(
        cov00_funcs, tau, c,
        eps=eps,
        method="adam",
        device=device
    )
    

    # ---------- construct full α vector ------------------
    alpha2_opt = tau - c * alpha1_opt
    alpha3_opt = 1.0 + (c - 1.0) * alpha1_opt - tau
    alpha_opt = torch.tensor(
        [alpha1_opt, alpha2_opt, alpha3_opt],
        device=device, dtype=dtype
    )                               # (3,)

    # ---------- final covariance & trace -----------------
    cov_f = general_get_trace_variance_function_alpha_mcar(
        moment_fn, tau, c, return_full=True
    )
    cov_opt = cov_f(alpha1_opt)
    cov00 = float(cov_opt[0, 0].item())

    # ======================================================
    # >>> baseline: α₂ = 0  →  α₁ = τ / c, α₃ from budget
    # ======================================================
    # alpha1_base = tau / c
    # # ensure α₁ lies in (eps, 1−eps); clip if needed
    # alpha1_base = max(eps, min(1.0 - eps, alpha1_base))
    # alpha2_base = 0.0
    # alpha3_base = 1.0 + (c - 1.0) * alpha1_base - tau
    # cov_base    = cov_f(torch.tensor(alpha1_base, device=device))
    # cov00_base  = float(cov_base[0, 0].item())

    # print(f"[MCAR]  Cov00(opt)   = {cov00:8.3f}")
    # print(f"[MCAR]  Cov00(base)  = {cov00_base:8.3f}  (α2 = 0 baseline)")

    return alpha_opt, cov00, cov_opt




# ======================================================================
# 5) 3-fold cross-fit efficient estimator for MAR (Algorithm 5)
# ======================================================================

@torch.no_grad()
def lm_mono_debias_estimate_mar_crossfit(
    X: Tensor, Y: Tensor,
    W1: Tensor, W2: Tensor, V: Tensor,
    R: Tensor,                                          # (n,) in {1,2,3}
    alpha_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    method: str = "mlp"
) -> Tuple[Tensor, Tensor]:

    # ----- Step 0 : pre-compute α(x) for every row --------------------------
    # α_all : (n, 3)   α1_all : (n,)  == P(R=1 | X,W1,W2)
    alpha_all = alpha_fn(X, W1, W2)          # single forward pass
    alpha1_all = alpha_all[:, 0]             # keep only the pattern-1 prob

    # ---------- Step 1 : deterministic 3-way split -------------------------
    device, dtype = X.device, X.dtype
    n, d = X.shape
    idx1, idx2, idx3 = sample_split(n, 3, device=device)
    folds = [(idx1, idx2, idx3), (idx2, idx3, idx1), (idx3, idx1, idx2)]

    def subset(idxs: Tensor):
        """Return data restricted to `idxs` (1-D LongTensor)."""
        return (X[idxs], Y[idxs], W1[idxs], W2[idxs],
                V[idxs], R[idxs], alpha1_all[idxs])

    theta_list, cov_list, size_list = [], [], []

    # ---------- 3-fold rotation loop --------------------------------------
    for D1, D2, D3 in folds:
        D1_t = torch.as_tensor(D1, device=device, dtype=torch.long)
        D2_t = torch.as_tensor(D2, device=device, dtype=torch.long)
        D3_t = torch.as_tensor(D3, device=device, dtype=torch.long)

        # Fetch data subsets (+ α₁)
        X1, Y1, W11, W21, V1, R1, a1_1 = subset(D1_t)
        X2, Y2, W12, W22, V2, R2, a1_2 = subset(D2_t)
        X3, Y3, W13, W23, V3, R3, a1_3 = subset(D3_t)

        # ---------- Step 2 : ψ / φ on D3 ∩ {R=1} --------------------------
        mask_r1_D3 = (R3 == 1).squeeze(1)
        # Weighted least-squares using inverse-probability weights 1/α₁
        theta_pre = lm_fit_wls(
            X3[mask_r1_D3],
            Y3[mask_r1_D3],
            w=1.0 / a1_3[mask_r1_D3]
        )

        psi_1, psi_2, psi_3 = lm_build_all_psi_weighted(
            X3[mask_r1_D3], Y3[mask_r1_D3],
            W13[mask_r1_D3], W23[mask_r1_D3], V3[mask_r1_D3],
            theta_pre, w=mask_r1_D3.float().mean() / a1_3[mask_r1_D3], method=method
        )

        phi_1, phi_2, phi_3 = general_build_all_phi_mar(
            psi_2, psi_3,
            alpha_fn                     # ← callable, not tensor
        )

        # ---------- Step 3 : moment & M-matrix on D2-------------
        moments = general_estimate_moments_mar(
            psi_1, phi_1, phi_2, phi_3,
            X2, Y2, W12, W22, V2, R2, a1_2
        )
        M_hat = general_estimate_m_matrix_mar(moments)

        # ---------- Step 4 : debias on D1 --------------------------------
        theta_k = lm_mono_debias_estimate(
            X1, Y1, W11, W21, V1, R1,
            [phi_1, phi_2, phi_3], M_hat,
            w_init= 1.0 / a1_1
        )
        cov_k = general_estimate_variance_mar(moments)

        theta_list.append(theta_k)
        cov_list.append(cov_k)
        size_list.append(len(D1))

    # ---------- Step 6 : weighted average ---------------------------------
    weights = torch.tensor(size_list, device=device, dtype=dtype) / n
    theta_final = sum(w * th for w, th in zip(weights, theta_list))
    cov_final   = sum(w * cv for w, cv in zip(weights, cov_list))

    return theta_final, cov_final




# ======================================================================
# 2) 2-fold search for alpha* for MAR (Algorithm 4)
# ======================================================================

def lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00(
    X:  Tensor,
    Y:  Tensor,
    W1: Tensor,
    W2: Tensor,
    V:  Tensor,
    R:  Tensor,                # (n,) int64 in {1,2,3}
    tau: float,
    c:  float,
    method: str = "mlp",
    alpha_hidden_dim: int = 32,
    alpha_epochs: int = 600,
    alpha_lr: float = 0.01,
    lambda_lr: float = 170,
) -> Tuple[nn.Module, float, Tensor]:
    """
    Single‐split MAR debiasing with budget constraint enforced via
    a Lagrange multiplier (primal–dual) optimization.

    Args:
        X, Y, W1, W2, V:  data tensors of shape (n, ·)
        R:                missingness indicator (n,) in {1,2,3}
        tau, c:           budget constraint parameters
        method:           "mlp" or other method for ψ/φ construction
        alpha_hidden_dim: hidden layer size in AlphaModel
        alpha_epochs:     number of epochs for α training
        alpha_lr:         learning rate for α
        lambda_lr:        learning rate for the Lagrange multiplier

    Returns:
        alpha_model: trained AlphaModel
        cov00:       scalar Var[θ̂₀]
        cov_full:    full covariance matrix Cov(θ̂) of shape (d, d)
    """
    device = X.device

    # 1) Instantiate alpha_model
    dim_x, dim_w1, dim_w2 = X.size(1), W1.size(1), W2.size(1)
    alpha_model = AlphaModel(dim_x, dim_w1, dim_w2, hidden_dim=alpha_hidden_dim, hidden_dim2=alpha_hidden_dim).to(device)

    # 2) Split data 50/50 into moment (D1) and build (D2) sets
    n = X.size(0)
    idx_moment, idx_build = sample_split(n, 2, device=device)
    def subset(idxs: Tensor):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]
    X1, Y1, W11, W21, V1, R1 = subset(idx_moment)
    X2, Y2, W12, W22, V2, R2 = subset(idx_build)

    # 3) Fit ψ functions on build set (R2 == 1)
    mask2 = (R2 == 1).squeeze(1)
    theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])
    psi_1, psi_2, psi_3 = lm_build_all_psi(
        X2[mask2], Y2[mask2], W12[mask2], W22[mask2], V2[mask2],
        theta_pre, method=method
    )

    # 4) Build φ functions (depend on alpha_model)
    phi_1, phi_2, phi_3 = general_build_all_phi_function_mar(psi_2, psi_3)

    # 5) Prepare MAR‐weighted moment_fn on moment set (R1 == 1)
    mask1 = (R1 == 1).squeeze(1)
    moment_fn = general_estimate_moments_function_mar(
        psi_1, phi_1, phi_2, phi_3,
        X1[mask1], Y1[mask1], W11[mask1], W21[mask1], V1[mask1],
        c=c
    )

    #6) Primal–dual training of alpha_model
    # alpha_model = train_alpha_with_penalty(
    #     alpha_model=alpha_model,
    #     moment_fn=moment_fn,
    #     tau=tau,
    #     lambda_pen=(lambda_lr*c - 160 * tau),
    #     lr_alpha=alpha_lr,
    #     alpha_epochs=alpha_epochs
    # )

    alpha_model = train_alpha_aug_lagrange(
        alpha_model     = alpha_model,
        moment_fn       = moment_fn,
        tau             = tau,
        lambda_init     = 0.0,
        rho_init        = 40.0,
        lr_alpha        = 2e-3,          # initial LR
        alpha_epochs    = alpha_epochs,           # total epochs
        scheduler_name  = "cosine",      # <-- keeps LR adjustment
        # scheduler_kw  = {},            # optional extra args
    )

    # 7) Final evaluation of variance and full covariance
    cov00_fn = general_get_cov00_function_alpha_mar(moment_fn)
    cov00 = float(cov00_fn(alpha_model).item())
    print(f"[MAR] cov00 = {cov00:.6f}")

    # 2) Compute the expected constraint term E[c*alpha1 + alpha2]
    cost = moment_fn(alpha_model)["E[c alpha1 + alpha2]"].item()
    print(f"[MAR] Final constraint E[c*alpha1 + alpha2] = {cost:.6f} (tau = {tau:.6f})")

    # 3) Verify whether the constraint is satisfied, up to the numerical tolerance of
    #    the augmented-Lagrangian solve (the empirical budget is enforced only up to a
    #    tolerance delta, cf. Algorithm 2).
    rel_excess = (cost - tau) / tau
    if rel_excess <= BUDGET_RTOL:
        print(f"[MAR] Constraint satisfied (relative excess {rel_excess:+.2e}, "
              f"tolerance {BUDGET_RTOL:.0e})")
    else:
        print(f"[MAR] Constraint exceeded beyond tolerance: relative excess "
              f"{rel_excess:+.2e} > {BUDGET_RTOL:.0e}")
    trace_or_cov_fn = general_get_trace_variance_function_alpha_mar(
        moment_fn=moment_fn,
        return_full=True
    )
    cov_full = trace_or_cov_fn(alpha_model)  # Tensor of shape (d, d)

    return alpha_model, cov00, cov_full


def lm_mono_debias_budget_constrained_obtain_alpha_mar_trace(
    X:  Tensor,
    Y:  Tensor,
    W1: Tensor,
    W2: Tensor,
    V:  Tensor,
    R:  Tensor,                 # (n,) int64 in {1,2,3}
    tau: float,
    c:   float,
    method: str = "mlp",
    alpha_hidden_dim: int = 32,
    alpha_epochs: int = 1500,
    alpha_lr: float = 0.01,
) -> Tuple[nn.Module, float, Tensor]:
    """
    Single-split MAR debiasing with budget constraint, specialized for
    minimizing Tr(Cov(θ̂)).

    Returns:
        alpha_model : trained AlphaModel
        trace_val   : float = Tr(Cov(θ̂)) at the trained α
        cov_full    : Tensor (d, d) = full covariance matrix Cov(θ̂)
    """
    device = X.device

    # 1) Alpha model
    dim_x, dim_w1, dim_w2 = X.size(1), W1.size(1), W2.size(1)
    alpha_model = AlphaModel(
        dim_x, dim_w1, dim_w2,
        hidden_dim=alpha_hidden_dim,
        hidden_dim2=alpha_hidden_dim
    ).to(device)

    # 2) 50/50 split
    n = X.size(0)
    idx_moment, idx_build = sample_split(n, 2, device=device)
    def subset(idxs: Tensor):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]
    X1, Y1, W11, W21, V1, R1 = subset(idx_moment)
    X2, Y2, W12, W22, V2, R2 = subset(idx_build)

    # 3) ψ on build set (R2 == 1)
    mask2 = (R2 == 1).squeeze(1)
    theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])
    psi_1, psi_2, psi_3 = lm_build_all_psi(
        X2[mask2], Y2[mask2], W12[mask2], W22[mask2], V2[mask2],
        theta_pre, method=method
    )

    # 4) φ builders (depend on alpha_model)
    phi_1, phi_2, phi_3 = general_build_all_phi_function_mar(psi_2, psi_3)

    # 5) MAR-weighted moment_fn on moment set (R1 == 1)
    mask1 = (R1 == 1).squeeze(1)
    moment_fn = general_estimate_moments_function_mar(
        psi_1, phi_1, phi_2, phi_3,
        X1[mask1], Y1[mask1], W11[mask1], W21[mask1], V1[mask1],
        c=c
    )

    # 6) Train α to minimize Tr(Cov)
    alpha_model = train_alpha_aug_lagrange_trace(
        alpha_model   = alpha_model,
        moment_fn     = moment_fn,
        tau           = tau,
        lambda_init   = 5.0,                 # stronger initial multiplier
        rho_init      = 100.0,                # penalty on same order as Tr(Cov)
        lr_alpha      = 1e-2,                 # smaller, more stable step
        alpha_epochs  = alpha_epochs,
        scheduler_name= "linear",
        scheduler_kw  = {"start_factor": 1.0, "end_factor": 0.01},
    )

    # 7) Evaluate trace and full covariance
    trace_fn = general_get_trace_variance_function_alpha_mar(moment_fn, return_full=False)
    trace_val = float(trace_fn(alpha_model).item())
    print(f"[MAR] trace(Cov) = {trace_val:.6f}")

    cost = moment_fn(alpha_model)["E[c alpha1 + alpha2]"].item()
    print(f"[MAR] Final constraint E[c*alpha1 + alpha2] = {cost:.6f} (tau = {tau:.6f})")
    rel_excess = (cost - tau) / tau
    if rel_excess <= BUDGET_RTOL:
        print(f"[MAR] Constraint satisfied (relative excess {rel_excess:+.2e}, "
              f"tolerance {BUDGET_RTOL:.0e})")
    else:
        print(f"[MAR] Constraint exceeded beyond tolerance: relative excess "
              f"{rel_excess:+.2e} > {BUDGET_RTOL:.0e}")

    cov_full_fn = general_get_trace_variance_function_alpha_mar(moment_fn, return_full=True)
    cov_full = cov_full_fn(alpha_model)  # (d, d)

    return alpha_model, trace_val, cov_full
# ======================================================================
# 1) Algorithm 1 wrapper  (choose alpha* & evaluate)
# ======================================================================



def lm_fix_alpha(
    *,
    # --------------- batch sizes -----------------
    n1: int,
    n2: int,
    reps: int,
    # --------------- model dims ------------------
    d_x: int,
    d_u1: int,
    d_u2: int,
    # --------------- model parameters ------------
    theta_star: torch.Tensor,        # (d,)
    beta1_star: torch.Tensor,
    beta2_star: torch.Tensor,
    # --------------- noise -----------------------
    sigma_eps: float,
    # --------------- MCAR / CI -------------------
    alpha_level: float,
    tau: float,
    c: float,
    alpha_init: torch.Tensor,        # (3,)
    # --------------- misc ------------------------
    seed: int = 42,
) -> Dict[str, float | torch.Tensor]:

    device, dtype = theta_star.device, theta_star.dtype
    set_global_seed(seed)
    reset_m_spec_norm()

    # ---------- Stage-1 : obtain α* -------------------------
    X1, Y1, W1_1, W2_1, V1, R1 = lm_generate_obs_data_mcar(
        n=n1,
        d_x=d_x, d_u1=d_u1, d_u2=d_u2,
        theta_star=theta_star,
        beta1_star=beta1_star,
        beta2_star=beta2_star,
        alpha=alpha_init,
        Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
        sigma_eps=sigma_eps,
    )

    alpha_opt, cov00_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )
    # MAR variant: obtain alpha_model via constrained search
    alpha_model_opt, cov00_model_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )

    # baseline α (α₂ = 0)
    alpha1_base  = tau / c
    alpha_baseline = torch.tensor(
        [alpha1_base,
         0.0,
         1.0 + (c - 1.0) * alpha1_base - tau],
        device=device, dtype=dtype
    )

    # alpha_baseline = torch.tensor(
    #     [0.5,
    #     0.0,
    #     0.5],
    #     device=device, dtype=dtype
    # )

    # ------------- containers for stage-2 -------------------
    err_opt, err_mar, err_base, err_ols = [], [], [], []
    len_opt, len_mar, len_base, len_ols = [], [], [], []
    cov_opt, cov_mar, cov_base, cov_ols = [], [], [], []

    # =================== Stage-2 loop =======================
    for rep in tqdm(range(reps),total=reps):
        set_global_seed(seed + rep)
        # 1) Generate complete data once
        X2, Y2, W1_2, W2_2, V2 = lm_generate_complete_data(
            n=n2, d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
            sigma_eps=sigma_eps,
        )
        # ----- Batch-A : α_opt -----------------------------------
        X2A, Y2A, W1A, W2A, V2A, R2A = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2,
            alpha=alpha_opt
        )
        theta_opt_vec, cov_opt_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2A, Y2A, W1A, W2A, V2A, R2A,
            alpha=alpha_opt, method="mlp"
        )

        se_opt_1 = torch.sqrt(cov_opt_mat[0, 0] / n2).item()
        ci_opt_low, ci_opt_high = wald_ci(
            mu_hat=theta_opt_vec[0].item(),
            se=se_opt_1,
            alpha_level=alpha_level
        )
        # ----- Batch-MAR : α_model_opt ------------------------
        X2M, Y2M, W1M, W2M, V2M, R2M = general_generate_mar(
            X2, Y2, W1_2, W2_2, V2,
            alpha_fn=alpha_model_opt
        )
        theta_mar_vec, cov_mar_mat = lm_mono_debias_estimate_mar_crossfit(
            X2M, Y2M, W1M, W2M, V2M, R2M,
            alpha_fn=alpha_model_opt, method="mlp"
        )
        se_mar_1 = torch.sqrt(cov_mar_mat[0, 0] / n2).item()
        ci_mar_low, ci_mar_high = wald_ci(
            mu_hat=theta_mar_vec[0].item(),
            se=se_mar_1,
            alpha_level=alpha_level
        )
        # ----- Batch-B : α_baseline -----------------------------
        X2B, Y2B, W1B, W2B, V2B, R2B = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2,
            alpha=alpha_baseline
        )

        theta_base_vec, cov_base_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2B, Y2B, W1B, W2B, V2B, R2B,
            alpha=alpha_baseline, method="mlp"
        )

        se_base_1 = torch.sqrt(cov_base_mat[0, 0] / n2).item()
        ci_base_low, ci_base_high = wald_ci(
            mu_hat=theta_base_vec[0].item(),
            se=se_base_1,
            alpha_level=alpha_level
        )

        # ----- OLS reference ------------------------------------
        mask_ols  = ~torch.isnan(Y2B).view(-1)
        X_ols = X2B[mask_ols]
        Y_ols = Y2B[mask_ols]
        n_ols = X_ols.shape[0]
        d_ols = X_ols.shape[1]
        theta_ols_vec = lm_fit_ols(X_ols, Y_ols)

        # Estimate covariance for OLS
        residuals = Y_ols - (X_ols @ theta_ols_vec.view(-1, 1))
        s_squared = (residuals.T @ residuals).item() / (n_ols - d_ols)
        inv_xtx = torch.inverse(X_ols.T @ X_ols)
        # value = s_squared * inv_xtx[0, 0].item() * (n_ols - d_ols)
        # print(f"[DEBUG]  s_squared * inv_xtx[0,0] * (n_ols-d_ols) = {value:.6f}")
        se_ols_1 = torch.sqrt(s_squared * inv_xtx[0, 0]).item()
        ci_ols_low, ci_ols_high = wald_ci(
            mu_hat=theta_ols_vec[0].item(),
            se=se_ols_1,
            alpha_level=alpha_level
        )

        # ----- accumulate metrics ------------------------------
        err_opt .append(torch.linalg.norm(theta_opt_vec[0]  - theta_star[0]).item())
        err_mar .append(torch.linalg.norm(theta_mar_vec[0] - theta_star[0]).item())
        err_base.append(torch.linalg.norm(theta_base_vec[0] - theta_star[0]).item())
        err_ols .append(torch.linalg.norm(theta_ols_vec[0]  - theta_star[0]).item())

        len_opt .append(ci_opt_high  - ci_opt_low)
        len_mar .append(ci_mar_high  - ci_mar_low)
        len_base.append(ci_base_high - ci_base_low)
        len_ols.append(ci_ols_high - ci_ols_low)

        cov_opt .append(int(ci_opt_low  <= theta_star[0].item() <= ci_opt_high))
        cov_mar .append(int(ci_mar_low  <= theta_star[0].item() <= ci_mar_high))
        cov_base.append(int(ci_base_low <= theta_star[0].item() <= ci_base_high))
        cov_ols.append(int(ci_ols_low <= theta_star[0].item() <= ci_ols_high))

    # ---------------- aggregate & return ---------------------
    return dict(
        alpha_opt = ((alpha_opt.cpu() * 1e4).round() / 1e4).tolist(),
        cov00_opt = float(cov00_opt),
        max_M_spec_norm = float(get_m_spec_norm()),
        mean_l2_opt = float(sum(err_opt)  / reps),
        mean_l2_mar = float(sum(err_mar)  / reps),
        mean_l2_base= float(sum(err_base) / reps),
        mean_l2_ols = float(sum(err_ols)  / reps),
        mean_len_opt = float(sum(len_opt)  / reps),
        mean_len_mar = float(sum(len_mar)  / reps),
        mean_len_base= float(sum(len_base) / reps),
        mean_len_ols = float(sum(len_ols) / reps),
        covg_opt  = float(sum(cov_opt)  / reps),
        covg_mar  = float(sum(cov_mar)  / reps),
        covg_base = float(sum(cov_base) / reps),
        covg_ols = float(sum(cov_ols) / reps),
    )

def lm_fix_alpha_trace_l2only(
    *,
    # --------------- batch sizes -----------------
    n1: int,
    n2: int,
    reps: int,
    # --------------- model dims ------------------
    d_x: int,
    d_u1: int,
    d_u2: int,
    # --------------- model parameters ------------
    theta_star: torch.Tensor,        # (d,)
    beta1_star: torch.Tensor,
    beta2_star: torch.Tensor,
    # --------------- noise -----------------------
    sigma_eps: float,
    # --------------- MCAR / CI -------------------
    alpha_level: float,              # kept for signature compatibility; unused
    tau: float,
    c: float,
    alpha_init: torch.Tensor,        # (3,)
    # --------------- misc ------------------------
    seed: int = 42,
) -> Dict[str, float]:
    """
    L2-only evaluation:
      - Stage-1 obtains alphas via *trace*-optimized routines (MCAR + MAR).
      - Stage-2 records ONLY vector L2 errors ||theta_hat - theta_star||_2 for:
        MCAR(trace-opt), MAR(trace-opt), MCAR(baseline), and OLS.
      - No CI length, coverage, or cov00 returned.
    """
    device, dtype = theta_star.device, theta_star.dtype
    set_global_seed(seed)

    # ---------- Stage-1: obtain alpha* (trace objective) ----------
    X1, Y1, W1_1, W2_1, V1, R1 = lm_generate_obs_data_mcar(
        n=n1,
        d_x=d_x, d_u1=d_u1, d_u2=d_u2,
        theta_star=theta_star,
        beta1_star=beta1_star,
        beta2_star=beta2_star,
        alpha=alpha_init,
        Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
        sigma_eps=sigma_eps,
    )

    # MCAR: alpha that minimizes Tr(Cov)
    alpha_opt, _trace_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_trace(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )
    # MAR: function-valued alpha that minimizes Tr(Cov)
    alpha_model_opt, _trace_model_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mar_trace(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )

    # Baseline alpha with alpha2 = 0 and alpha3 = 1 + (c-1) * alpha1 - tau
    alpha1_base = tau / c
    alpha_baseline = torch.tensor(
        [alpha1_base,
         0.0,
         1.0 + (c - 1.0) * alpha1_base - tau],
        device=device, dtype=dtype
    )

    # ------------- containers for Stage-2 (L2 only) -------------
    err_opt, err_mar, err_base, err_ols = [], [], [], []

    # =================== Stage-2 loop ===========================
    for rep in tqdm(range(reps), total=reps):
        set_global_seed(seed + rep)

        # 1) Generate complete data once
        X2, Y2, W1_2, W2_2, V2 = lm_generate_complete_data(
            n=n2, d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
            sigma_eps=sigma_eps,
        )

        # ----- Batch-A : alpha_opt (MCAR) ------------------------
        X2A, Y2A, W1A, W2A, V2A, R2A = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2,
            alpha=alpha_opt
        )
        theta_opt_vec, _ = lm_mono_debias_estimate_mcar_crossfit(
            X2A, Y2A, W1A, W2A, V2A, R2A,
            alpha=alpha_opt, method="mlp"
        )
        theta_opt_vec = theta_opt_vec.view(-1)

        # ----- Batch-MAR : alpha_model_opt -----------------------
        X2M, Y2M, W1M, W2M, V2M, R2M = general_generate_mar(
            X2, Y2, W1_2, W2_2, V2,
            alpha_fn=alpha_model_opt
        )
        theta_mar_vec, _ = lm_mono_debias_estimate_mar_crossfit(
            X2M, Y2M, W1M, W2M, V2M, R2M,
            alpha_fn=alpha_model_opt, method="mlp"
        )
        theta_mar_vec = theta_mar_vec.view(-1)

        # ----- Batch-B : alpha_baseline (MCAR) -------------------
        X2B, Y2B, W1B, W2B, V2B, R2B = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2,
            alpha=alpha_baseline
        )
        theta_base_vec, _ = lm_mono_debias_estimate_mcar_crossfit(
            X2B, Y2B, W1B, W2B, V2B, R2B,
            alpha=alpha_baseline, method="mlp"
        )
        theta_base_vec = theta_base_vec.view(-1)

        # ----- OLS reference on observed rows --------------------
        mask_ols = ~torch.isnan(Y2B).view(-1)
        X_ols = X2B[mask_ols]
        Y_ols = Y2B[mask_ols]
        theta_ols_vec = lm_fit_ols(X_ols, Y_ols).view(-1)

        # ----- accumulate vector-L2 errors -----------------------
        theta_star_vec = theta_star.view(-1)
        err_opt .append(torch.linalg.norm(theta_opt_vec  - theta_star_vec).item())
        err_mar .append(torch.linalg.norm(theta_mar_vec  - theta_star_vec).item())
        err_base.append(torch.linalg.norm(theta_base_vec - theta_star_vec).item())
        err_ols .append(torch.linalg.norm(theta_ols_vec  - theta_star_vec).item())

    # ---------------- aggregate & return (L2 only) --------------
    return dict(
        mean_l2_opt  = float(sum(err_opt)  / reps),
        mean_l2_mar  = float(sum(err_mar)  / reps),
        mean_l2_base = float(sum(err_base) / reps),
        mean_l2_ols  = float(sum(err_ols)  / reps),
    )




# ======================================================================
def lm_change_alpha_every_iter(
    *,
    # -------- batch sizes --------
    n1: int, n2: int, reps: int,
    # -------- model dims ---------
    d_x: int, d_u1: int, d_u2: int,
    # -------- true parameters ----
    theta_star: torch.Tensor,
    beta1_star: torch.Tensor,
    beta2_star: torch.Tensor,
    # -------- noise --------------
    sigma_eps: float,
    # -------- MCAR / CI ----------
    alpha_level: float, tau: float, c: float,
    alpha_init: torch.Tensor,
    # -------- misc ---------------
    seed: int = 42,
) -> Dict[str, float | torch.Tensor]:

    device, dtype = theta_star.device, theta_star.dtype

    # -------- baseline alpha (alpha_2 = 0): constant, so build it outside the loop --------
    alpha1_base  = max(1e-6, min(1 - 1e-6, tau / c))
    alpha_baseline = torch.tensor(
        [alpha1_base, 0.0, 1.0 + (c - 1.0) * alpha1_base - tau],
        device=device, dtype=dtype
    )

    # -------- containers --------
    err_opt, err_mar, err_base, err_ols = [], [], [], []
    len_opt, len_mar, len_base, len_ols = [], [], [], []
    cov_opt, cov_mar, cov_base, cov_ols = [], [], [], []

    # ======== Stage-2 loop; the Stage-1 rule search is redone on every pass ========
    for rep in tqdm(range(reps), total=reps):
        set_global_seed(seed + rep)

        # ---------- Stage-1 : obtain α*_opt & α_model_opt ----------
        X1, Y1, W1_1, W2_1, V1, R1 = lm_generate_obs_data_mcar(
            n=n1,
            d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            alpha=alpha_init,
            sigma_eps=sigma_eps,
        )

        alpha_opt, cov00_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
            X1, Y1, W1_1, W2_1, V1, R1,
            tau=tau, c=c, method="mlp"
        )

        alpha_model_opt, cov00_model_opt, _ = (
            lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00(
                X1, Y1, W1_1, W2_1, V1, R1,
                tau=tau, c=c, method="mlp"
            )
        )

        # ---------- Stage-2 : generate complete data ----------
        X2, Y2, W1_2, W2_2, V2 = lm_generate_complete_data(
            n=n2, d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            sigma_eps=sigma_eps,
        )

        # ===== (A) MCAR-α_opt =====
        XA, YA, W1A, W2A, VA, RA = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2, alpha=alpha_opt
        )
        theta_opt_vec, cov_opt_mat = lm_mono_debias_estimate_mcar_crossfit(
            XA, YA, W1A, W2A, VA, RA,
            alpha=alpha_opt, method="mlp"
        )
        se_opt = torch.sqrt(cov_opt_mat[0, 0] / n2).item()
        ci_opt_low, ci_opt_high = wald_ci(theta_opt_vec[0].item(), se_opt, alpha_level)

        # ===== (B) MAR-α_model_opt =====
        XM, YM, W1M, W2M, VM, RM = general_generate_mar(
            X2, Y2, W1_2, W2_2, V2, alpha_fn=alpha_model_opt
        )
        theta_mar_vec, cov_mar_mat = lm_mono_debias_estimate_mar_crossfit(
            XM, YM, W1M, W2M, VM, RM,
            alpha_fn=alpha_model_opt, method="mlp"
        )
        se_mar = torch.sqrt(cov_mar_mat[0, 0] / n2).item()
        ci_mar_low, ci_mar_high = wald_ci(theta_mar_vec[0].item(), se_mar, alpha_level)

        # ===== (C) MCAR-baseline α₂=0 =====
        XB, YB, W1B, W2B, VB, RB = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2, alpha=alpha_baseline
        )
        theta_base_vec, cov_base_mat = lm_mono_debias_estimate_mcar_crossfit(
            XB, YB, W1B, W2B, VB, RB,
            alpha=alpha_baseline, method="mlp"
        )
        se_base = torch.sqrt(cov_base_mat[0, 0] / n2).item()
        ci_base_low, ci_base_high = wald_ci(theta_base_vec[0].item(), se_base, alpha_level)

        # ===== (D) OLS on fully observed rows =====
        mask = ~torch.isnan(YB).view(-1)
        X_ols, Y_ols = X2[mask], Y2[mask]
        n_ols, d_ols = X_ols.shape
        theta_ols_vec = lm_fit_ols(X_ols, Y_ols)

        resid = Y_ols - (X_ols @ theta_ols_vec.view(-1, 1))
        s2 = (resid.T @ resid).item() / (n_ols - d_ols)
        inv_xtx = torch.inverse(X_ols.T @ X_ols)
        se_ols = torch.sqrt(s2 * inv_xtx[0, 0]).item()
        ci_ols_low, ci_ols_high = wald_ci(theta_ols_vec[0].item(), se_ols, alpha_level)

        # ===== accumulate metrics =====
        err_opt .append(abs(theta_opt_vec[0] - theta_star[0]).item())
        err_mar .append(abs(theta_mar_vec[0] - theta_star[0]).item())
        err_base.append(abs(theta_base_vec[0] - theta_star[0]).item())
        err_ols .append(abs(theta_ols_vec[0] - theta_star[0]).item())

        len_opt .append(ci_opt_high  - ci_opt_low)
        len_mar .append(ci_mar_high  - ci_mar_low)
        len_base.append(ci_base_high - ci_base_low)
        len_ols.append(ci_ols_high   - ci_ols_low)

        cov_opt .append(int(ci_opt_low  <= theta_star[0].item() <= ci_opt_high))
        cov_mar .append(int(ci_mar_low  <= theta_star[0].item() <= ci_mar_high))
        cov_base.append(int(ci_base_low <= theta_star[0].item() <= ci_base_high))
        cov_ols.append(int(ci_ols_low   <= theta_star[0].item() <= ci_ols_high))

    # ---------------- aggregate & return ----------------
    def _m(x): return float(sum(x) / reps)

    return dict(
        alpha_opt = ((alpha_opt.cpu() * 1e4).round() / 1e4).tolist(),  # last rep
        cov00_opt = float(cov00_opt),                                  # last rep
        mean_l2_opt   = _m(err_opt),
        mean_l2_mar   = _m(err_mar),
        mean_l2_base  = _m(err_base),
        mean_l2_ols   = _m(err_ols),
        mean_len_opt  = _m(len_opt),
        mean_len_mar  = _m(len_mar),
        mean_len_base = _m(len_base),
        mean_len_ols  = _m(len_ols),
        covg_opt   = _m(cov_opt),
        covg_mar   = _m(cov_mar),
        covg_base  = _m(cov_base),
        covg_ols   = _m(cov_ols),
    )



def lm_mcar_extended(
    *,
    # --------------- batch sizes -----------------
    n1: int,
    n2: int,
    reps: int,
    # --------------- model dims ------------------
    d_x: int,
    d_u1: int,
    d_u2: int,
    # --------------- model parameters ------------
    theta_star: torch.Tensor,        # (d,)
    beta1_star: torch.Tensor,
    beta2_star: torch.Tensor,
    # --------------- noise -----------------------
    sigma_eps: float,
    # --------------- MCAR / CI -------------------
    alpha_level: float,
    tau: float,
    c: float,
    alpha_init: torch.Tensor,        # (3,)
    # --------------- misc ------------------------
    seed: int = 42,
) -> Dict[str, float | torch.Tensor]:

    device, dtype = theta_star.device, theta_star.dtype
    set_global_seed(seed)

    # ---------- Stage-1 : obtain α* -------------------------
    X1, Y1, W1_1, W2_1, V1, R1 = lm_generate_obs_data_mcar(
        n=n1,
        d_x=d_x, d_u1=d_u1, d_u2=d_u2,
        theta_star=theta_star,
        beta1_star=beta1_star,
        beta2_star=beta2_star,
        alpha=alpha_init,
        Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
        sigma_eps=sigma_eps,
    )

    alpha_opt, cov00_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )
    # MAR variant: obtain alpha_model via constrained search
    alpha_model_opt, cov00_model_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )

    # baseline α (α₂ = 0)
    alpha1_base  = tau / c
    alpha_baseline = torch.tensor(
        [alpha1_base,
         0.0,
         1.0 + (c - 1.0) * alpha1_base - tau],
        device=device, dtype=dtype
    )

    # ------------- containers for stage-2 -------------------
    err_opt, err_mar, err_base, err_ols = [], [], [], []
    len_opt, len_mar, len_base, len_ols = [], [], [], []
    cov_opt, cov_mar, cov_base, cov_ols = [], [], [], []

    # --- NEW: containers for MAR with constant alpha --------
    err_const_opt, err_const_base = [], []
    len_const_opt, len_const_base = [], []
    cov_const_opt, cov_const_base = [], []

    alpha_const_opt  = alpha_opt.detach().clone()
    alpha_const_base = alpha_baseline.detach().clone()

    def const_fn_opt(X, W1=None, W2=None):
        return alpha_const_opt.to(X).expand(X.shape[0], 3)

    def const_fn_base(X, W1=None, W2=None):
        return alpha_const_base.to(X).expand(X.shape[0], 3)

    # =================== Stage-2 loop =======================
    for rep in tqdm(range(reps), total=reps):
        set_global_seed(seed + rep)
        # 1) Generate complete data once
        X2, Y2, W1_2, W2_2, V2 = lm_generate_complete_data(
            n=n2, d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
            sigma_eps=sigma_eps,
        )
        # ----- Batch-A : α_opt -----------------------------------
        X2A, Y2A, W1A, W2A, V2A, R2A = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2,
            alpha=alpha_opt
        )
        theta_opt_vec, cov_opt_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2A, Y2A, W1A, W2A, V2A, R2A,
            alpha=alpha_opt, method="mlp"
        )
        se_opt_1 = torch.sqrt(cov_opt_mat[0, 0] / n2).item()
        ci_opt_low, ci_opt_high = wald_ci(
            mu_hat=theta_opt_vec[0].item(),
            se=se_opt_1,
            alpha_level=alpha_level
        )

        # ----- Batch-MAR : α_model_opt ------------------------
        X2M, Y2M, W1M, W2M, V2M, R2M = general_generate_mar(
            X2, Y2, W1_2, W2_2, V2,
            alpha_fn=alpha_model_opt
        )
        theta_mar_vec, cov_mar_mat = lm_mono_debias_estimate_mar_crossfit(
            X2M, Y2M, W1M, W2M, V2M, R2M,
            alpha_fn=alpha_model_opt, method="mlp"
        )
        se_mar_1 = torch.sqrt(cov_mar_mat[0, 0] / n2).item()
        ci_mar_low, ci_mar_high = wald_ci(
            mu_hat=theta_mar_vec[0].item(),
            se=se_mar_1,
            alpha_level=alpha_level
        )

        # ----- Batch-B : α_baseline -----------------------------
        X2B, Y2B, W1B, W2B, V2B, R2B = general_generate_mcar(
            X2, Y2, W1_2, W2_2, V2,
            alpha=alpha_baseline
        )
        theta_base_vec, cov_base_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2B, Y2B, W1B, W2B, V2B, R2B,
            alpha=alpha_baseline, method="mlp"
        )
        se_base_1 = torch.sqrt(cov_base_mat[0, 0] / n2).item()
        ci_base_low, ci_base_high = wald_ci(
            mu_hat=theta_base_vec[0].item(),
            se=se_base_1,
            alpha_level=alpha_level
        )

        # ----- OLS reference ------------------------------------
        mask_ols  = ~torch.isnan(Y2B).view(-1)
        X_ols = X2B[mask_ols]
        Y_ols = Y2B[mask_ols]
        n_ols = X_ols.shape[0]
        d_ols = X_ols.shape[1]
        theta_ols_vec = lm_fit_ols(X_ols, Y_ols)

        residuals = Y_ols - (X_ols @ theta_ols_vec.view(-1, 1))
        s_squared = (residuals.T @ residuals).item() / (n_ols - d_ols)
        inv_xtx = torch.inverse(X_ols.T @ X_ols)
        se_ols_1 = torch.sqrt(s_squared * inv_xtx[0, 0]).item()
        ci_ols_low, ci_ols_high = wald_ci(
            mu_hat=theta_ols_vec[0].item(),
            se=se_ols_1,
            alpha_level=alpha_level
        )

        # --- NEW: MAR with constant α = alpha_opt --------------
        X2CO, Y2CO, W1CO, W2CO, V2CO, R2CO = general_generate_mar(
            X2, Y2, W1_2, W2_2, V2,
            alpha_fn=const_fn_opt
        )
        theta_const_opt_vec, cov_const_opt_mat = lm_mono_debias_estimate_mar_crossfit(
            X2CO, Y2CO, W1CO, W2CO, V2CO, R2CO,
            alpha_fn=const_fn_opt, method="mlp"
        )
        se_const_opt_1 = torch.sqrt(cov_const_opt_mat[0, 0] / n2).item()
        ci_const_opt_low, ci_const_opt_high = wald_ci(
            mu_hat=theta_const_opt_vec[0].item(),
            se=se_const_opt_1,
            alpha_level=alpha_level
        )

        # --- NEW: MAR with constant α = alpha_baseline ---------
    
        X2CB, Y2CB, W1CB, W2CB, V2CB, R2CB = general_generate_mar(
            X2, Y2, W1_2, W2_2, V2,
            alpha_fn=const_fn_base
        )
        theta_const_base_vec, cov_const_base_mat = lm_mono_debias_estimate_mar_crossfit(
            X2CB, Y2CB, W1CB, W2CB, V2CB, R2CB,
            alpha_fn=const_fn_base, method="mlp"
        )
        se_const_base_1 = torch.sqrt(cov_const_base_mat[0, 0] / n2).item()
        ci_const_base_low, ci_const_base_high = wald_ci(
            mu_hat=theta_const_base_vec[0].item(),
            se=se_const_base_1,
            alpha_level=alpha_level
        )

        # ----- accumulate metrics ------------------------------
        err_opt .append(torch.linalg.norm(theta_opt_vec[0]  - theta_star[0]).item())
        err_mar .append(torch.linalg.norm(theta_mar_vec[0] - theta_star[0]).item())
        err_base.append(torch.linalg.norm(theta_base_vec[0] - theta_star[0]).item())
        err_ols .append(torch.linalg.norm(theta_ols_vec[0]  - theta_star[0]).item())

        # --- NEW: accumulate for constant-α MAR ----------------
        err_const_opt .append(torch.linalg.norm(theta_const_opt_vec[0]  - theta_star[0]).item())
        err_const_base.append(torch.linalg.norm(theta_const_base_vec[0] - theta_star[0]).item())

        len_opt .append(ci_opt_high  - ci_opt_low)
        len_mar .append(ci_mar_high  - ci_mar_low)
        len_base.append(ci_base_high - ci_base_low)
        len_ols.append(ci_ols_high - ci_ols_low)

        # --- NEW: lengths for constant-α MAR -------------------
        len_const_opt .append(ci_const_opt_high  - ci_const_opt_low)
        len_const_base.append(ci_const_base_high - ci_const_base_low)

        cov_opt .append(int(ci_opt_low  <= theta_star[0].item() <= ci_opt_high))
        cov_mar .append(int(ci_mar_low  <= theta_star[0].item() <= ci_mar_high))
        cov_base.append(int(ci_base_low <= theta_star[0].item() <= ci_base_high))
        cov_ols.append(int(ci_ols_low <= theta_star[0].item() <= ci_ols_high))

        # --- NEW: coverage for constant-α MAR ------------------
        cov_const_opt .append(int(ci_const_opt_low  <= theta_star[0].item() <= ci_const_opt_high))
        cov_const_base.append(int(ci_const_base_low <= theta_star[0].item() <= ci_const_base_high))

    # ---------------- aggregate & return ---------------------
    return dict(
        alpha_opt = ((alpha_opt.cpu() * 1e4).round() / 1e4).tolist(),
        cov00_opt = float(cov00_opt),
        mean_l2_opt = float(sum(err_opt)  / reps),
        mean_l2_mar = float(sum(err_mar)  / reps),
        mean_l2_base= float(sum(err_base) / reps),
        mean_l2_ols = float(sum(err_ols)  / reps),
        mean_l2_const_opt  = float(sum(err_const_opt)  / reps),
        mean_l2_const_base = float(sum(err_const_base) / reps),

        mean_len_opt = float(sum(len_opt)  / reps),
        mean_len_mar = float(sum(len_mar)  / reps),
        mean_len_base= float(sum(len_base) / reps),
        mean_len_ols = float(sum(len_ols) / reps),
        mean_len_const_opt  = float(sum(len_const_opt)  / reps),
        mean_len_const_base = float(sum(len_const_base) / reps),

        covg_opt  = float(sum(cov_opt)  / reps),
        covg_mar  = float(sum(cov_mar)  / reps),
        covg_base = float(sum(cov_base) / reps),
        covg_ols  = float(sum(cov_ols)  / reps),
        covg_const_opt  = float(sum(cov_const_opt)  / reps),
        covg_const_base = float(sum(cov_const_base) / reps),
    )