"""
politeness.py
-------------------------
Run the politeness experiment.

Usage
-----
python politeness.py --reps 5 --tau_vals 3,4,5
"""

# ── stdlib ────────────────────────────────────────────────────────────
import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Dict

# ── third‑party ───────────────────────────────────────────────────────
import torch
import pandas as pd
import numpy as np
from torch import Tensor

# ── internal packages ────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.estimators import (
    lm_fit_ols,
    reset_m_spec_norm,
    get_m_spec_norm,
)
from src.lm_mono_debias import (
    lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00,
    lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00,
    lm_mono_debias_estimate_mcar_crossfit,
    lm_mono_debias_estimate_mar_crossfit,
    general_generate_mcar,
    general_generate_mar,
)
from src.utils import set_global_seed, dump_run_simple, wald_ci
from tqdm import tqdm

# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Politeness Experiment")

    # ---------- device & RNG ----------
    parser.add_argument("--device", default="auto", help="'auto' = CUDA if available; otherwise 'cpu', 'cuda', 'cuda:1', ...")
    parser.add_argument("--seed", default=43, type=int)

    # ---------- batch sizes & repetitions ----------
    parser.add_argument("--n1", default=1500, type=int, help="Number of samples for stage 1")
    parser.add_argument("--n2_per_rep", default=3000, type=int, help="Number of samples for each stage 2 repetition")
    parser.add_argument("--reps", default=100, type=int)

    # ---------- MCAR / CI parameters ----------
    parser.add_argument("--alpha_level", default=0.1, type=float)
    parser.add_argument("--tau_vals", default="3,5,10", help="Comma-separated list of tau values")
    parser.add_argument("--c_vals", default="20,50,100,200", help="Comma-separated list of c values")
    parser.add_argument("--alpha_init", default="1.0,0.0,0.0",
                        help="Stage-1 masking distribution over the three patterns; "
                             "the default leaves the stage-1 sample fully observed")

    parser.add_argument("--n_total", default=5515, type=int, help="Total number of rows to use from the dataset")


    return parser.parse_args()

# =====================================================================
# Data Loading
# =====================================================================
def load_and_preprocess_data(n_total: int, seed: int) -> Dict[str, Tensor]:
    """Loads and preprocesses the politeness data."""
    ROOT = Path(__file__).parent
    DATA_DIR = ROOT / "data"
    CSV_PATH = DATA_DIR / "polite_scores_with_ds.csv"

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    features_to_use = ["Feature_3", "Feature_10"]
    data = pd.read_csv(CSV_PATH)
    data = data.dropna(subset=["gpt_score", "predicted_score", "Normalized Score"] + features_to_use)

    data = data.head(n_total).sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(data)

    W1 = torch.tensor(data["gpt_score"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    W2 = torch.tensor(data["predicted_score"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    Y = torch.tensor(data["Normalized Score"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    
    X_features = torch.tensor(data[features_to_use].values, dtype=torch.float32)
    X = torch.cat((X_features, torch.ones(n, 1, dtype=torch.float32)), dim=1)
    
    V = (torch.abs(W1 - Y) <= torch.abs(W2 - Y)).float()

    return {"X": X, "Y": Y, "W1": W1, "W2": W2, "V": V}

# =====================================================================
# Experiment Core Logic
# =====================================================================
def run_politeness_experiment(
    *,
    X: Tensor, Y: Tensor, W1: Tensor, W2: Tensor, V: Tensor,
    n1: int, n2_per_rep: int, reps: int,
    theta_star: Tensor, alpha_level: float,
    tau: float, c: float, alpha_init: Tensor,
    seed: int = 42,
) -> Dict[str, float]:
    """Runs the core politeness experiment for a given tau and c."""
    device, dtype = X.device, X.dtype
    torch.manual_seed(seed)
    reset_m_spec_norm()

    N = X.shape[0]
    n2_pool_size = N - n1

    if N < n1 + n2_per_rep:
        raise ValueError(f"Dataset too small: need >= n1+n2_per_rep={n1+n2_per_rep}, got {N}")

    # -------- Stage-1: obtain α* on a fixed split ----------
    perm = torch.randperm(N, device=device)
    idx1, idx2_pool = perm[:n1], perm[n1:]
    X1, Y1, W1_1, W2_1, V1 = X[idx1], Y[idx1], W1[idx1], W2[idx1], V[idx1]

    max_retries = 5
    success = False
    for attempt in range(max_retries):
        try:
            # R1 ~ Categorical(alpha_init)
            alpha_init = alpha_init.to(device=device, dtype=dtype)
            R1 = torch.multinomial(alpha_init.view(1, -1).expand(n1, -1), 1) + 1
            Y1_masked = Y1.clone().float()
            V1_masked = V1.clone().float()
            r1 = R1.view(-1, 1)
            Y1_masked[r1 == 2] = torch.nan
            Y1_masked[r1 == 3] = torch.nan
            V1_masked[r1 == 3] = torch.nan

            # learn α* (MCAR) and α_model (MAR)
            alpha_opt, cov00_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
                X1, Y1_masked, W1_1, W2_1, V1_masked, R1, tau=tau, c=c, method="mlp"
            )
            alpha_model_opt, _, _ = lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00(
                X1, Y1_masked, W1_1, W2_1, V1_masked, R1, tau=tau, c=c, method="mlp"
            )
            success = True
            break
        except Exception as e:
            print(f"[Warning] attempt {attempt+1}: {e}")
            continue
    
    if not success:
        print(f"[Fail] after {max_retries} retries, pass.")
        return {}

    # baseline α (α2=0)
    alpha1_base = float(tau) / float(c)
    alpha_baseline = torch.tensor(
        [alpha1_base, 0.0, 1.0 + (c - 1.0) * alpha1_base - tau],
        device=device, dtype=dtype
    ).clamp(min=0)
    alpha_baseline = alpha_baseline / alpha_baseline.sum().clamp_min(1e-12)

    # -------- containers ----------
    err_opt, err_mar, err_base, err_ols = [], [], [], []
    len_opt, len_mar, len_base, len_ols = [], [], [], []
    cov_opt, cov_mar, cov_base, cov_ols = [], [], [], []
    theta_star_val = float(theta_star[0].item())

    # =================== Stage-2 loop ==================
    for rep in tqdm(range(reps), total=reps, desc=f"c={c}, tau={tau}"):
        torch.manual_seed(seed + rep)

        subperm = torch.randperm(n2_pool_size, device=device)
        idx2 = idx2_pool[subperm[:n2_per_rep]]
        X2, Y2, W1_2, W2_2, V2 = X[idx2], Y[idx2], W1[idx2], W2[idx2], V[idx2]

        # ---- Batch-A: MCAR with alpha_opt ----
        alphaA = alpha_opt.to(device).clamp(min=0)
        alphaA = alphaA / alphaA.sum().clamp_min(1e-12)
        X2A_obs, Y2A_obs, W1A_obs, W2A_obs, V2A_obs, R2A = general_generate_mcar(X=X2, Y=Y2, W1=W1_2, W2=W2_2, V=V2, alpha=alphaA)
        theta_opt_vec, cov_opt_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2A_obs, Y2A_obs, W1A_obs, W2A_obs, V2A_obs, R2A, alpha=alphaA, method="mlp"
        )
        se_opt_1 = torch.sqrt(cov_opt_mat[0, 0] / n2_per_rep).item()
        ci_opt_low, ci_opt_high = wald_ci(theta_opt_vec[0].item(), se_opt_1, alpha_level)
        err_opt.append(torch.linalg.norm(theta_opt_vec - theta_star).item())
        len_opt.append(ci_opt_high - ci_opt_low)
        cov_opt.append(int(ci_opt_low <= theta_star_val <= ci_opt_high))

        # ---- Batch-MAR: MAR with alpha_model_opt ----
        X2M_obs, Y2M_obs, W1M_obs, W2M_obs, V2M_obs, R2M = general_generate_mar(X=X2, Y=Y2, W1=W1_2, W2=W2_2, V=V2, alpha_fn=alpha_model_opt)
        theta_mar_vec, cov_mar_mat = lm_mono_debias_estimate_mar_crossfit(
            X2M_obs, Y2M_obs, W1M_obs, W2M_obs, V2M_obs, R2M, alpha_fn=alpha_model_opt, method="mlp"
        )
        se_mar_1 = torch.sqrt(cov_mar_mat[0, 0] / n2_per_rep).item()
        ci_mar_low, ci_mar_high = wald_ci(theta_mar_vec[0].item(), se_mar_1, alpha_level)
        err_mar.append(torch.linalg.norm(theta_mar_vec - theta_star).item())
        len_mar.append(ci_mar_high - ci_mar_low)
        cov_mar.append(int(ci_mar_low <= theta_star_val <= ci_mar_high))

        # ---- Batch-B: MCAR baseline ----
        X2B_obs, Y2B_obs, W1B_obs, W2B_obs, V2B_obs, R2B = general_generate_mcar(X=X2, Y=Y2, W1=W1_2, W2=W2_2, V=V2, alpha=alpha_baseline)
        theta_base_vec, cov_base_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2B_obs, Y2B_obs, W1B_obs, W2B_obs, V2B_obs, R2B, alpha=alpha_baseline, method="mlp"
        )
        se_base_1 = torch.sqrt(cov_base_mat[0, 0] / n2_per_rep).item()
        ci_base_low, ci_base_high = wald_ci(theta_base_vec[0].item(), se_base_1, alpha_level)
        err_base.append(torch.linalg.norm(theta_base_vec - theta_star).item())
        len_base.append(ci_base_high - ci_base_low)
        cov_base.append(int(ci_base_low <= theta_star_val <= ci_base_high))

        # ---- OLS on observed Y ----
        mask_ols = ~torch.isnan(Y2B_obs).view(-1)
        X_ols, Y_ols = X2B_obs[mask_ols], Y2B_obs[mask_ols]
        n_ols, d_ols = X_ols.shape
        theta_ols_vec = lm_fit_ols(X_ols, Y_ols)
        residuals = Y_ols - (X_ols @ theta_ols_vec.view(-1, 1))
        s_squared = (residuals.T @ residuals).item() / (n_ols - d_ols)
        inv_xtx = torch.inverse(X_ols.T @ X_ols)
        se_ols_1 = torch.sqrt(s_squared * inv_xtx[0, 0]).item()
        ci_ols_low, ci_ols_high = wald_ci(theta_ols_vec[0].item(), se_ols_1, alpha_level)
        err_ols.append(torch.linalg.norm(theta_ols_vec - theta_star).item())
        len_ols.append(ci_ols_high - ci_ols_low)
        cov_ols.append(int(ci_ols_low <= theta_star_val <= ci_ols_high))

    # -------- return --------
    return dict(
        alpha_opt=((alpha_opt.detach().cpu() * 1e4).round() / 1e4).tolist(),
        cov00_opt=float(cov00_opt),
        max_M_spec_norm=float(get_m_spec_norm()),
        mean_l2_opt=float(sum(err_opt) / reps),
        mean_l2_mar=float(sum(err_mar) / reps),
        mean_l2_base=float(sum(err_base) / reps),
        mean_l2_ols=float(sum(err_ols) / reps),
        mean_len_opt=float(sum(len_opt) / reps),
        mean_len_mar=float(sum(len_mar) / reps),
        mean_len_base=float(sum(len_base) / reps),
        mean_len_ols=float(sum(len_ols) / reps),
        covg_opt=float(sum(cov_opt) / reps),
        covg_mar=float(sum(cov_mar) / reps),
        covg_base=float(sum(cov_base) / reps),
        covg_ols=float(sum(cov_ols) / reps),
    )

def main(args):
    """Main driver."""
    # ── setup ──────────────────────────────────────────────────────────────────
    set_global_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float32
    
    # -- paths and constants
    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    ALPHA_LEVEL = args.alpha_level
    N1 = args.n1
    N2_PER_REP = args.n2_per_rep
    N_TOTAL = args.n_total

    # -- load data
    data = load_and_preprocess_data(n_total=N_TOTAL, seed=args.seed)
    X, Y, W1, W2, V = (
        data["X"].to(device, dtype),
        data["Y"].to(device, dtype),
        data["W1"].to(device, dtype),
        data["W2"].to(device, dtype),
        data["V"].to(device, dtype),
    )

    # -- calculate theta_star
    theta_star = lm_fit_ols(X, Y)

    # -- experiment params
    tau_vals = [float(v) for v in args.tau_vals.split(",")]
    c_vals = [float(v) for v in args.c_vals.split(",")]
    alpha_init = torch.tensor(
        [float(v) for v in args.alpha_init.split(",")], device=device, dtype=dtype
    )

    # -- results container
    results = []

    # ── loop ───────────────────────────────────────────────────────────────────
    for tau in tau_vals:
        for c in c_vals:
            if c < 1.0: continue
            
            res = run_politeness_experiment(
                X=X, Y=Y, W1=W1, W2=W2, V=V,
                n1=N1, n2_per_rep=N2_PER_REP, reps=args.reps,
                theta_star=theta_star, alpha_level=ALPHA_LEVEL,
                tau=tau, c=c, alpha_init=alpha_init,
                seed=args.seed,
            )
            res.update({"tau": tau, "c": c})
            results.append(res)

    # ── print and save ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index(["tau", "c"]).round(4).sort_index()
    print(df)

    # save summary + params --------------------------------
    params = {
        "cmd": " ".join(sys.argv),
        "args": vars(args),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if not df.empty:
        out_dir = dump_run_simple(df=df, params=params)
        print(f"[INFO] results saved to {out_dir}")
    else:
        print("[INFO] No results to save.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
