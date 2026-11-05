"""Reproducible bounded-calibration study; requires only Python and NumPy.

Run from any directory. Outputs are generated next to this script.
The linear-regression and politeness experiments are in the repository root.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

RHO, TAU, LOWER, KAPPA = 3.0, 0.6, 0.01, 0.001
Z90 = 1.6448536269514722
SCENARIOS = {
    "global": (np.array([1.0, 2.0]), np.array([4.0, 4.0])),
    "none": (np.array([16.0, 9.0]), np.array([1.0, 1.0])),
    "local": (np.array([16.0, 1.0]), np.array([1.0, 4.0])),
}


def oracle(g, h, gold=False):
    """Exact known-stratum budget calibration of the clipped oracle formula."""
    mix = g < (RHO - 1) * h

    def rule(lam):
        pooled = np.clip(np.sqrt((g + h) / (RHO * lam)), LOWER, 1)
        a = np.where(mix, np.clip(np.sqrt(g / ((RHO - 1) * lam)), LOWER, 1), pooled)
        s = np.where(mix, np.clip(np.sqrt(h / lam), LOWER, 1), pooled)
        if gold:
            a = pooled
            s = pooled
        return a, s

    lo, hi = 1e-14, 1.0
    while np.mean((RHO - 1) * rule(hi)[0] + rule(hi)[1]) > TAU:
        hi *= 2
    for _ in range(70):
        mid = (lo + hi) / 2
        a, s = rule(mid)
        if np.mean((RHO - 1) * a + s) > TAU:
            lo = mid
        else:
            hi = mid
    return rule(hi)


def risk(g, h, a, s):
    return float(np.mean(g / a + h / s))


def generate(rng, n, g, h):
    x = rng.integers(0, 2, n)
    sign = 2 * rng.integers(0, 2, n) - 1
    mu = np.sqrt(h[x])
    high = rng.random(n) < h[x] / (h[x] + 4 * g[x])
    magnitude = np.where(high, mu + 2 * g[x] / mu, mu / 2)
    return x, (sign > 0).astype(int), sign * magnitude


def fit_nuisance(x, v, y, complete, a):
    """Complete-case IPW linear fits, with a reported eigenvalue safeguard."""
    n = len(x)
    obs = np.flatnonzero(complete)
    w = 1 / a[x[obs]]
    q = max(float(np.sum(w) / n), KAPPA)
    theta = float(np.dot(w, y[obs]) / n / q)
    response = (y[obs] - theta) / q
    c2 = 2 * x[obs] + v[obs]
    gram2 = np.bincount(c2, weights=w, minlength=4) / n
    cross2 = np.bincount(c2, weights=w * response, minlength=4) / n
    gram3 = np.bincount(x[obs], weights=w, minlength=2) / n
    cross3 = np.bincount(x[obs], weights=w * response, minlength=2) / n
    return theta, q, cross2 / np.maximum(gram2, KAPPA), cross3 / np.maximum(gram3, KAPPA)


def candidate_rules():
    grid = np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15,
                     0.175, 0.2, 0.225, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0])
    pairs = np.array([(a, s) for a in grid for s in grid if a <= s])
    ix, iy = np.meshgrid(np.arange(len(pairs)), np.arange(len(pairs)), indexing="ij")
    a = np.column_stack((pairs[ix.ravel(), 0], pairs[iy.ravel(), 0]))
    s = np.column_stack((pairs[ix.ravel(), 1], pairs[iy.ravel(), 1]))
    feasible = np.mean((RHO - 1) * a + s, axis=1) <= TAU + 1e-12
    return a[feasible], s[feasible]


def learn_rules(initial, candidates, cm):
    x, v, y = initial
    m = len(x) // 2
    fit = fit_nuisance(x[:m], v[:m], y[:m], np.ones(m, dtype=bool), np.ones(2))
    theta, q, coef2, coef3 = fit
    xv, vv, yv = x[m:], v[m:], y[m:]
    psi1 = (yv - theta) / q
    psi2, psi3 = coef2[2 * xv + vv], coef3[xv]
    moments = [
        np.bincount(xv, weights=values, minlength=2) / len(xv)
        for values in (psi1**2, psi1 * psi2, psi1 * psi3, psi2**2, psi3**2)
    ]
    a, s = candidates
    wg, wh = 1 / a - 1 / s, 1 / s - 1
    t = (moments[0] / a).sum(axis=1)
    d = (wg * moments[1] + wh * moments[2]).sum(axis=1)
    b = (wg * moments[3] + wh * moments[4]).sum(axis=1)
    cal = np.clip(np.divide(d, b, out=np.zeros_like(d), where=b > 0), -cm, cm)
    objective = t - 2 * cal * d + cal**2 * b

    def best(mask):
        ii = np.flatnonzero(mask)
        idx = ii[np.argmin(objective[ii])]
        return a[idx], s[idx]

    counts = np.bincount(xv, minlength=2)
    ghat = np.bincount(xv, weights=(psi1 - psi2)**2, minlength=2) / np.maximum(counts, 1)
    hhat = np.bincount(xv, weights=(psi2 - psi3)**2, minlength=2) / np.maximum(counts, 1)
    return {
        "PCAL-grid": best(np.ones(len(a), dtype=bool)),
        "Plug-in": oracle(np.maximum(ghat, 1e-8), np.maximum(hhat, 0)),
        "Gold-grid": best(np.all(a == s, axis=1)),
        "Constant-grid": best((a[:, 0] == a[:, 1]) & (s[:, 0] == s[:, 1])),
    }


def influence_components(x, v, y, complete, compared, a, s, fit):
    theta, q, coef2, coef3 = fit
    av, sv = a[x], s[x]
    u = np.zeros(len(x))
    u[complete] = (y[complete] - theta) / q / av[complete]
    h = (1 - compared / sv) * coef3[x]
    # V is used only if the comparison is observed. Gold cases reveal V as well.
    observed = np.flatnonzero(compared)
    h[observed] += (
        1 / sv[observed] - complete[observed] / av[observed]
    ) * coef2[2 * x[observed] + v[observed]]
    return u, h


def infer(data, uniform, a, s, cm):
    x, v, y = data
    n = len(x)
    complete, compared = uniform < a[x], uniform < s[x]
    folds = np.array_split(np.arange(n), 3)
    theta_hat, eta = 0.0, np.empty(n)
    for k in range(3):
        ev, cal, tr = folds[k], folds[(k + 1) % 3], folds[(k + 2) % 3]
        fit = fit_nuisance(x[tr], v[tr], y[tr], complete[tr], a)
        uc, hc = influence_components(
            x[cal], v[cal], y[cal], complete[cal], compared[cal], a, s, fit
        )
        bhat = np.mean(hc**2)
        mhat = float(np.clip(-np.mean(uc * hc) / bhat, -cm, cm)) if bhat > 0 else 0.0
        ue, he = influence_components(
            x[ev], v[ev], y[ev], complete[ev], compared[ev], a, s, fit
        )
        observed = ev[complete[ev]]
        weights = 1 / a[x[observed]]
        base_q = max(float(weights.sum() / len(ev)), KAPPA)
        base = float(np.dot(weights, y[observed]) / len(ev) / base_q)
        theta_hat += len(ev) / n * (base + mhat * np.mean(he))
        eta[ev] = ue + mhat * he
    variance = float(np.var(eta, ddof=0))
    halfwidth = Z90 * math.sqrt(variance / n)
    cost = float(np.mean(RHO * complete + (compared & ~complete)))
    return theta_hat, variance, int(abs(theta_hat) <= halfwidth), 2 * halfwidth, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--n0", type=int, default=2000)
    parser.add_argument("--n1", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--output", default="controlled_results")
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / args.output
    out.mkdir(parents=True, exist_ok=True)
    candidates = candidate_rules()
    records, oracle_rows = [], []
    for snum, (scenario, (g, h)) in enumerate(SCENARIOS.items()):
        true_mixed, true_gold = oracle(g, h), oracle(g, h, gold=True)
        grid_risk = np.mean(g / candidates[0] + h / candidates[1], axis=1)
        const_r = oracle(np.repeat(g.mean(), 2), np.repeat(h.mean(), 2))
        oracle_rows.append(dict(
            scenario=scenario, mixed=risk(g, h, *true_mixed),
            gold=risk(g, h, *true_gold), constant=risk(g, h, *const_r),
            grid=float(np.min(grid_risk)),
            preference_0=float(true_mixed[1][0] - true_mixed[0][0]),
            preference_1=float(true_mixed[1][1] - true_mixed[0][1]),
        ))
        for rep in range(args.replications):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, snum, rep]))
            initial = generate(rng, args.n0, g, h)
            data = generate(rng, args.n1, g, h)
            uniform = rng.random(args.n1)
            rules = learn_rules(initial, candidates, args.radius)
            rules["Oracle-gold"] = true_gold
            for method, (a, s) in rules.items():
                assert np.all(a >= LOWER) and np.all(a <= s) and np.all(s <= 1)
                expected_cost = float(np.mean((RHO - 1) * a + s))
                assert expected_cost <= TAU + 1e-10
                est, var, cov, length, cost = infer(data, uniform, a, s, args.radius)
                records.append(dict(
                    scenario=scenario, rep=rep, method=method, estimate=est,
                    estimated_rootn_variance=var, covered=cov, ci_length=length,
                    realized_cost=cost, expected_cost=expected_cost,
                    preference_0=float(s[0] - a[0]), preference_1=float(s[1] - a[1]),
                    true_efficient_risk=risk(g, h, a, s),
                ))
        print(f"Completed {scenario}: {args.replications} replications", flush=True)
    with (out / "replications.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for scenario in SCENARIOS:
        for method in ["PCAL-grid", "Plug-in", "Gold-grid", "Constant-grid", "Oracle-gold"]:
            rows = [r for r in records if r["scenario"] == scenario and r["method"] == method]
            estimates = np.array([r["estimate"] for r in rows])
            row = dict(scenario=scenario, method=method, bias=float(estimates.mean()),
                       empirical_rootn_variance=float(args.n1 * estimates.var(ddof=1)))
            centered = estimates - estimates.mean()
            fourth = float(np.mean(centered**4))
            sample_var = float(estimates.var(ddof=1))
            r = len(estimates)
            row["empirical_rootn_variance_mcse"] = args.n1 * math.sqrt(
                max(0.0, fourth - (r - 3) / (r - 1) * sample_var**2) / r
            )
            for key in ["covered", "ci_length", "realized_cost", "expected_cost",
                        "preference_0", "preference_1", "true_efficient_risk"]:
                values = np.array([r[key] for r in rows])
                row[key] = float(values.mean())
                row[key + "_mcse"] = float(values.std(ddof=1) / math.sqrt(len(values)))
            summary.append(row)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = dict(vars(args), numpy_version=np.__version__, rho=RHO, tau=TAU,
                    lower=LOWER, kappa=KAPPA, candidate_count=len(candidates[0]),
                    oracle_values=oracle_rows)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = []
    for row in summary:
        lines.append(
            f"{row['scenario'].capitalize()} & {row['method']} & "
            f"{row['empirical_rootn_variance']:.2f} ({row['empirical_rootn_variance_mcse']:.2f}) & {row['covered']:.3f} & "
            f"{row['ci_length']:.3f} & {row['expected_cost']:.3f} \\\\"
        )
    (out / "main_table.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for row in summary:
        lines.append(
            f"{row['scenario'].capitalize()} & {row['method']} & "
            f"{row['bias']:.4f} & {row['covered_mcse']:.3f} & "
            f"{row['preference_0']:.3f} & {row['preference_1']:.3f} & "
            f"{row['true_efficient_risk']:.2f} \\\\"
        )
    (out / "supp_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    for row in summary:
        print(row["scenario"], row["method"],
              "variance", round(row["empirical_rootn_variance"], 3),
              "coverage", round(row["covered"], 3),
              "preference", round(row["preference_0"], 3), round(row["preference_1"], 3))


if __name__ == "__main__":
    main()
