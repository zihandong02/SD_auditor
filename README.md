# Preference-Calibrated Active Learning (PCAL)

Code and data for the numerical experiments in *Labels or Comparative Preferences?
Budget-Constrained Inference with AI-Generated Outputs*.

The repository contains the three experiments reported in the paper — a linear-regression
simulation, a semi-synthetic study on a politeness corpus, and a controlled two-stratum
study of when preferences are worth acquiring — together with the archived output of the
runs behind every submitted figure and table.

---

## Requirements

**Python 3.11 or 3.12** with the package versions in `requirements.txt`.
`numpy==2.3.1` does not install on 3.9 or 3.10. The pinned `torch==2.5.1` also requires
a supported Python version and platform; use Python 3.11 or 3.12 on Apple silicon macOS.
A GPU is *not* required; `lm_simulation.py` and `politeness.py` accept `--device cpu` and
fall back to CPU automatically. The controlled study needs only NumPy.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use the pinned dependencies, CPU execution and a single thread for reproduction.
The commands below specify the seeds and experiment settings. Floating-point results
can vary across runtimes; see [Hardware and random numbers](#hardware-and-random-numbers).

---

## Layout

```
lm_simulation.py        linear-regression simulation
politeness.py           politeness semi-synthetic study
plot.py                 plotting helper
src/
  data_generation.py    synthetic data generator and missingness mechanisms
  estimators.py         moment estimators, calibration matrix and radius check, acquisition search
  lm_mono_debias.py     experiment drivers and cross-fitted estimators
  models/               MLP nuisance and acquisition-rule networks
  utils/                seeding, device selection, Wald intervals, run dumping
controlled/
  controlled_acquisition.py        controlled two-stratum study (NumPy only)
  test_controlled_acquisition.py   fast deterministic checks
  controlled_results/              archived output, C_M = 1
  controlled_radius2/, controlled_radius5/   archived output, C_M = 2 and 5
data/
  polite_scores_with_ds.csv   politeness corpus with AI-predicted scores
results/                archived runs behind the submitted figures
```

Each run of `lm_simulation.py` or `politeness.py` writes `results/_<YYYYMMDD_HHMMSS>/`
containing `summary.csv` (one row per budget level), `summary.txt` and `params.json` (the
full argument set and command line). Figures are drawn from `summary.csv` by `plot.py`,
for example

```bash
python plot.py --csv results/_20261102_223146/summary.csv --out_dir figures --c 50 --prefix politeness_ci_cov
```

which writes `figures/politeness_ci_cov_vs_tau_c50.pdf`; use `--prefix lm_ci_cov` for the
linear-regression study.

---

## Reproducing the figures and tables

Every plotted point in the submitted figures comes from the archived run listed below.

### Linear-regression simulation

```bash
export OMP_NUM_THREADS=1                # as in the archived runs
export CUDA_VISIBLE_DEVICES=""          # hide any GPU; see Hardware and random numbers

# Figure 1 (main text, c = 10)          -> results/_20261026_185540
python lm_simulation.py --tau_vals 1,2,3,4,5,6,7,8,9 --reps 100 --seed 3000 --device cpu

# Appendix, c = 5                        -> results/_20261026_185354
python lm_simulation.py --tau_vals 1,2,3,4 --c 5 --reps 100 --seed 3000 --device cpu

# Appendix, c = 20                       -> results/_20261026_185614
python lm_simulation.py --tau_vals 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --c 20 --reps 100 --seed 3000 --device cpu
```

These use the defaults `--n1 2000 --n2 20000 --sigma_eps 2.0 --alpha_level 0.1
--C_M 10`. Note that `--sigma_eps 2.0` produces standard deviations 16, 4 and 6
for `eps`, `eps_1` and `eps_2`; the multipliers 8, 2, 3 are fixed in
`src/data_generation.py`. In this script and in `politeness.py`, `--n1` is the size of the
initial sample and `--n2` (`--n2_per_rep`) that of the Stage 2 sample, the paper's n0 and
n1.

The archived runs were produced under `torchrun`, one process per tau. Distributed
execution only splits the tau grid, so the serial commands above give the same rows; we
checked the archives against separate single-tau runs. To run the distributed form,
launch it under `torchrun`, which sets the environment variables `torch.distributed`
requires:

```bash
torchrun --nproc_per_node=9 lm_simulation.py --tau_vals 1,2,3,4,5,6,7,8,9 --reps 100 --seed 3000 --device cpu
```

### Politeness semi-synthetic study

```bash
export OMP_NUM_THREADS=1                # as in the archived runs
export CUDA_VISIBLE_DEVICES=""          # hide any GPU; see Hardware and random numbers

# Figure 2 (main text, c = 50)           -> results/_20261102_223146
python politeness.py --tau_vals 25,27,29,31,33,35,37,39,41 --c_vals 50 --device cpu

# Appendix, c = 20                       -> results/_20261102_224114
python politeness.py --tau_vals 10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15 --c_vals 20 --device cpu

# Appendix, c = 100                      -> results/_20261102_223447
python politeness.py --tau_vals 35,40,45,50,55,60,65,70,75,80 --c_vals 100 --device cpu

# Appendix, c = 200                      -> results/_20261102_222708
python politeness.py --tau_vals 60,65,70,75,80,85,90,95,100 --c_vals 200 --device cpu
```

These use the defaults `--n1 1500 --n2_per_rep 3000 --n_total 5515 --reps 100
--alpha_init 1.0,0.0,0.0 --alpha_level 0.1 --seed 43 --C_M 10`, so only the budget grid
and the cost ratio have to be given. As in the linear-regression commands, `--device cpu`
and the hidden GPU keep the whole run on the CPU, as in the archives. The default
`--alpha_init 1.0,0.0,0.0` leaves the stage-1 sample fully observed; the acquisition rule
is learned from that sample and then reused across the 100 replications, so the reported rule is one draw rather than an
average over pilots.

A quick check that the installation works, without waiting for a full run:

```bash
export CUDA_VISIBLE_DEVICES=""          # as above, if the machine has a GPU
python lm_simulation.py --device cpu --reps 1 --tau_vals 3 --n1 200 --n2 400
python politeness.py    --device cpu --reps 2 --tau_vals 50 --c_vals 100
```

### Controlled study

```bash
python controlled/controlled_acquisition.py                                        # C_M = 1
python controlled/controlled_acquisition.py --radius 2 --output controlled_radius2
python controlled/controlled_acquisition.py --radius 5 --output controlled_radius5
python -m unittest discover -s controlled                                          # fast checks
```

These use the defaults `--replications 1000 --n0 2000 --n1 10000 --seed 20260918
--radius 1` (1000 replications for each of the three regimes); the two sensitivity runs
change only `--radius` and `--output`. In the three grid comparators (PCAL-grid, Gold-grid,
Constant-grid), each stratum's gold and preference probabilities lie on the grid {0.01,
0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4, 0.6, 0.8, 1}, with
gold probability at most preference probability; Plug-in and Oracle-gold use continuous
rules. Each run takes about 15
seconds and writes its output next to the script (`controlled/controlled_results/` for the
first command); `metadata.json` records the arguments and the NumPy version (2.3.1 for the
archived outputs).

`main_table.tex` and `supp_table.tex` hold the rows of the controlled-study table in the
main text and of the diagnostics table in the Supplementary Appendix; the
radius-sensitivity table consists of the PCAL-grid rows of the three `summary.csv` files.
The paper uses no TeX table from the two sensitivity runs, so `controlled_radius2/` and
`controlled_radius5/` keep only `replications.csv`, `summary.csv` and `metadata.json`.
The unit tests check the exact moments and budgets, the size of the candidate grid (8447
rules), that unobserved labels cannot affect inference, and that the tables in
`controlled_results/` equal the manuscript's rows, which are embedded in the test.

---

## Reading the output

`summary.csv` reports, for each budget level `tau`, the mean estimation error, mean
confidence interval length and empirical coverage of the four methods. The error
(`mean_l2_*`) is the absolute error of the first coordinate in the linear-regression study
and the Euclidean error of the coefficient vector in the politeness study. The column
suffixes predate the paper's terminology; `plot.py` now labels the curves as the paper does:

| CSV suffix | Legend label | Paper |
|---|---|---|
| `_mar` | `PCAL` | PCAL (covariate-aware) |
| `_opt` | `PCAL-CA` | PCAL-CA (covariate-agnostic) |
| `_base` | `label+unlabeled` | label+unlabeled |
| `_ols` | `label-only` | label-only |

`alpha_opt` is the learned covariate-agnostic acquisition rule
`(alpha_1, alpha_2, alpha_3)`; `cov00_opt` is the design criterion at that rule, evaluated
on the second of the two Stage 1 cross-fitting folds (the search minimises their average).
Because inference targets the first coordinate of `theta`, the criterion minimised is the
asymptotic variance of that coordinate.

`max_M_spec_norm_stage1_ca` and `max_M_spec_norm_stage1_aware` are the largest spectral
norms of the calibration matrix over every candidate rule of the covariate-agnostic and
covariate-aware searches in Stage 1, and `max_M_spec_norm` is the largest over every
inference fold and replication in Stage 2 (see [Calibration radius](#calibration-radius)).

The controlled study writes, in each output directory, `replications.csv` (one row per
regime, replication and method: estimate, interval length, coverage indicator, expected and
realized cost, allocation and design diagnostics), `summary.csv` (means with Monte Carlo
standard errors), the two TeX tables and `metadata.json`.

---

## Calibration radius

For the linear-regression and politeness studies the paper fixes a calibration radius
`C_M = 10` (`--C_M`). The code applies the unconstrained coefficient `M = -A B^{-1}`, which
equals the bounded minimiser whenever `||M||_sp <= C_M`. `src/estimators.py` checks this
every time `M` is formed: in Stage 1 at every candidate rule of both acquisition searches,
at the selected covariate-agnostic rule on both cross-fitting folds and at the selected
covariate-aware rule on its calibration split, and in Stage 2 at every inference fold. A run stops with `CalibrationRadiusExceeded` as soon as `||M||_sp > C_M`, or if `M` is
not finite, so a run that finishes has computed the bounded-calibration procedure exactly;
the maxima are written to `summary.csv`.

Largest `||M||_sp` recorded in the archived designs of each study:

| | Stage 1, covariate-agnostic | Stage 1, covariate-aware | Stage 2 |
|---|---|---|---|
| Politeness | 0.98 | 1.22 | 6.02 |
| Linear regression | 1.38 | 2.83 | 2.30 |

All values are recorded by the runs themselves, in the archived `summary.csv` files.

The controlled study implements bounded calibration directly: its scalar coefficient is
clipped to `[-C_M, C_M]`, with `C_M = 1` by default.

---

## Hardware and random numbers

PyTorch draws the simulated data from the generator of the device the tensors live on, and
the CPU and CUDA generators give different streams for the same seed, so the whole run
has to stay on the CPU. `--device cpu` alone does not ensure this on a machine with a GPU:
the nuisance networks and the linear-regression data generator use CUDA whenever it is
available, whatever `--device` says. The commands above therefore also set
`CUDA_VISIBLE_DEVICES=""`, which hides the GPU from PyTorch. Use `OMP_NUM_THREADS=1`
to keep reductions single-threaded. The least-squares fits explicitly select
`torch.linalg.lstsq(..., driver="gels")` so the solver choice is fixed.

Different numerical libraries and thread counts can change floating-point sums. Small
differences can affect the fitted acquisition rules and subsequent random draws, so exact
agreement across runtimes is not guaranteed. This is particularly relevant when `c >= 100`,
where the politeness study's covariate-agnostic criterion is nearly flat. Compare interval
lengths, coverage and calibration diagnostics as well as the learned rule when validating
a fresh run against the archived results.

---

## Notes for reproducers

- **Output filenames.** `plot.py` writes `{prefix}_vs_tau_c{value}.pdf`, so one run per
  cost ratio does not overwrite the previous one. Pass `--prefix politeness_ci_cov` or
  `--prefix lm_ci_cov` to get the names the paper uses.

- **Per-cell seeding.** Every `(tau, c)` cell reseeds to `--seed` before drawing the
  stage-1/stage-2 split (`politeness.py:121`), so splitting the budget grid across
  separate invocations reproduces the combined run exactly.

- **Under `torchrun` the tau grid is split across processes.** `torchrun` sets
  `WORLD_SIZE`, which turns on distributed mode by itself (`lm_simulation.py:158`), so
  `--distributed` is not needed; this is why the archives' `params.json` records
  `"distributed": false`. Rank `r` of `w` runs
  `tau_values[r::w]` (`lm_simulation.py:106`) and rank 0 gathers every rank's rows, so
  `summary.csv` always holds the full grid. Each tau is run with the same `--seed`
  regardless of rank, which is why the serial and distributed forms agree. The `cmd`
  field in `params.json` is the script's own `sys.argv`, so it omits the `torchrun`
  launcher and its options; use the commands above.

---

## Controlled study

The controlled study is a self-contained NumPy implementation of the bounded-calibration
algorithms in a two-stratum setting, used to examine when acquiring preferences is
useful. It shares no code with the two experiments above.

**Estimand and observations.** The estimand is the marginal mean of `Y`, equal to zero.
The two strata are equally likely. `W1 = 1` and `W2 = -1`, so the observed
reference-induced preference is `V = 1{Y > 0}`. Conditional on stratum `x`, an independent
random sign multiplies a positive two-point magnitude `T` with mean `sqrt(h[x])` and
variance `g[x]`, so the oracle information components are exactly `g[x]` and `h[x]`. The
stratum indicator is an auxiliary feature for acquisition and calibration, not a regressor
in the target. Every method uses the known stratum probabilities to enforce the expected
per-candidate budget of 0.6, with gold cost 3 and preference cost 1; realized costs
fluctuate, and the initial sample's cost is excluded.

**What is fitted.** Nuisance functions are saturated regressions on (stratum, preference)
and on stratum. Stage 1 fits with unit weights on the first half of the initial data and
evaluates the variance criterion, or the plug-in residual moments, on the second half.
Stage 2 uses inverse-probability-weighted complete-case nuisance fits and base estimates,
with three independent fold roles for nuisance fitting, bounded scalar calibration and
evaluation; only acquired labels enter each fit. The regression Gram eigenvalue safeguard is
0.001; it is not a regularization of the calibration covariance.

**Comparators.**

- *PCAL-grid*: exact search over the 8447 feasible two-stratum designs of the fixed
  probability grid.
- *Plug-in*: estimated `g, h` substituted into the continuous clipped oracle rule, with
  `g` floored at 1e-8; a scalar multiplier enforces the known population budget.
- *Gold-grid*: the same criterion and grid, restricted to no preference acquisition.
- *Constant-grid*: the same criterion and grid, restricted to stratum-constant rules.
- *Oracle-gold*: the optimal continuous gold-only rule for the true `g, h`, with fitted
  Stage 2 nuisance functions and inference.

All methods use the same bounded-calibration inference and radius, and within a
replication they share the initial sample, the potential Stage 2 outcomes and the
acquisition uniforms; the three radius runs share them as well. A replication's seed is a
deterministic function of the main seed, the regime and the replication index.
`metadata.json` records both the unrestricted and the finite-grid oracle values, because
the plug-in and oracle-gold rules are not restricted to the grid.

**Reporting cautions.** The empirical root-n variance is `n1` times the Monte Carlo
variance of the estimates, and its Monte Carlo standard error is estimated from the fourth
central moment; the coverage standard error is that of the binary coverage indicator.
Neither is a model-based standard error for the target. The true efficient risk is an
oracle diagnostic of a learned design, not the finite-sample variance of fitted
inference. Results at a single initial sample size are not a learning-rate experiment,
and the study, being correctly specified for the saturated auxiliary fits, does not test
arbitrary nuisance misspecification or noisy human preferences.

---

## Data

`data/polite_scores_with_ds.csv` contains 5515 online requests from Stack Exchange and
Wikipedia. Each row carries the human politeness score averaged over five evaluators
(`Normalized Score`), 21 linguistic features, and AI-predicted politeness scores from
GPT-4o-mini (`gpt_score`) and DeepSeek-V3.1 (`predicted_score`). The experiments use
features 3 and 10 as covariates and all 5515 rows: 1500 for stage 1 and 3000 drawn per
replication from the remaining 4015. The comparative preference label is derived at load
time as `V = 1{|W1 - Y| <= |W2 - Y|}`, not stored in the file.

The file is a derivative of the **Stanford Politeness Corpus** of Danescu-Niculescu-Mizil
et al. (2013), distributed via [ConvoKit](https://convokit.cornell.edu/documentation/wiki_politeness.html)
under CC BY 4.0. We restricted it to the cleaned 5515-request subset of Ji et al. (2025)
and added the two AI-score columns. See `LICENSE-DATA` for the full attribution and the
list of changes that CC BY 4.0 requires.

---

## License

Two licenses apply, because the code and the data have different provenance:

| | License | File |
|---|---|---|
| Source code (`*.py`, `README.md`) | MIT | `LICENSE` |
| `data/polite_scores_with_ds.csv` | CC BY 4.0, inherited from the original corpus | `LICENSE-DATA` |

Archived runs under `results/` and `controlled/` are outputs of the code and fall under MIT.
