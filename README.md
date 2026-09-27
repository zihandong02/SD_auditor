# Preference-Calibrated Active Learning (PCAL)

Code and data for the numerical experiments in *Labels or Comparative Preferences?
Budget-Constrained Inference with AI-Generated Outputs*.

The repository contains the two experiments reported in the paper — a linear-regression
simulation and a semi-synthetic study on a politeness corpus — together with the
archived output of the runs behind every submitted figure.

---

## Requirements

**Python >= 3.11** (`numpy==2.3.1` does not install on 3.9 or 3.10).
A GPU is *not* required; both scripts accept `--device cpu` and fall back to CPU
automatically.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The archived results were produced on Linux x86_64 with Python 3.11.16 and the torch
2.5.1+cu124 wheel; `requirements.txt` records that in a comment. pip resolves the wheel
for your own platform, and the CPU wheel of the same torch version gives the same
numbers.

---

## Layout

```
lm_simulation.py        linear-regression simulation (Section 6.1)
politeness.py           politeness semi-synthetic study (Section 6.2)
plot.py                 plotting helper
src/
  data_generation.py    synthetic data generator and missingness mechanisms
  estimators.py         moment estimators, calibration matrix, acquisition search
  lm_mono_debias.py     experiment drivers and cross-fitted estimators
  models/               MLP nuisance and acquisition-rule networks
  utils/                seeding, device selection, Wald intervals, run dumping
data/
  polite_scores_with_ds.csv   politeness corpus with AI-predicted scores
results/                archived runs behind the submitted figures
```

Each run writes `results/<timestamp>/` containing `summary.csv` (one row per budget
level), `summary.txt`, `params.json` (the full argument set and command line) and a
figure.

---

## Reproducing the figures

Every plotted point in the submitted figures comes from the archived run listed below.

### Linear-regression simulation (Section 6.1)

```bash
# Figure 1 (main text, c = 10)          -> results/_20250809_181224
python lm_simulation.py --tau_vals 1,2,3,4,5,6,7,8,9 --reps 100

# Appendix, c = 5                        -> results/_20250809_172450
python lm_simulation.py --tau_vals 1,2,3,4 --c 5 --reps 100

# Appendix, c = 20                       -> results/_20250809_174749
python lm_simulation.py --tau_vals 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --c 20 --reps 100
```

These use the defaults `--n1 2000 --n2 20000 --sigma_eps 2.0 --alpha_level 0.1
--seed 42`. Note that `--sigma_eps 2.0` produces standard deviations 16, 4 and 6 for
`eps`, `eps_1` and `eps_2`; the multipliers 8, 2, 3 are fixed in
`src/data_generation.py`.

The archived runs were produced with `--distributed` across several processes. That
flag only splits the tau grid, so the serial commands above reproduce the same grid;
we checked that the two agree to the four decimals stored in `summary.csv`. To run the
distributed form, launch it under `torchrun`, which sets the environment variables
`torch.distributed` requires — the flag is then redundant:

```bash
torchrun --nproc_per_node=4 lm_simulation.py --tau_vals 1,2,3,4,5,6,7,8,9 --reps 100
```

### Politeness semi-synthetic study (Section 6.2)

```bash
# Figure 2 (main text, c = 100)          -> results/_20260916_190200
python politeness.py --tau_vals 35,40,45,50,55,60,65,70,75,80 --c_vals 100

# Appendix, c = 20                       -> results/_20260916_190000
python politeness.py --tau_vals 10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15 --c_vals 20

# Appendix, c = 50                       -> results/_20260916_190100
python politeness.py --tau_vals 25,27,29,31,33,35,37,39,41 --c_vals 50

# Appendix, c = 200                      -> results/_20260916_190300
python politeness.py --tau_vals 60,65,70,75,80,85,90,95,100 --c_vals 200
```

These use the defaults `--n1 1500 --n2_per_rep 3000 --n_total 5515 --reps 100
--alpha_init 1.0,0.0,0.0 --alpha_level 0.1 --seed 43`, so only the budget grid and the
cost ratio have to be given. The default `--alpha_init 1.0,0.0,0.0` leaves the stage-1
sample fully observed; the acquisition rule is learned from that sample and then reused
across the 100 replications, so the reported rule is one draw rather than an average
over pilots.

A quick check that the installation works, without waiting for a full run:

```bash
python lm_simulation.py --device cpu --reps 1 --tau_vals 3 --n1 200 --n2 400
python politeness.py    --device cpu --reps 2 --tau_vals 50 --c_vals 100
```

---

## Reading the output

`summary.csv` reports, for each budget level `tau`, the mean L2 error, mean confidence
interval length and empirical coverage of the four methods. The column suffixes predate
the paper's terminology; `plot.py` now labels the curves as the paper does:

| CSV suffix | Legend label | Paper |
|---|---|---|
| `_mar` | `PCAL` | PCAL (covariate-aware) |
| `_opt` | `PCAL-CA` | PCAL-CA (covariate-agnostic) |
| `_base` | `label+unlabeled` | label+unlabeled |
| `_ols` | `label-only` | label-only |

`alpha_opt` is the learned covariate-agnostic acquisition rule
`(alpha_1, alpha_2, alpha_3)`; `cov00_opt` is the value of the design criterion at that
rule. Because inference targets the first coordinate of `theta`, the criterion minimised
is the asymptotic variance of that coordinate.

---

## Notes for reproducers

- **Output filenames.** `plot.py` writes `{prefix}_vs_tau_c{value}.pdf`, so one run per
  cost ratio does not overwrite the previous one. Pass `--prefix politeness_ci_cov` to
  get the names the paper uses.

- **Per-cell seeding.** Every `(tau, c)` cell reseeds to `--seed` before drawing the
  stage-1/stage-2 split (`politeness.py:118`), so splitting the budget grid across
  separate invocations reproduces the combined run exactly.

- **`--distributed` splits the tau grid across processes** — rank `r` of `w` runs
  `tau_values[r::w]` (`lm_simulation.py:102`) and rank 0 gathers every rank's rows, so
  `summary.csv` always holds the full grid. Each tau is run with the same `--seed`
  regardless of rank, which is why the serial and distributed forms agree. The `cmd`
  field in `params.json` records only the launching process's arguments; use the
  commands above.

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

Archived runs under `results/` are outputs of the code and fall under MIT.
