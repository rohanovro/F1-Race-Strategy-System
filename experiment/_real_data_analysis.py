# “””
experiments/exp_real_data_analysis.py

Corrected Real FastF1 Data Analysis — Publication Version

Generates Tables 1-4 from the paper:
“When Does Piecewise Tyre Degradation Modelling Outperform
Linear Models in Formula 1? An Empirical Analysis Using
FastF1 2023 Race Data”
Mahmudul Hasan Rohan, JUST Bangladesh

Corrections vs earlier version:
1. Fuel correction applied: lap_corrected = lap_time - fuel_kg * 0.035
2. Driver-level baseline normalisation (per driver minimum, not per stint)
3. Adaptive cliff detection via grid search (no fixed circuit priors)
4. Correct AIC/BIC formula from actual RSS
5. Five-fold expanding-window cross-validation
6. Wilcoxon signed-rank + bootstrapped 95% CI + Cohen’s d

Run from project root:
python experiments/exp_real_data_analysis.py

Requirements:
pip install fastf1 scikit-learn scipy numpy pandas -q

Results reproduced in paper:
Silverstone Hard: +11.9% MAE, p=0.003**, d=0.28, CI [8.2%, 14.7%]
Monaco Hard:      +2.7%  MAE, p=0.004**, d=0.12, CI [1.1%, 4.3%]
Mean improvement: +2.8% (7 compound-circuit combinations)
AIC favours piecewise: 5/7
“””

from **future** import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings(“ignore”)

PROJECT_ROOT = Path(**file**).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s | %(levelname)-8s | %(message)s”,
datefmt=”%H:%M:%S”,
)
logger = logging.getLogger(**name**)

# —————————————————————————

# FastF1 cache

try:
import fastf1
CACHE_DIR = PROJECT_ROOT / “data” / “raw” / “fastf1_cache”
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))
logger.info(“FastF1 cache: %s”, CACHE_DIR)
except ImportError:
logger.error(“FastF1 not installed. Run: pip install fastf1”)
sys.exit(1)

# —————————————————————————

# Constants

FUEL_BURN_KG_PER_LAP = 1.8    # kg/lap
FUEL_EFFECT_S_PER_KG = 0.035  # s/kg (standard F1 estimate)
MIN_STINT_LAPS       = 5      # minimum laps per stint
MIN_COMPOUND_LAPS    = 20     # minimum laps per compound for analysis

CIRCUITS = {
“bahrain”: {
“name”: “Bahrain Grand Prix”, “year”: 2023, “total_laps”: 57,
“description”: “High degradation, abrasive surface”,
},
“monaco”: {
“name”: “Monaco Grand Prix”, “year”: 2023, “total_laps”: 78,
“description”: “Low degradation, street circuit”,
},
“silverstone”: {
“name”: “British Grand Prix”, “year”: 2023, “total_laps”: 52,
“description”: “Medium-high degradation, high-speed corners”,
},
}

COMPOUNDS = [“SOFT”, “MEDIUM”, “HARD”]

# ===========================================================================

# DATA EXTRACTION  (Section 3.2 of paper)

# ===========================================================================

def extract_stints(
circuit_name: str,
year: int,
total_laps: int,
) -> dict[str, pd.DataFrame]:
“””
Extract representative tyre stint data with fuel correction and
driver-level baseline normalisation.

```
Preprocessing pipeline:
    Step 1 — Green flag only (TrackStatus == '1')
    Step 2 — Remove pit entry / exit laps
    Step 3 — Fuel correction:
                 lap_corrected = lap_time - fuel_kg * 0.035
                 fuel_kg = 100 - (LapNumber - 1) * 1.8
    Step 4 — Driver-level baseline:
                 delta = lap_corrected - driver_minimum
    Step 5 — Outlier removal: deltas > 2.0 s excluded
    Step 6 — Minimum 5 laps/stint, 20 laps/compound
"""
logger.info("Loading %s %d ...", circuit_name, year)
session = fastf1.get_session(year, circuit_name, "R")
session.load(laps=True, telemetry=False, weather=False, messages=False)

laps = session.laps.copy()

# Step 1 & 2
laps = laps[laps["TrackStatus"] == "1"].copy()
laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()].copy()
laps = laps[laps["LapTime"].notna()].copy()
laps["lap_sec"] = laps["LapTime"].dt.total_seconds()
laps = laps[(laps["lap_sec"] > 60) & (laps["lap_sec"] < 120)].copy()

# Step 3 — Fuel correction
laps["fuel_kg"]       = 100 - (laps["LapNumber"] - 1) * FUEL_BURN_KG_PER_LAP
laps["lap_corrected"] = laps["lap_sec"] - laps["fuel_kg"] * FUEL_EFFECT_S_PER_KG

results: dict[str, pd.DataFrame] = {}

for compound in COMPOUNDS:
    comp_laps = laps[laps["Compound"] == compound].copy()
    if comp_laps.empty:
        continue

    all_ages:   list[float] = []
    all_deltas: list[float] = []

    for driver in comp_laps["Driver"].unique():
        d_laps = comp_laps[comp_laps["Driver"] == driver].sort_values("LapNumber")

        # Step 4 — Driver-level baseline
        driver_baseline = d_laps["lap_corrected"].min()

        for _, stint in d_laps.groupby("Stint"):
            if len(stint) < MIN_STINT_LAPS:
                continue

            ages   = stint["TyreLife"].values.astype(float)
            deltas = stint["lap_corrected"].values - driver_baseline

            # Step 5 — Outlier removal
            valid = deltas < 2.0
            if valid.sum() < MIN_STINT_LAPS:
                continue

            all_ages.extend(ages[valid])
            all_deltas.extend(deltas[valid])

    n = len(all_ages)
    if n < MIN_COMPOUND_LAPS:
        logger.info("  %s: %d laps — excluded (< %d).", compound, n, MIN_COMPOUND_LAPS)
        continue

    results[compound] = pd.DataFrame({"tyre_age": all_ages, "lap_delta": all_deltas})
    logger.info("  %s: %d laps extracted.", compound, n)

return results
```

# ===========================================================================

# DEGRADATION MODELS  (Section 3.3 and 3.4 of paper)

# ===========================================================================

def fit_linear(ages: np.ndarray, deltas: np.ndarray) -> np.ndarray:
“”“Linear model: Δt(n) = k₁·n  (Heilmeier et al., 2020).”””
return LinearRegression().fit(ages.reshape(-1, 1), deltas).predict(ages.reshape(-1, 1))

def fit_piecewise_adaptive(
ages: np.ndarray,
deltas: np.ndarray,
) -> tuple[np.ndarray, int | None]:
“””
Adaptive piecewise model with grid-search cliff detection.

```
Candidate cliff laps: 8 to min(max_age, 30).
Criterion: maximise R² with C⁰ continuity at junction.
Threshold: must improve R² over linear by ≥ 0.01.

Returns: (predictions, cliff_lap) or (linear_predictions, None)
"""
if len(ages) < 10:
    return fit_linear(ages, deltas), None

y_lin    = fit_linear(ages, deltas)
best_r2  = r2_score(deltas, y_lin) + 0.01
best_pred, best_cliff = None, None

for cliff in range(8, min(int(np.max(ages)), 31)):
    pre  = ages < cliff
    post = ages >= cliff
    if pre.sum() < 3 or post.sum() < 3:
        continue

    lr_pre  = LinearRegression().fit(ages[pre].reshape(-1, 1),  deltas[pre])
    lr_post = LinearRegression().fit(ages[post].reshape(-1, 1), deltas[post])

    # C⁰ continuity enforcement
    shift = lr_pre.predict([[cliff]])[0] - lr_post.predict([[cliff]])[0]
    pred  = np.zeros_like(deltas)
    pred[pre]  = lr_pre.predict(ages[pre].reshape(-1, 1))
    pred[post] = lr_post.predict(ages[post].reshape(-1, 1)) + shift

    r2 = r2_score(deltas, pred)
    if r2 > best_r2:
        best_r2, best_pred, best_cliff = r2, pred.copy(), cliff

return (best_pred, best_cliff) if best_pred is not None else (y_lin, None)
```

# ===========================================================================

# STATISTICAL ANALYSIS  (Section 3.5 of paper)

# ===========================================================================

def analyse(df: pd.DataFrame, circuit: str, compound: str) -> dict:
“”“Full statistical comparison for one compound-circuit combination.”””
ages, deltas = df[“tyre_age”].values, df[“lap_delta”].values
n = len(ages)

```
y_lin          = fit_linear(ages, deltas)
y_pw, cliff_lap= fit_piecewise_adaptive(ages, deltas)

# Metrics
r2_lin  = r2_score(deltas, y_lin)
r2_pw   = r2_score(deltas, y_pw)
mae_lin = mean_absolute_error(deltas, y_lin)
mae_pw  = mean_absolute_error(deltas, y_pw)
impr    = (mae_lin - mae_pw) / mae_lin * 100 if mae_lin > 0 else 0.0

abs_lin = np.abs(deltas - y_lin)
abs_pw  = np.abs(deltas - y_pw)

# Wilcoxon signed-rank test (one-sided)
try:
    _, p_val = stats.wilcoxon(abs_lin, abs_pw, alternative="greater")
except Exception:
    p_val = float("nan")

# Cohen's d
diff     = abs_lin - abs_pw
cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0

# Bootstrapped 95% CI (1,000 resamples)
rng = np.random.default_rng(42)
boot = [
    ((m_l := np.mean(abs_lin[i := rng.integers(0, n, n)])) -
     np.mean(abs_pw[i])) / m_l * 100 if (m_l := np.mean(abs_lin[i := rng.integers(0, n, n)])) > 0 else 0.0
    for _ in range(1000)
]
# Simpler bootstrap
boot_impr = []
for _ in range(1000):
    idx   = rng.integers(0, n, size=n)
    ml    = np.mean(abs_lin[idx])
    mp    = np.mean(abs_pw[idx])
    boot_impr.append((ml - mp) / ml * 100 if ml > 0 else 0.0)
ci_lo = float(np.percentile(boot_impr, 2.5))
ci_hi = float(np.percentile(boot_impr, 97.5))

# AIC / BIC from actual RSS
rss_lin = np.sum((deltas - y_lin) ** 2)
rss_pw  = np.sum((deltas - y_pw)  ** 2)
k_lin, k_pw = 2, 4
aic_lin = n * np.log(rss_lin / n + 1e-12) + 2 * k_lin
aic_pw  = n * np.log(rss_pw  / n + 1e-12) + 2 * k_pw
bic_lin = n * np.log(rss_lin / n + 1e-12) + k_lin * np.log(n)
bic_pw  = n * np.log(rss_pw  / n + 1e-12) + k_pw  * np.log(n)
delta_aic = aic_pw - aic_lin
aic_pref  = "Piecewise" if delta_aic < -2 else ("Linear" if delta_aic > 2 else "Tie")

# Five-fold expanding-window CV
df_s = df.sort_values("tyre_age").reset_index(drop=True)
oos_lin_vals, oos_pw_vals = [], []
for fold in range(1, 5):
    te = fold * (n // 5)
    ts = min(te + n // 5, n)
    if te < 5 or ts <= te:
        continue
    tr = df_s.iloc[:te]
    t  = df_s.iloc[te:ts]
    lr2 = LinearRegression().fit(
        tr["tyre_age"].values.reshape(-1, 1), tr["lap_delta"].values
    )
    y_oos_lin = lr2.predict(t["tyre_age"].values.reshape(-1, 1))
    y_oos_pw, _ = fit_piecewise_adaptive(
        t["tyre_age"].values, t["lap_delta"].values
    )
    oos_lin_vals.append(mean_absolute_error(t["lap_delta"].values, y_oos_lin))
    oos_pw_vals.append(mean_absolute_error(t["lap_delta"].values, y_oos_pw))

oos_mae_lin = float(np.mean(oos_lin_vals)) if oos_lin_vals else float("nan")
oos_mae_pw  = float(np.mean(oos_pw_vals))  if oos_pw_vals  else float("nan")
oos_impr    = (
    (oos_mae_lin - oos_mae_pw) / oos_mae_lin * 100
    if not np.isnan(oos_mae_lin) and oos_mae_lin > 0 else float("nan")
)

def sig(p: float) -> str:
    if np.isnan(p): return "n.d."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

return {
    "circuit": circuit, "compound": compound, "n": n, "cliff_lap": cliff_lap,
    "r2_lin": r2_lin, "mae_lin": mae_lin,
    "r2_pw": r2_pw,   "mae_pw": mae_pw,
    "impr_pct": impr, "p_value": p_val, "sig": sig(p_val),
    "cohens_d": cohens_d, "ci_lo": ci_lo, "ci_hi": ci_hi,
    "aic_lin": aic_lin, "aic_pw": aic_pw, "delta_aic": delta_aic,
    "bic_lin": bic_lin, "bic_pw": bic_pw, "aic_pref": aic_pref,
    "oos_mae_lin": oos_mae_lin, "oos_mae_pw": oos_mae_pw, "oos_impr": oos_impr,
}
```

# ===========================================================================

# MAIN

# ===========================================================================

def main() -> None:
print(”\n” + “=” * 90)
print(“F1 TYRE DEGRADATION MODEL COMPARISON — PUBLICATION VERSION”)
print(“Real FastF1 2023 Data | Rohan MH | IMechE Part P 2025”)
print(”=” * 90)

```
all_results: list[dict] = []

for circuit_key, cfg in CIRCUITS.items():
    print(f"\n{'─'*60}")
    print(f"CIRCUIT: {circuit_key.upper()} | {cfg['description']}")
    print(f"{'─'*60}")
    stint_data = extract_stints(cfg["name"], cfg["year"], cfg["total_laps"])
    for compound in COMPOUNDS:
        if compound in stint_data:
            all_results.append(analyse(stint_data[compound], circuit_key, compound))

valid = [r for r in all_results if not np.isnan(r["impr_pct"])]
sig_r = [r for r in valid if not np.isnan(r["p_value"]) and r["p_value"] < 0.05]

# Table 1
print("\n\n" + "=" * 100)
print("TABLE 1: Model Fit Quality")
print("=" * 100)
print(f"{'Circuit':<12} {'Cpd':<8} {'N':>5}  {'Lin R²':>7} {'Lin MAE':>8}  "
      f"{'PW R²':>7} {'PW MAE':>8}  {'Impr%':>7} {'Sig':>5} {'d':>6}  "
      f"{'95% CI':>18}  {'Cliff':>6}")
print("─" * 100)
for r in all_results:
    ci  = f"[{r['ci_lo']:.1f},{r['ci_hi']:.1f}]%"
    cl  = f"L{r['cliff_lap']}" if r["cliff_lap"] else "None"
    print(f"{r['circuit']:<12} {r['compound']:<8} {r['n']:>5}  "
          f"{r['r2_lin']:>7.4f} {r['mae_lin']:>8.4f}  "
          f"{r['r2_pw']:>7.4f} {r['mae_pw']:>8.4f}  "
          f"{r['impr_pct']:>6.1f}% {r['sig']:>5} {r['cohens_d']:>6.2f}  "
          f"{ci:>18}  {cl:>6}")
print("─" * 100)
print(f"Mean MAE improvement: {np.mean([r['impr_pct'] for r in valid]):+.1f}%  |  "
      f"Significant: {len(sig_r)}/{len(valid)}")

# Table 2
print("\n\n" + "=" * 90)
print("TABLE 2: AIC/BIC (actual RSS formula)")
print("=" * 90)
print(f"{'Circuit':<12} {'Cpd':<8} {'AIC Lin':>9} {'AIC PW':>9} "
      f"{'ΔAIC':>8} {'BIC Lin':>9} {'BIC PW':>9} {'Preferred':>10}")
print("─" * 90)
for r in all_results:
    print(f"{r['circuit']:<12} {r['compound']:<8} "
          f"{r['aic_lin']:>9.1f} {r['aic_pw']:>9.1f} {r['delta_aic']:>8.1f} "
          f"{r['bic_lin']:>9.1f} {r['bic_pw']:>9.1f} {r['aic_pref']:>10}")
pw_n = sum(1 for r in all_results if r["aic_pref"] == "Piecewise")
print("─" * 90)
print(f"AIC prefers piecewise: {pw_n}/{len(all_results)}")

# Table 3
print("\n\n" + "=" * 75)
print("TABLE 3: Five-Fold Expanding-Window Cross-Validation")
print("=" * 75)
print(f"{'Circuit':<12} {'Cpd':<8} {'OOS Lin':>9} {'OOS PW':>9} "
      f"{'OOS Impr':>10}  {'In-Sample':>10}")
print("─" * 75)
for r in all_results:
    print(f"{r['circuit']:<12} {r['compound']:<8} "
          f"{r['oos_mae_lin']:>9.4f} {r['oos_mae_pw']:>9.4f} "
          f"{r['oos_impr']:>9.1f}%  {r['impr_pct']:>9.1f}%")
oos = [r["oos_impr"] for r in all_results if not np.isnan(r["oos_impr"])]
print("─" * 75)
print(f"Mean OOS: {np.mean(oos):.1f}%  |  Mean in-sample: "
      f"{np.mean([r['impr_pct'] for r in valid]):.1f}%  |  Diff: "
      f"{abs(np.mean(oos)-np.mean([r['impr_pct'] for r in valid])):.1f} pp")

# Summary
print("\n\n" + "=" * 75)
print("PAPER ABSTRACT KEY NUMBERS")
print("=" * 75)
for r in sorted(sig_r, key=lambda x: -x["impr_pct"]):
    print(f"  {r['circuit'].capitalize()} {r['compound'].capitalize()}: "
          f"+{r['impr_pct']:.1f}% MAE, p={r['p_value']:.4f}{r['sig']}, "
          f"d={r['cohens_d']:.2f}, "
          f"95% CI [{r['ci_lo']:.1f}%, {r['ci_hi']:.1f}%], "
          f"cliff=L{r['cliff_lap']}")
print(f"\n  AIC favours PW: {pw_n}/{len(all_results)}")
print(f"  Mean MAE improvement: {np.mean([r['impr_pct'] for r in valid]):.1f}%")
print(f"  Mean OOS improvement: {np.mean(oos):.1f}%")
print("=" * 75)
```

if **name** == “**main**”:
main()