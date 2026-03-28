# ============================================================================
# CORRECTED F1 REAL DATA ANALYSIS — COMPLETE VERSION
# ============================================================================

# Step 1: Install packages
!pip install fastf1 scikit-learn scipy pandas numpy -q

# Step 2: Imports
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import fastf1
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

# Step 3: Create cache directory (FIXED)
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')
print("Cache enabled successfully")

# Step 4: Constants
FUEL_BURN_RATE = 1.8  # kg per lap
FUEL_EFFECT = 0.035   # seconds per kg

# Step 5: Circuit config
CIRCUITS = {
    "bahrain": {"name": "Bahrain Grand Prix", "year": 2023, "total_laps": 57},
    "monaco": {"name": "Monaco Grand Prix", "year": 2023, "total_laps": 78},
    "silverstone": {"name": "British Grand Prix", "year": 2023, "total_laps": 52},
}

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# ============================================================================
# Step 6: Data extraction with proper driver normalization
# ============================================================================

def extract_stints_corrected(circuit_name, year, total_laps):
    """Extract real stint data with proper driver normalization"""
    print(f"\nLoading {circuit_name} {year}...")
    session = fastf1.get_session(year, circuit_name, "R")
    session.load(laps=True, telemetry=False, weather=False)
    
    laps = session.laps.copy()
    
    # Filter to green flag laps only
    laps = laps[laps["TrackStatus"] == "1"].copy()
    
    # Remove pit laps
    laps = laps[~laps["PitInTime"].notna()].copy()
    laps = laps[~laps["PitOutTime"].notna()].copy()
    
    # Remove invalid laps
    laps = laps[laps["LapTime"].notna()].copy()
    laps["lap_time_sec"] = laps["LapTime"].dt.total_seconds()
    
    # Remove unrealistic lap times
    laps = laps[(laps["lap_time_sec"] > 60) & (laps["lap_time_sec"] < 120)].copy()
    
    # Fuel correction
    laps["fuel_kg"] = 100 - (laps["LapNumber"] - 1) * FUEL_BURN_RATE
    laps["fuel_kg"] = laps["fuel_kg"].clip(lower=0)
    laps["lap_time_fuel_corrected"] = laps["lap_time_sec"] - laps["fuel_kg"] * FUEL_EFFECT
    
    results = {}
    
    for compound in COMPOUNDS:
        comp_laps = laps[laps["Compound"] == compound].copy()
        if comp_laps.empty:
            print(f"  {compound}: No data")
            continue
        
        all_ages = []
        all_deltas = []
        
        # Process each driver separately
        for driver in comp_laps["Driver"].unique():
            driver_laps = comp_laps[comp_laps["Driver"] == driver].sort_values("LapNumber")
            
            # Get driver's fastest lap as baseline
            driver_baseline = driver_laps["lap_time_fuel_corrected"].min()
            
            # Group by stint
            for stint_num, stint in driver_laps.groupby("Stint"):
                if len(stint) < 5:
                    continue
                
                ages = stint["TyreLife"].values
                deltas = stint["lap_time_fuel_corrected"].values - driver_baseline
                
                # Remove outliers
                valid = deltas < 3.0
                if valid.sum() < 5:
                    continue
                
                all_ages.extend(ages[valid])
                all_deltas.extend(deltas[valid])
        
        if len(all_ages) >= 20:
            results[compound] = pd.DataFrame({
                "tyre_age": all_ages,
                "lap_delta": all_deltas
            })
            print(f"  {compound}: {len(all_ages)} laps extracted")
        else:
            print(f"  {compound}: Insufficient data ({len(all_ages)} laps)")
    
    return results

# ============================================================================
# Step 7: Model fitting functions
# ============================================================================

def fit_linear(ages, deltas):
    """Fit linear model"""
    lr = LinearRegression().fit(ages.reshape(-1, 1), deltas)
    return lr.predict(ages.reshape(-1, 1))

def fit_piecewise_adaptive(ages, deltas):
    """Fit adaptive piecewise model — detect cliff from data"""
    n = len(ages)
    if n < 10:
        return fit_linear(ages, deltas)
    
    best_r2 = -np.inf
    best_predictions = None
    
    # Test cliff at different points
    max_age = int(np.max(ages))
    test_cliffs = range(8, min(max_age, 30))
    
    for cliff in test_cliffs:
        pre_mask = ages < cliff
        post_mask = ages >= cliff
        
        if pre_mask.sum() < 3 or post_mask.sum() < 3:
            continue
        
        # Fit pre-cliff
        pre_ages = ages[pre_mask].reshape(-1, 1)
        pre_deltas = deltas[pre_mask]
        lr_pre = LinearRegression().fit(pre_ages, pre_deltas)
        
        # Fit post-cliff
        post_ages = ages[post_mask].reshape(-1, 1)
        post_deltas = deltas[post_mask]
        lr_post = LinearRegression().fit(post_ages, post_deltas)
        
        # Predict
        pred = np.zeros_like(deltas)
        pred[pre_mask] = lr_pre.predict(ages[pre_mask].reshape(-1, 1))
        pred[post_mask] = lr_post.predict(ages[post_mask].reshape(-1, 1))
        
        # Ensure continuity
        pre_at_cliff = lr_pre.predict(np.array([[cliff]]))[0]
        post_at_cliff = lr_post.predict(np.array([[cliff]]))[0]
        pred[post_mask] += (pre_at_cliff - post_at_cliff)
        
        r2 = r2_score(deltas, pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_predictions = pred
    
    if best_predictions is None:
        return fit_linear(ages, deltas)
    
    return best_predictions

# ============================================================================
# Step 8: Run analysis
# ============================================================================

print("\n" + "="*80)
print("CORRECTED F1 REAL DATA ANALYSIS")
print("="*80)

all_results = []

for circuit_key, cfg in CIRCUITS.items():
    print(f"\n{'='*80}")
    print(f"CIRCUIT: {circuit_key.upper()}")
    print("="*80)
    
    stint_data = extract_stints_corrected(cfg["name"], cfg["year"], cfg["total_laps"])
    
    for compound in COMPOUNDS:
        if compound not in stint_data:
            print(f"\n{compound}: NO DATA")
            continue
        
        df = stint_data[compound]
        ages = df["tyre_age"].values
        deltas = df["lap_delta"].values
        
        # Filter to reasonable ages
        valid = ages <= 40
        ages = ages[valid]
        deltas = deltas[valid]
        
        if len(ages) < 20:
            print(f"\n{compound}: Only {len(ages)} laps — insufficient")
            continue
        
        # Fit models
        y_lin = fit_linear(ages, deltas)
        y_pw = fit_piecewise_adaptive(ages, deltas)
        
        # Metrics
        mae_lin = mean_absolute_error(deltas, y_lin)
        mae_pw = mean_absolute_error(deltas, y_pw)
        r2_lin = r2_score(deltas, y_lin)
        r2_pw = r2_score(deltas, y_pw)
        
        improvement = ((mae_lin - mae_pw) / mae_lin) * 100
        
        # Statistical test
        abs_err_lin = np.abs(deltas - y_lin)
        abs_err_pw = np.abs(deltas - y_pw)
        if len(abs_err_lin) > 10:
            _, p_val = stats.wilcoxon(abs_err_lin, abs_err_pw, alternative="greater")
        else:
            p_val = float("nan")
        
        # Cohen's d
        diff = abs_err_lin - abs_err_pw
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        all_results.append({
            "circuit": circuit_key,
            "compound": compound,
            "n_laps": len(ages),
            "r2_lin": r2_lin,
            "mae_lin": mae_lin,
            "r2_pw": r2_pw,
            "mae_pw": mae_pw,
            "improvement": improvement,
            "p_value": p_val,
            "cohens_d": cohens_d,
        })
        
        print(f"\n{compound}:")
        print(f"  Laps: {len(ages)}")
        print(f"  Linear: R²={r2_lin:.4f}, MAE={mae_lin:.4f}s")
        print(f"  Piecewise: R²={r2_pw:.4f}, MAE={mae_pw:.4f}s")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  p-value: {p_val:.4f}" if not np.isnan(p_val) else "  p-value: N/A")
        print(f"  Cohen's d: {cohens_d:.2f}")

# ============================================================================
# Step 9: Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Circuit':<12} {'Compound':<8} {'N':>5} {'Lin R²':>8} {'Lin MAE':>9} {'PW R²':>8} {'PW MAE':>9} {'Impr%':>7} {'p-val':>8} {'d':>6}")
print("-"*80)

for r in all_results:
    p_str = f"{r['p_value']:.4f}" if not np.isnan(r['p_value']) else "N/A"
    print(f"{r['circuit']:<12} {r['compound']:<8} {r['n_laps']:>5} {r['r2_lin']:>8.4f} {r['mae_lin']:>9.4f} {r['r2_pw']:>8.4f} {r['mae_pw']:>9.4f} {r['improvement']:>6.1f}% {p_str:>8} {r['cohens_d']:>6.2f}")

if all_results:
    valid = [r for r in all_results if not np.isnan(r["improvement"])]
    print(f"\nMean improvement: {np.mean([r['improvement'] for r in valid]):.1f}%")
    sig = [r for r in valid if not np.isnan(r["p_value"]) and r["p_value"] < 0.05]
    print(f"Statistically significant (p<0.05): {len(sig)}/{len(valid)}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)