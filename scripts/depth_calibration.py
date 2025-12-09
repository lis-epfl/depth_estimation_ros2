#!/usr/bin/env python3
"""
Depth Calibration Script for Stereo Pairs

Computes baseline_scale and offset_factor from measured depth data.
Edit the DATA section below with your measurements, then run the script.

Correction formula:
    Z_corrected = baseline_scale * Z_est / (1 + offset_factor * Z_est)
"""

import numpy as np

# =============================================================================
# DATA - EDIT THIS SECTION WITH YOUR MEASUREMENTS
# =============================================================================

# Ground truth distances (meters) - same for all pairs
Z_REAL = [3.75, 1.3, 0.77, 2.3]

# Estimated distances (meters) for each stereo pair [0, 0, 0, 0]
Z_EST = {
    "0_1": [4.0, 1.3, 0.76, 2.35],
    "1_2": [3.7, 1.24, 0.77, 2.28],
    "2_3": [4.0, 1.34, 0.82, 2.43],
    "3_0": [4.16, 1.3, 0.76, 2.44],
}

# =============================================================================
# PROCESSING - DO NOT EDIT BELOW THIS LINE
# =============================================================================

PAIR_ORDER = ["0_1", "1_2", "2_3", "3_0"]


def fit_depth_correction(z_real: np.ndarray, z_est: np.ndarray) -> tuple:
    """Fit depth correction parameters using least squares in inverse-depth space."""
    inv_real = 1.0 / z_real
    inv_est = 1.0 / z_est

    # Linear fit: inv_est = slope * inv_real + intercept
    slope, intercept = np.polyfit(inv_real, inv_est, 1)

    baseline_scale = slope
    offset_factor = -intercept

    # Compute R² for fit quality
    inv_est_pred = slope * inv_real + intercept
    ss_res = np.sum((inv_est - inv_est_pred) ** 2)
    ss_tot = np.sum((inv_est - np.mean(inv_est)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    # Compute corrected depths and errors
    z_corrected = baseline_scale * z_est / (1 + offset_factor * z_est)
    errors = np.abs(z_corrected - z_real)

    return baseline_scale, offset_factor, r_squared, z_corrected, errors


def main():
    z_real = np.array(Z_REAL)
    n_points = len(z_real)

    print("=" * 70)
    print("DEPTH CALIBRATION RESULTS")
    print("=" * 70)
    print(f"\nMeasurements: {n_points} distances")
    print(f"Z_real: {Z_REAL}")
    print()

    results = {}

    for pair in PAIR_ORDER:
        z_est = np.array(Z_EST[pair])

        if len(z_est) != n_points:
            print(f"ERROR: Pair {pair} has {len(z_est)} measurements, expected {n_points}")
            continue

        baseline_scale, offset_factor, r_squared, z_corrected, errors = fit_depth_correction(z_real, z_est)

        results[pair] = {
            'baseline_scale': baseline_scale,
            'offset_factor': offset_factor,
        }

        # Analyze ratio pattern
        ratios = z_real / z_est
        ratio_change = ratios[-1] - ratios[0]
        if abs(ratio_change) < 0.05 * np.mean(ratios):
            pattern = "constant"
        elif ratio_change > 0:
            pattern = "increasing"
        else:
            pattern = "decreasing"

        print(f"Pair {pair}:")
        print(f"  Ratios (Z_real/Z_est): {[f'{r:.3f}' for r in ratios]} -> {pattern}")
        if n_points > 2:
            print(f"  Fit R²: {r_squared:.6f}")
        print(f"  baseline_scale: {baseline_scale:.6f}")
        print(f"  offset_factor:  {offset_factor:.6f}")
        print(f"  Verification:")
        for i in range(n_points):
            print(f"    Z_real={z_real[i]:.2f} | Z_est={z_est[i]:.2f} | "
                  f"Z_corrected={z_corrected[i]:.4f} | error={errors[i]*1000:.1f}mm")
        print()

    # Output YAML
    print("=" * 70)
    print("YAML OUTPUT")
    print("=" * 70)

    baseline_scales = [round(float(results[p]['baseline_scale']), 6) for p in PAIR_ORDER]
    offset_factors = [round(float(results[p]['offset_factor']), 6) for p in PAIR_ORDER]

    print(f"""
# depth_corrections.yaml
# Correction: Z_corrected = baseline_scale * Z_est / (1 + offset_factor * Z_est)

depth_correction:
  baseline_scales: {baseline_scales}
  offset_factors: {offset_factors}
""")


if __name__ == "__main__":
    main()
