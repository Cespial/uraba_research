"""
04b_olofsson_estimators.py
==========================
Implements Olofsson et al. (2014) stratified area estimators with 95% CIs.
Replaces Kappa with Quantity Disagreement (QD) and Allocation Disagreement (AD)
per Pontius & Millones (2011).

Reads: outputs/phase3_stats/classification_metrics.json
Writes: outputs/phase3_stats/olofsson_area_estimates.json

No GEE dependency — pure numpy computation on existing confusion matrices.

CM Indexing Note:
    The 7x7 confusion matrices use indices 0-6 where:
      idx 0 = background/placeholder (always zeros)
      idx 1 = class 1 BDen (Bosque denso)
      idx 2 = class 2 BSec (Bosque secundario)
      idx 3 = class 3 Past (Pasturas)
      idx 4 = class 4 Cult (Cultivos — always zero area)
      idx 5 = class 6 Urb  (Urbano)
      idx 6 = class 7 Suel (Suelo desnudo — always zero area)
    Class 5 (Agua) is post-processed from JRC water mask and NOT in the CM.

    Rows = reference classes; Columns = mapped (predicted) classes.
    For Olofsson we need rows=mapped strata → TRANSPOSE.

References:
    Olofsson, P. et al. (2014). Good practices for estimating area and
        assessing accuracy of land change. RSE, 148, 42–57.
    Pontius, R.G. & Millones, M. (2011). Death to Kappa: Birth of quantity
        disagreement and allocation disagreement. IJRS, 32(15), 4407–4429.
"""

import os
import json
import numpy as np
from datetime import datetime


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_AREA_HA = 3665888.0  # 30 municipalities, ~36,817 km²

# CM indices for RF-classified classes with non-zero mapped area
# BDen(idx 1), BSec(idx 2), Past(idx 3), Urb(idx 5)
CM_ACTIVE_INDICES = [1, 2, 3, 5]

# Class IDs in the LULC scheme → CM indices
# Class 1 BDen → CM idx 1
# Class 2 BSec → CM idx 2
# Class 3 Past → CM idx 3
# Class 5 Agua → NOT in CM (post-processed)
# Class 6 Urb  → CM idx 5
CM_IDX_TO_CLASS = {1: 1, 2: 2, 3: 3, 5: 6}

# For Olofsson output, these are the 5 reported classes
REPORT_CLASSES = [1, 2, 3, 5, 6]
CLASS_NAMES = {
    1: 'Bosque denso',
    2: 'Bosque secundario',
    3: 'Pasturas',
    5: 'Agua',
    6: 'Urbano',
}

# CM index for Suel (class 7) — will be merged into Pasturas
CM_SUEL_IDX = 6

PERIOD_KEYS = ['pre_acuerdo', 'transicion', 'post_acuerdo_1', 'post_acuerdo_2']


# ============================================================
# CONFUSION MATRIX EXTRACTION
# ============================================================

def extract_olofsson_matrix(cm_7x7, class_areas_ha):
    """
    Extract 4x4 confusion matrix for Olofsson estimation.

    Steps:
    1. Extract columns for mapped strata (CM indices 1,2,3,5) from all 7 rows
    2. Transpose to get rows=mapped strata
    3. Merge Suel reference (idx 6) into Pasturas reference (idx 3)
    4. Drop background (idx 0) and Cult (idx 4) reference (always zero)
    5. Return 4x4 matrix with rows=mapped, cols=reference

    Args:
        cm_7x7: 7x7 numpy array (rows=reference, cols=mapped)
        class_areas_ha: dict with mapped areas per class

    Returns:
        cm_4x4: (4,4) array, rows=mapped {BDen,BSec,Past,Urb}, cols=ref same
        mapped_areas: (4,) ha for {BDen, BSec, Past, Urb}
        agua_area: ha for Agua (deterministic, not in Olofsson)
    """
    cm = np.array(cm_7x7, dtype=float)

    # Step 1: Extract columns for mapped strata and transpose
    # Columns are mapped (predicted) classes in original CM
    # After transpose, these become rows (mapped strata)
    # Use ALL 7 reference rows initially to preserve sample counts

    mapped_idx = CM_ACTIVE_INDICES  # [1, 2, 3, 5]
    # For each mapped stratum, get its column from the original CM
    # Then the result after transpose = rows=mapped, cols=all 7 ref classes
    cm_mapped = cm[:, mapped_idx].T  # shape (4, 7)

    # Step 2: Build 4x4 by selecting reference columns and merging Suel→Past
    ref_idx_active = [1, 2, 3, 5]  # BDen, BSec, Past, Urb in ref
    cm_4x4 = cm_mapped[:, ref_idx_active].copy()  # (4, 4)

    # Add Suel reference (idx 6) to Pasturas reference (column index 2 in our 4x4)
    suel_col = cm_mapped[:, CM_SUEL_IDX]
    cm_4x4[:, 2] += suel_col  # merge Suel ref into Past ref

    # Mapped areas (ha)
    mapped_areas = np.array([
        class_areas_ha['1']['area_ha'],   # BDen
        class_areas_ha['2']['area_ha'],   # BSec
        class_areas_ha['3']['area_ha'],   # Past
        class_areas_ha['6']['area_ha'],   # Urb
    ])

    agua_area = class_areas_ha['5']['area_ha']

    return cm_4x4, mapped_areas, agua_area


# ============================================================
# OLOFSSON AREA ESTIMATORS
# ============================================================

def olofsson_area_estimates(cm, mapped_areas, total_area):
    """
    Compute Olofsson et al. (2014) unbiased area estimates with 95% CIs.

    Args:
        cm: (k,k) confusion matrix, rows=mapped strata, cols=reference classes
        mapped_areas: (k,) mapped area per stratum (ha)
        total_area: total area covered by these strata (ha)

    Returns:
        dict with area estimates, CIs, adjusted accuracies
    """
    k = cm.shape[0]

    # Proportional weights (mapped area proportions)
    W = mapped_areas / total_area

    # Sample sizes per stratum (row sums)
    n_i = cm.sum(axis=1)

    # Proportions: p_ij = n_ij / n_i
    p_ij = np.zeros((k, k), dtype=float)
    for i in range(k):
        if n_i[i] > 0:
            p_ij[i, :] = cm[i, :] / n_i[i]

    # --- Unbiased area proportions (Eq. 9) ---
    area_proportions = np.zeros(k)
    for j in range(k):
        area_proportions[j] = np.sum(W * p_ij[:, j])

    area_estimates = total_area * area_proportions

    # --- Variance of area estimates (Eq. 10) ---
    area_var = np.zeros(k)
    for j in range(k):
        s = 0.0
        for i in range(k):
            if n_i[i] > 1:
                s += (W[i] ** 2) * p_ij[i, j] * (1 - p_ij[i, j]) / (n_i[i] - 1)
        area_var[j] = (total_area ** 2) * s

    area_se = np.sqrt(area_var)
    area_ci95 = 1.96 * area_se

    # --- Adjusted Overall Accuracy (Eq. 1) ---
    adjusted_oa = np.sum(W * np.diag(p_ij))

    # Variance of OA
    oa_var = 0.0
    for i in range(k):
        if n_i[i] > 1:
            oa_var += (W[i] ** 2) * p_ij[i, i] * (1 - p_ij[i, i]) / (n_i[i] - 1)

    # --- User's Accuracy (Eq. 2): UA_i = p_ii ---
    adjusted_ua = np.diag(p_ij).copy()
    ua_se = np.zeros(k)
    for i in range(k):
        if n_i[i] > 1:
            ua_se[i] = np.sqrt(adjusted_ua[i] * (1 - adjusted_ua[i]) / (n_i[i] - 1))

    # --- Producer's Accuracy (Eq. 3): PA_j = W_j * p_jj / p̂_·j ---
    adjusted_pa = np.zeros(k)
    pa_se = np.zeros(k)
    for j in range(k):
        if area_proportions[j] > 0:
            adjusted_pa[j] = W[j] * p_ij[j, j] / area_proportions[j]

            # SE of PA (simplified Eq. 6)
            N_j_hat = area_proportions[j]
            v = 0.0
            for i in range(k):
                if n_i[i] > 1:
                    v += (W[i] ** 2) * p_ij[i, j] * (1 - p_ij[i, j]) / (n_i[i] - 1)
            pa_se[j] = np.sqrt(v) / N_j_hat if N_j_hat > 0 else 0

    return {
        'area_estimates_ha': area_estimates.tolist(),
        'area_se_ha': area_se.tolist(),
        'area_ci95_ha': area_ci95.tolist(),
        'area_proportions': area_proportions.tolist(),
        'adjusted_oa': float(adjusted_oa),
        'oa_se': float(np.sqrt(oa_var)),
        'adjusted_ua': adjusted_ua.tolist(),
        'ua_se': ua_se.tolist(),
        'adjusted_pa': adjusted_pa.tolist(),
        'pa_se': pa_se.tolist(),
        'mapped_areas_ha': mapped_areas.tolist(),
        'W_proportions': W.tolist(),
        'n_samples_per_stratum': n_i.tolist(),
    }


# ============================================================
# QUANTITY AND ALLOCATION DISAGREEMENT
# ============================================================

def compute_qd_ad(cm, mapped_areas, total_area):
    """
    Compute Quantity Disagreement (QD) and Allocation Disagreement (AD)
    per Pontius & Millones (2011), using area-weighted proportions.

    Args:
        cm: (k,k) confusion matrix, rows=mapped, cols=reference
        mapped_areas: (k,) mapped areas (ha)
        total_area: total area (ha)

    Returns:
        dict with QD, AD, overall_disagreement
    """
    k = cm.shape[0]
    W = mapped_areas / total_area
    n_i = cm.sum(axis=1)

    # Proportions within strata
    p_ij = np.zeros((k, k), dtype=float)
    for i in range(k):
        if n_i[i] > 0:
            p_ij[i, :] = cm[i, :] / n_i[i]

    # Population proportion matrix P_ij = W_i * p_ij
    P = np.zeros((k, k))
    for i in range(k):
        P[i, :] = W[i] * p_ij[i, :]

    # Marginals
    q_mapped = P.sum(axis=1)   # row marginals ≈ W
    q_ref = P.sum(axis=0)      # col marginals = adjusted area proportions

    # Per-category QD and AD
    qd_per_class = np.zeros(k)
    ad_per_class = np.zeros(k)

    for j in range(k):
        qd_per_class[j] = abs(q_mapped[j] - q_ref[j])

        commission_j = q_mapped[j] - P[j, j]
        omission_j = q_ref[j] - P[j, j]
        ad_per_class[j] = 2 * min(commission_j, omission_j)

    QD = 0.5 * np.sum(qd_per_class)
    AD = 0.5 * np.sum(ad_per_class)

    return {
        'quantity_disagreement': float(QD),
        'allocation_disagreement': float(AD),
        'overall_disagreement': float(QD + AD),
        'qd_per_class': qd_per_class.tolist(),
        'ad_per_class': ad_per_class.tolist(),
        'overall_agreement': float(1 - QD - AD),
    }


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_all_periods(metrics_path, output_path):
    """Process all 4 periods and generate Olofsson estimates."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    results = {
        'metadata': {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'method': 'Olofsson et al. (2014) stratified area estimators',
            'total_area_ha': TOTAL_AREA_HA,
            'reported_classes': REPORT_CLASSES,
            'class_names': CLASS_NAMES,
            'olofsson_strata': ['BDen', 'BSec', 'Past', 'Urb'],
            'note_agua': 'Agua (class 5) post-processed from JRC water mask; reported as deterministic area',
            'note_suel_merge': 'Suel (class 7, zero mapped area) reference samples merged into Pasturas',
            'note_cm_indexing': 'CM indices: 0=background, 1=BDen, 2=BSec, 3=Past, 4=Cult, 5=Urb, 6=Suel',
            'note_transposition': 'CM stored as rows=reference, transposed to rows=mapped for Olofsson',
            'disagreement_method': 'Pontius & Millones (2011)',
        },
        'periods': {},
    }

    for period_key in PERIOD_KEYS:
        period_data = metrics[period_key]
        cm_7x7 = period_data['confusion_matrix']
        class_areas = period_data['class_areas_ha']

        # Extract 4x4 matrix (transposed, Suel merged into Past)
        cm_4x4, mapped_areas, agua_area = extract_olofsson_matrix(
            cm_7x7, class_areas
        )

        # Total area for Olofsson = study area minus Agua
        olofsson_total = TOTAL_AREA_HA - agua_area

        # Olofsson area estimates
        olof = olofsson_area_estimates(cm_4x4, mapped_areas, olofsson_total)

        # QD and AD
        qd_ad = compute_qd_ad(cm_4x4, mapped_areas, olofsson_total)

        # Per-class summary (4 RF-classified classes)
        olof_class_names = ['Bosque denso', 'Bosque secundario', 'Pasturas', 'Urbano']
        olof_class_ids = [1, 2, 3, 6]
        per_class = {}

        for idx, (cls_id, cls_name) in enumerate(zip(olof_class_ids, olof_class_names)):
            per_class[str(cls_id)] = {
                'name': cls_name,
                'mapped_area_ha': round(float(mapped_areas[idx]), 0),
                'adjusted_area_ha': round(float(olof['area_estimates_ha'][idx]), 0),
                'area_se_ha': round(float(olof['area_se_ha'][idx]), 0),
                'area_ci95_ha': round(float(olof['area_ci95_ha'][idx]), 0),
                'area_lower_ha': round(float(olof['area_estimates_ha'][idx] - olof['area_ci95_ha'][idx]), 0),
                'area_upper_ha': round(float(olof['area_estimates_ha'][idx] + olof['area_ci95_ha'][idx]), 0),
                'adjusted_ua': round(float(olof['adjusted_ua'][idx]), 4),
                'ua_se': round(float(olof['ua_se'][idx]), 4),
                'adjusted_pa': round(float(olof['adjusted_pa'][idx]), 4),
                'pa_se': round(float(olof['pa_se'][idx]), 4),
            }

        # Add Agua as deterministic class
        per_class['5'] = {
            'name': 'Agua',
            'mapped_area_ha': round(agua_area, 0),
            'adjusted_area_ha': round(agua_area, 0),
            'area_se_ha': 0,
            'area_ci95_ha': 0,
            'area_lower_ha': round(agua_area, 0),
            'area_upper_ha': round(agua_area, 0),
            'adjusted_ua': 1.0,
            'ua_se': 0.0,
            'adjusted_pa': 1.0,
            'pa_se': 0.0,
            'note': 'Post-processed from JRC water mask; not classified by RF',
        }

        # Total area check (all 5 classes)
        total_adj = sum(olof['area_estimates_ha']) + agua_area

        results['periods'][period_key] = {
            'year': period_data['year'],
            'label': period_data['label'],
            'n_validation_olofsson': int(sum(olof['n_samples_per_stratum'])),
            'n_validation_original': period_data['n_validation'],
            'pixel_counted_oa': period_data['overall_accuracy'],
            'pixel_counted_kappa': period_data['kappa'],
            'adjusted_oa': round(olof['adjusted_oa'], 4),
            'oa_se': round(olof['oa_se'], 4),
            'quantity_disagreement': round(qd_ad['quantity_disagreement'], 4),
            'allocation_disagreement': round(qd_ad['allocation_disagreement'], 4),
            'overall_disagreement': round(qd_ad['overall_disagreement'], 4),
            'agua_area_ha': round(agua_area, 0),
            'total_adjusted_area_ha': round(total_adj, 0),
            'area_sum_check_pct': round(total_adj / TOTAL_AREA_HA * 100, 2),
            'confusion_matrix_4x4_mapped_rows': cm_4x4.astype(int).tolist(),
            'per_class': per_class,
        }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results):
    """Print a human-readable summary."""
    print("=" * 80)
    print("OLOFSSON AREA ESTIMATES — SUMMARY")
    print("=" * 80)

    for period_key in PERIOD_KEYS:
        p = results['periods'][period_key]
        print(f"\n{p['label']} ({p['year']})")
        print(f"  Pixel-counted OA: {p['pixel_counted_oa']:.1%}  |  Kappa: {p['pixel_counted_kappa']:.3f}")
        print(f"  Adjusted OA:      {p['adjusted_oa']:.1%} ± {p['oa_se']:.1%}")
        print(f"  QD: {p['quantity_disagreement']:.4f}  |  AD: {p['allocation_disagreement']:.4f}")
        print(f"  Total area check: {p['area_sum_check_pct']:.1f}%  (n_val={p['n_validation_olofsson']})")
        print()
        header = f"  {'Class':<20} {'Mapped':>12} {'Adjusted':>12} {'± 95%CI':>12} {'UA':>8} {'PA':>8}"
        print(header)
        print(f"  {'-'*74}")
        for cls_id in REPORT_CLASSES:
            c = p['per_class'][str(cls_id)]
            ci = f"± {c['area_ci95_ha']:>9,.0f}" if c['area_ci95_ha'] > 0 else "  (determ.)"
            print(f"  {c['name']:<20} {c['mapped_area_ha']:>12,.0f} "
                  f"{c['adjusted_area_ha']:>12,.0f} {ci:>12} "
                  f"{c['adjusted_ua']:>8.3f} {c['adjusted_pa']:>8.3f}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, 'outputs', 'phase3_stats', 'classification_metrics.json')
    output_path = os.path.join(base_dir, 'outputs', 'phase3_stats', 'olofsson_area_estimates.json')

    print(f"Reading: {metrics_path}")
    results = process_all_periods(metrics_path, output_path)
    print(f"Written: {output_path}\n")

    print_summary(results)

    # Verification
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)
    for pk in PERIOD_KEYS:
        p = results['periods'][pk]
        qd, ad, oa = p['quantity_disagreement'], p['allocation_disagreement'], p['adjusted_oa']
        check = abs((qd + ad) - (1 - oa))
        print(f"  {pk}: sum={p['total_adjusted_area_ha']:,.0f}ha ({p['area_sum_check_pct']:.1f}%); "
              f"QD+AD={qd+ad:.4f} vs 1-OA={1-oa:.4f} (Δ={check:.4f})")


if __name__ == '__main__':
    main()
