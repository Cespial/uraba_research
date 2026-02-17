#!/usr/bin/env python3
"""
08b_carbon_uncertainty.py
=========================
Phase 2 enhancement: Carbon stock estimation with propagated uncertainty.

Combines:
  - Olofsson et al. (2014) stratified area estimates (with area SE)
  - Tier 2 Colombia carbon pools (Alvarez et al. 2012) with pool-level SE

Uncertainty propagation follows the error-propagation formula for products
of independent random variables:
  Var(C_total) = sum_i [ c_i^2 * Var(A_i) + A_i^2 * Var(c_i)
                        + Var(A_i) * Var(c_i) ]

where:
  c_i   = total carbon density for class i (Mg C/ha)
  A_i   = Olofsson-adjusted area for class i (ha)
  Var(A_i) = (area_se_ha)^2
  Var(c_i) = sum of squared pool standard errors (c_above_se^2 + c_below_se^2
             + c_soil_se^2 + c_dead_se^2)

Net carbon change between periods propagates uncertainty as:
  Var(delta_C) = Var(C_t1) + Var(C_t2)

Outputs:
  outputs/phase3_stats/ecosystem_services_results.json  (overwritten)
"""

import json
import math
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OLOFSSON_PATH = os.path.join(
    PROJECT_ROOT, 'outputs', 'phase3_stats', 'olofsson_area_estimates.json'
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, 'outputs', 'phase3_stats', 'ecosystem_services_results.json'
)

# ---------------------------------------------------------------------------
# Import Tier 2 carbon pools from gee_config
# We avoid initialising Earth Engine by importing only the dict we need.
# ---------------------------------------------------------------------------
# gee_config.py runs ee.Initialize() at import time, which would fail
# without credentials.  Instead we parse the CARBON_POOLS dict directly
# from the config file to keep this script self-contained and runnable
# without GEE credentials.
# ---------------------------------------------------------------------------

def _load_carbon_pools():
    """
    Parse CARBON_POOLS from gee_config.py without triggering ee.Initialize().
    Returns a dict identical in structure to gee_config.CARBON_POOLS.
    """
    config_path = os.path.join(PROJECT_ROOT, 'gee_config.py')
    with open(config_path, 'r') as f:
        source = f.read()

    # Extract the CARBON_POOLS block
    start_marker = 'CARBON_POOLS = {'
    start_idx = source.index(start_marker)
    # Find the matching closing brace
    brace_depth = 0
    end_idx = start_idx
    for i in range(start_idx, len(source)):
        if source[i] == '{':
            brace_depth += 1
        elif source[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                end_idx = i + 1
                break

    block = source[start_idx:end_idx]
    # Remove inline comments (everything after # on each line)
    import re
    block_clean = re.sub(r'#[^\n]*', '', block)

    local_ns = {}
    exec(block_clean, {}, local_ns)  # noqa: S102
    return local_ns['CARBON_POOLS']


CARBON_POOLS = _load_carbon_pools()

# ---------------------------------------------------------------------------
# LULC class names (lightweight; avoids importing gee_config)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    1: 'Bosque denso',
    2: 'Bosque secundario',
    3: 'Pasturas',
    4: 'Cultivos',
    5: 'Agua',
    6: 'Urbano',
    7: 'Suelo desnudo',
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def total_carbon_density(pools: dict) -> float:
    """Sum of all pool mean densities (Mg C/ha)."""
    return pools['c_above'] + pools['c_below'] + pools['c_soil'] + pools['c_dead']


def total_carbon_density_variance(pools: dict) -> float:
    """Variance of total carbon density = sum of squared pool SEs."""
    return (
        pools['c_above_se'] ** 2
        + pools['c_below_se'] ** 2
        + pools['c_soil_se'] ** 2
        + pools['c_dead_se'] ** 2
    )


def compute_period_carbon(period_data: dict) -> dict:
    """
    For a single period, compute total carbon stock and its uncertainty.

    Returns a dict with:
      - carbon_Mg_C: total carbon (Mg C)
      - carbon_se: standard error (Mg C)
      - carbon_ci95: half-width of 95 % CI (Mg C)
      - per_class: breakdown by class
    """
    per_class = period_data.get('per_class', {})

    total_carbon = 0.0
    total_variance = 0.0
    class_details = {}

    # Iterate over ALL 7 LULC classes (some may be absent from Olofsson)
    for cls_id in range(1, 8):
        cls_str = str(cls_id)
        pools = CARBON_POOLS[cls_id]
        c_i = total_carbon_density(pools)          # Mg C/ha
        var_c_i = total_carbon_density_variance(pools)  # (Mg C/ha)^2

        if cls_str in per_class:
            A_i = per_class[cls_str]['adjusted_area_ha']  # ha
            var_A_i = per_class[cls_str]['area_se_ha'] ** 2  # ha^2
        else:
            # Class not reported in Olofsson (e.g. Cultivos, Suelo desnudo
            # had zero mapped area and were merged).  Treat area = 0.
            A_i = 0.0
            var_A_i = 0.0

        # Carbon stock for this class
        C_i = c_i * A_i  # Mg C

        # Variance of C_i using error propagation for product of
        # two independent random variables:
        # Var(X*Y) = E[X]^2 Var(Y) + E[Y]^2 Var(X) + Var(X) Var(Y)
        var_C_i = (c_i ** 2) * var_A_i + (A_i ** 2) * var_c_i + var_A_i * var_c_i

        total_carbon += C_i
        total_variance += var_C_i

        class_details[cls_str] = {
            'name': CLASS_NAMES[cls_id],
            'c_density_MgC_ha': round(c_i, 2),
            'c_density_se_MgC_ha': round(math.sqrt(var_c_i), 2),
            'area_ha': round(A_i, 1),
            'area_se_ha': round(math.sqrt(var_A_i), 1),
            'carbon_Mg_C': round(C_i, 0),
            'carbon_se_Mg_C': round(math.sqrt(var_C_i), 0),
        }

    total_se = math.sqrt(total_variance)
    ci95 = 1.96 * total_se

    return {
        'carbon_Mg_C': round(total_carbon, 0),
        'carbon_se': round(total_se, 0),
        'carbon_ci95': round(ci95, 0),
        'carbon_lower_ci95': round(total_carbon - ci95, 0),
        'carbon_upper_ci95': round(total_carbon + ci95, 0),
        'per_class': class_details,
    }


def compute_carbon_change(result_t1: dict, result_t2: dict, label: str) -> dict:
    """
    Net carbon change (t2 - t1) with propagated uncertainty.
    Var(delta) = Var(C_t1) + Var(C_t2)  (independent periods).
    """
    net = result_t2['carbon_Mg_C'] - result_t1['carbon_Mg_C']
    var_net = result_t1['carbon_se'] ** 2 + result_t2['carbon_se'] ** 2
    se_net = math.sqrt(var_net)
    ci95 = 1.96 * se_net

    return {
        'label': label,
        'net_Mg_C': round(net, 0),
        'net_se': round(se_net, 0),
        'net_ci95': round(ci95, 0),
        'net_lower_ci95': round(net - ci95, 0),
        'net_upper_ci95': round(net + ci95, 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 65)
    print('08b  CARBON STOCK ESTIMATION WITH PROPAGATED UNCERTAINTY')
    print(f'     {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 65)

    # ------------------------------------------------------------------
    # 1. Load Olofsson area estimates
    # ------------------------------------------------------------------
    with open(OLOFSSON_PATH, 'r') as f:
        olofsson = json.load(f)

    periods_data = olofsson['periods']
    period_keys = list(periods_data.keys())

    print(f'\nLoaded Olofsson estimates for {len(period_keys)} periods')
    print(f'Carbon pool source: Tier 2 Colombia (Alvarez et al. 2012)')

    # Print carbon density table
    print('\nCarbon density table (Mg C/ha):')
    print(f'  {"Class":<22s} {"Above":>6s} {"Below":>6s} {"Soil":>6s} {"Dead":>6s} {"Total":>6s} {"SE_tot":>7s}')
    for cls_id in range(1, 8):
        p = CARBON_POOLS[cls_id]
        c_tot = total_carbon_density(p)
        se_tot = math.sqrt(total_carbon_density_variance(p))
        print(f'  {CLASS_NAMES[cls_id]:<22s} {p["c_above"]:6.1f} {p["c_below"]:6.1f} '
              f'{p["c_soil"]:6.1f} {p["c_dead"]:6.1f} {c_tot:6.1f} {se_tot:7.2f}')

    # ------------------------------------------------------------------
    # 2. Compute per-period carbon with uncertainty
    # ------------------------------------------------------------------
    period_results = {}

    for pk in period_keys:
        pd = periods_data[pk]
        result = compute_period_carbon(pd)
        year = pd['year']
        label = pd['label']

        period_results[pk] = result
        period_results[pk]['year'] = year

        # Preserve non-carbon ecosystem service fields from any existing
        # results (water_yield_mm, baseflow_mm, habitat_quality) if we
        # later want to merge.  For now we store only carbon fields.

        total_Tg = result['carbon_Mg_C'] / 1e6
        se_Tg = result['carbon_se'] / 1e6
        ci95_Tg = result['carbon_ci95'] / 1e6

        print(f'\n  {label} ({year})')
        print(f'    Total C: {total_Tg:,.2f} Tg C  (SE: {se_Tg:,.2f}, 95%CI: +/-{ci95_Tg:,.2f})')

        # Per-class summary
        for cls_str in sorted(result['per_class'].keys(), key=int):
            cd = result['per_class'][cls_str]
            if cd['carbon_Mg_C'] == 0:
                continue
            print(f'      {cd["name"]:<22s}  {cd["carbon_Mg_C"]/1e6:8.2f} Tg C  '
                  f'(SE: {cd["carbon_se_Mg_C"]/1e6:6.2f})')

    # ------------------------------------------------------------------
    # 3. Carbon change between consecutive periods
    # ------------------------------------------------------------------
    print('\n' + '-' * 65)
    print('  NET CARBON CHANGE (with uncertainty)')
    print('-' * 65)

    change_pairs = [
        ('pre_acuerdo', 'transicion', '2013-2016'),
        ('transicion', 'post_acuerdo_1', '2016-2020'),
        ('post_acuerdo_1', 'post_acuerdo_2', '2020-2024'),
    ]

    change_results = {}

    for pk1, pk2, label_years in change_pairs:
        change_key = f'carbon_change_{label_years.replace("-", "_")}'
        change = compute_carbon_change(period_results[pk1], period_results[pk2], label_years)
        change_results[change_key] = change

        net_Tg = change['net_Mg_C'] / 1e6
        se_Tg = change['net_se'] / 1e6
        ci95_Tg = change['net_ci95'] / 1e6
        sign = '+' if net_Tg >= 0 else ''

        print(f'  {label_years}: {sign}{net_Tg:,.2f} Tg C  '
              f'(SE: {se_Tg:,.2f}, 95%CI: [{(net_Tg - ci95_Tg):,.2f}, {(net_Tg + ci95_Tg):,.2f}])')

    # ------------------------------------------------------------------
    # 4. Build output JSON (same top-level structure as existing file,
    #    but enriched with uncertainty fields)
    # ------------------------------------------------------------------
    # Read existing results to preserve non-carbon fields
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            existing = json.load(f)

    output = {}

    for pk in period_keys:
        pr = period_results[pk]
        # Merge with existing ecosystem service fields if present
        entry = {}
        if pk in existing:
            entry.update(existing[pk])

        entry['year'] = pr['year']
        entry['carbon_Mg_C'] = pr['carbon_Mg_C']
        entry['carbon_se'] = pr['carbon_se']
        entry['carbon_ci95'] = pr['carbon_ci95']
        entry['carbon_lower_ci95'] = pr['carbon_lower_ci95']
        entry['carbon_upper_ci95'] = pr['carbon_upper_ci95']
        entry['carbon_per_class'] = pr['per_class']

        output[pk] = entry

    for ck, cv in change_results.items():
        output[ck] = cv

    # Metadata
    output['_metadata'] = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'carbon_pool_source': 'Tier 2 Colombia (Alvarez et al. 2012)',
        'area_source': 'Olofsson et al. (2014) stratified estimators',
        'uncertainty_method': 'Error propagation for products of independent RVs: '
                             'Var(C_i) = c_i^2*Var(A_i) + A_i^2*Var(c_i) + Var(A_i)*Var(c_i)',
        'ci_level': '95% (z=1.96)',
        'change_uncertainty': 'Var(delta) = Var(C_t1) + Var(C_t2)',
    }

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\nResults written to: {OUTPUT_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()
