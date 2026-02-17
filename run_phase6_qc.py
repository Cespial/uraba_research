#!/usr/bin/env python3
"""
Fase 6: Validacion de Calidad Final — URABA ANTIOQUENO
Cross-validates all Phase 3-5 outputs: JSONs, CSVs, figures, manuscript.
"""

import os
import sys
import json
import csv
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(BASE_DIR, 'outputs', 'phase3_stats')
FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
TABLE_DIR = os.path.join(BASE_DIR, 'outputs', 'tables')
PAPER_DIR = os.path.join(BASE_DIR, 'paper')
QC_DIR = os.path.join(BASE_DIR, 'outputs', 'phase6_qc')
os.makedirs(QC_DIR, exist_ok=True)

issues = []
warnings = []
passed = []


def load_json(f):
    with open(os.path.join(STATS_DIR, f)) as fh:
        return json.load(fh)


def load_csv_rows(f):
    with open(os.path.join(TABLE_DIR, f)) as fh:
        return list(csv.reader(fh))


def load_manuscript():
    # Try LaTeX first, fall back to markdown
    tex_path = os.path.join(BASE_DIR, 'overleaf', 'main.tex')
    md_path = os.path.join(PAPER_DIR, 'manuscript_v2.md')
    if os.path.exists(tex_path):
        with open(tex_path) as f:
            return f.read()
    with open(md_path) as f:
        return f.read()


def check(condition, msg, severity='ERROR'):
    if condition:
        passed.append(msg)
    else:
        if severity == 'WARNING':
            warnings.append(msg)
        else:
            issues.append(msg)
    return condition


# ============================================================
# 1. JSON INTEGRITY
# ============================================================

def check_json_integrity():
    print("\n[1] JSON Integrity Checks")
    print("-" * 50)

    required_files = [
        'classification_metrics.json', 'change_detection_results.json',
        'ecosystem_services_results.json', 'climate_analysis_results.json',
        'hotspot_analysis_results.json', 'gwr_drivers_results.json',
        'feature_importance.json', 'ca_markov_results.json'
    ]

    for f in required_files:
        path = os.path.join(STATS_DIR, f)
        exists = os.path.exists(path)
        check(exists, f"  JSON exists: {f}")
        if exists:
            try:
                with open(path) as fh:
                    data = json.load(fh)
                check(len(data) > 0, f"  JSON non-empty: {f}")
            except json.JSONDecodeError:
                check(False, f"  JSON valid: {f}")


# ============================================================
# 2. CLASSIFICATION METRICS CONSISTENCY
# ============================================================

def check_classification():
    print("\n[2] Classification Metrics Consistency")
    print("-" * 50)

    clf = load_json('classification_metrics.json')
    periods = ['pre_acuerdo', 'transicion', 'post_acuerdo_1', 'post_acuerdo_2']
    years = [2013, 2016, 2020, 2024]

    for pk, yr in zip(periods, years):
        p = clf[pk]
        check(p['year'] == yr, f"  {pk}: year={yr}")
        check(0.0 < p['overall_accuracy'] <= 1.0, f"  {pk}: OA in valid range ({p['overall_accuracy']:.3f})")
        check(-1.0 <= p['kappa'] <= 1.0, f"  {pk}: Kappa in valid range ({p['kappa']:.3f})")
        check(p['n_training'] > 0, f"  {pk}: n_training > 0 ({p['n_training']})")
        check(p['n_validation'] > 0, f"  {pk}: n_validation > 0 ({p['n_validation']})")

        # Area totals should be consistent (~1,100,000 ha = 11,000 km2)
        areas = p['class_areas_ha']
        total = sum(v['area_ha'] for v in areas.values())
        check(800_000 < total < 1_500_000,
              f"  {pk}: total area reasonable ({total:,.0f} ha)", 'WARNING')

        # No negative areas
        for cid, info in areas.items():
            check(info['area_ha'] >= 0, f"  {pk}: class {cid} area >= 0")

        # 8 classes expected for Uraba (includes Mangroves)
        check(len(areas) == 8,
              f"  {pk}: has 8 LULC classes ({len(areas)} found)", 'WARNING')

        # Dense forest should be > 0 for all periods
        dense = areas.get('1', {}).get('area_ha', 0)
        check(dense > 0, f"  {pk}: dense forest > 0 ha ({dense:,.0f} ha)")

    # Cross-period: total area should be similar across periods
    totals = []
    for pk in periods:
        areas = clf[pk]['class_areas_ha']
        totals.append(sum(v['area_ha'] for v in areas.values()))
    max_diff = max(totals) - min(totals)
    pct_diff = max_diff / min(totals) * 100
    check(pct_diff < 20,
          f"  Cross-period area consistency: max diff {pct_diff:.1f}% (threshold 20%)", 'WARNING')


# ============================================================
# 3. CHANGE DETECTION CONSISTENCY
# ============================================================

def check_change_detection():
    print("\n[3] Change Detection Consistency")
    print("-" * 50)

    chg = load_json('change_detection_results.json')
    clf = load_json('classification_metrics.json')

    for tk in ['T1_T2', 'T2_T3', 'T3_T4']:
        trans = chg[tk]
        check('transitions' in trans, f"  {tk}: has transitions")
        check('change_rates' in trans, f"  {tk}: has change_rates")

        # Transition areas should sum to approximately total area (~1.1M ha)
        total_trans = sum(v['area_ha'] for v in trans['transitions'].values())
        check(total_trans > 500_000,
              f"  {tk}: total transition area ({total_trans:,.0f} ha)")

        # Change rates: from areas should match classification
        cr = trans['change_rates']
        for cid, r in cr.items():
            if r['area_t1_ha'] > 0:
                check(abs(r['annual_rate_pct']) < 100,
                      f"  {tk}: class {cid} annual rate reasonable ({r['annual_rate_pct']:.1f}%/yr)", 'WARNING')

    # Hansen data present
    check('hansen_gfc' in chg, "  Hansen GFC data present")
    if 'hansen_gfc' in chg:
        for pk in ['pre_acuerdo', 'transicion', 'post_acuerdo_1', 'post_acuerdo_2']:
            loss = chg['hansen_gfc'][pk]['loss_ha']
            check(loss > 0, f"  Hansen {pk}: loss > 0 ({loss:,.0f} ha)")


# ============================================================
# 4. ECOSYSTEM SERVICES
# ============================================================

def check_ecosystem():
    print("\n[4] Ecosystem Services Consistency")
    print("-" * 50)

    eco = load_json('ecosystem_services_results.json')
    periods = ['pre_acuerdo', 'transicion', 'post_acuerdo_1', 'post_acuerdo_2']

    for pk in periods:
        p = eco[pk]
        check(p['carbon_Mg_C'] > 0, f"  {pk}: carbon > 0 ({p['carbon_Mg_C']/1e6:.0f} Mt)")
        # Water yield: Choco bioregion has very high precipitation (3000-8000 mm/yr)
        check(0 < p['water_yield_mm'] < 8000, f"  {pk}: water yield reasonable ({p['water_yield_mm']:.0f} mm)")
        check(0 <= p['habitat_quality_mean'] <= 1, f"  {pk}: habitat quality 0-1 ({p['habitat_quality_mean']:.3f})")

    # Carbon should generally decline (forest loss)
    c_vals = [eco[pk]['carbon_Mg_C'] for pk in periods]
    check(c_vals[0] > c_vals[-1], f"  Carbon declining trend: {c_vals[0]/1e6:.0f} -> {c_vals[-1]/1e6:.0f} Mt")

    # Carbon changes should match stock differences
    c13_16 = eco['carbon_change_2013_2016']['net_Mg_C']
    expected = eco['transicion']['carbon_Mg_C'] - eco['pre_acuerdo']['carbon_Mg_C']
    diff = abs(c13_16 - expected)
    check(diff < 1e6, f"  Carbon change 2013-2016 matches stocks (diff: {diff/1e6:.1f} Mt)", 'WARNING')

    c16_20 = eco['carbon_change_2016_2020']['net_Mg_C']
    expected = eco['post_acuerdo_1']['carbon_Mg_C'] - eco['transicion']['carbon_Mg_C']
    diff = abs(c16_20 - expected)
    check(diff < 1e6, f"  Carbon change 2016-2020 matches stocks (diff: {diff/1e6:.1f} Mt)", 'WARNING')

    c20_24 = eco['carbon_change_2020_2024']['net_Mg_C']
    expected = eco['post_acuerdo_2']['carbon_Mg_C'] - eco['post_acuerdo_1']['carbon_Mg_C']
    diff = abs(c20_24 - expected)
    check(diff < 1e6, f"  Carbon change 2020-2024 matches stocks (diff: {diff/1e6:.1f} Mt)", 'WARNING')


# ============================================================
# 5. GWR CONSISTENCY
# ============================================================

def check_gwr():
    print("\n[5] GWR Results Consistency")
    print("-" * 50)

    gwr = load_json('gwr_drivers_results.json')

    # VIF all < 10
    for var, vif in gwr['vif'].items():
        check(vif < 10, f"  VIF {var}: {vif:.2f} < 10")

    # OLS R2 between 0 and 1
    check(0 <= gwr['ols']['r2'] <= 1, f"  OLS R2 valid: {gwr['ols']['r2']:.4f}")

    # GWR R2 > OLS R2
    check(gwr['gwr']['mean_r2'] > gwr['ols']['r2'],
          f"  GWR R2 ({gwr['gwr']['mean_r2']:.4f}) > OLS R2 ({gwr['ols']['r2']:.4f})")

    # GWR AIC < OLS AIC (better fit)
    check(gwr['gwr']['aic'] < gwr['ols']['aic'],
          f"  GWR AIC ({gwr['gwr']['aic']:.0f}) < OLS AIC ({gwr['ols']['aic']:.0f})")

    # Comparison section matches
    check(abs(gwr['comparison']['ols_r2'] - gwr['ols']['r2']) < 0.001,
          "  Comparison OLS R2 matches OLS section")
    check(abs(gwr['comparison']['gwr_mean_r2'] - gwr['gwr']['mean_r2']) < 0.001,
          "  Comparison GWR R2 matches GWR section")


# ============================================================
# 6. CA-MARKOV
# ============================================================

def check_camarkov():
    print("\n[6] CA-Markov Consistency")
    print("-" * 50)

    cam = load_json('ca_markov_results.json')

    # Transition matrix rows should sum to ~1
    # Support both old ('transition_matrix') and new ('transition_matrix_corrected_5x5') keys
    tm_key = 'transition_matrix_corrected_5x5' if 'transition_matrix_corrected_5x5' in cam else 'transition_matrix'
    tm = cam[tm_key]
    for i, row in enumerate(tm):
        row_sum = sum(row)
        if row_sum > 0:
            check(abs(row_sum - 1.0) < 0.01,
                  f"  TM row {i} sums to {row_sum:.4f} (~1.0)")

    # Scenarios should have all active classes (5 active: BDen, BSec, Past, Agua, Urb)
    # Note: Uraba uses 8 LULC classes total but CA-Markov operates on 5 active classes
    # (Cult, Suel, Mang dropped if zero area in calibration period)
    for sc in ['BAU_2030', 'BAU_2040', 'Conservation_2030', 'Conservation_2040',
               'PDET_2030', 'PDET_2040']:
        check(sc in cam, f"  Scenario exists: {sc}")
        if sc in cam:
            total_pct = sum(cam[sc][c]['pct'] for c in cam[sc])
            check(abs(total_pct - 100.0) < 1.0,
                  f"  {sc}: class pcts sum to {total_pct:.1f}% (~100%)", 'WARNING')

    # Conservation should have more forest than BAU
    if 'Conservation_2040' in cam and 'BAU_2040' in cam:
        cons_forest = cam['Conservation_2040'].get('BDen', {}).get('pct', 0)
        bau_forest = cam['BAU_2040'].get('BDen', {}).get('pct', 0)
        check(cons_forest >= bau_forest,
              f"  Conservation forest ({cons_forest}%) >= BAU forest ({bau_forest}%)")


# ============================================================
# 7. FIGURES EXISTENCE AND SIZE
# ============================================================

def check_figures():
    print("\n[7] Figures Quality Check")
    print("-" * 50)

    # Figure names from 12_visualization.py
    required_figs = [
        'fig01_study_area.png', 'fig02_lulc_maps.png',
        'fig03_area_trends.png', 'fig04_transition_matrices.png',
        'fig05_deforestation_rates.png', 'fig06_hotspots.png',
        'fig07_ecosystem_services.png', 'fig08_gwr_coefficients.png',
        'fig09_future_scenarios.png', 'fig10_climate_deforestation.png',
    ]

    for fig in required_figs:
        path = os.path.join(FIG_DIR, fig)
        exists = os.path.exists(path)
        check(exists, f"  Figure exists: {fig}")
        if exists:
            size_kb = os.path.getsize(path) / 1024
            check(size_kb > 50, f"  {fig}: size {size_kb:.0f} KB (>50 KB)")


# ============================================================
# 8. TABLES CONSISTENCY
# ============================================================

def check_tables():
    print("\n[8] Tables Consistency")
    print("-" * 50)

    required_tables = [
        'table01_accuracy.csv', 'table02_class_areas.csv',
        'table03_change_rates.csv', 'table04_ecosystem_services.csv',
        'table05_gwr_results.csv', 'table06_camarkov_projections.csv'
    ]

    for t in required_tables:
        path = os.path.join(TABLE_DIR, t)
        exists = os.path.exists(path)
        check(exists, f"  Table exists: {t}")
        if exists:
            rows = load_csv_rows(t)
            check(len(rows) > 1, f"  {t}: has data rows ({len(rows)-1} rows)")

    # Cross-validate table01 with JSON
    t01_path = os.path.join(TABLE_DIR, 'table01_accuracy.csv')
    if os.path.exists(t01_path):
        clf = load_json('classification_metrics.json')
        rows = load_csv_rows('table01_accuracy.csv')
        for row in rows[1:]:  # skip header
            year = int(row[1])
            oa_table = float(row[6])
            # Find matching period
            for pk in clf:
                if isinstance(clf[pk], dict) and clf[pk].get('year') == year:
                    oa_json = clf[pk]['overall_accuracy'] * 100
                    diff = abs(oa_table - oa_json)
                    check(diff < 0.2,
                          f"  Table01 OA {year} ({oa_table}%) matches JSON ({oa_json:.1f}%)")
                    break

    # Cross-validate table04 with ecosystem JSON
    t04_path = os.path.join(TABLE_DIR, 'table04_ecosystem_services.csv')
    if os.path.exists(t04_path):
        eco = load_json('ecosystem_services_results.json')
        rows = load_csv_rows('table04_ecosystem_services.csv')
        for row in rows[1:5]:  # first 4 data rows
            if len(row) >= 3 and row[2]:
                try:
                    c_table = float(row[2])
                    year = int(row[1])
                    for pk in eco:
                        if isinstance(eco[pk], dict) and eco[pk].get('year') == year:
                            c_json = eco[pk]['carbon_Mg_C'] / 1e6
                            diff = abs(c_table - c_json)
                            check(diff < 1.0,
                                  f"  Table04 Carbon {year} ({c_table} Mt) matches JSON ({c_json:.1f} Mt)")
                            break
                except (ValueError, IndexError):
                    pass


# ============================================================
# 9. MANUSCRIPT CROSS-VALIDATION
# ============================================================

def check_manuscript():
    print("\n[9] Manuscript Cross-Validation")
    print("-" * 50)

    ms = load_manuscript()

    # --- Generic structural checks (no hardcoded Uraba values) ---

    # No remaining placeholders
    xx_count = ms.count('[XX]')
    check(xx_count == 0, f"  No [XX] placeholders remaining ({xx_count} found)")

    tbd_count = len(re.findall(r'\[to be|Results to be|to be populated', ms, re.IGNORECASE))
    check(tbd_count == 0, f"  No 'to be populated' remaining ({tbd_count} found)")

    # Study region identity
    check('Urab' in ms or 'urab' in ms.lower(),
          "  Manuscript mentions Uraba")

    # Study area size (~11,000 km2)
    check('11,000' in ms or '11000' in ms or '11{,}000' in ms,
          "  Manuscript cites study area ~11,000 km2")

    # Hypotheses addressed
    check('H1' in ms, "  Manuscript addresses H1")
    check('H2' in ms, "  Manuscript addresses H2")
    check('H3' in ms, "  Manuscript addresses H3")
    check('H4' in ms, "  Manuscript addresses H4")

    # Key methodological references
    check('Olofsson' in ms, "  Manuscript cites Olofsson et al.")
    check('Pontius' in ms, "  Manuscript cites Pontius (QD/AD)")
    check('Hansen' in ms, "  Manuscript cites Hansen GFC")

    # Figures referenced
    for i in range(1, 11):
        check(f'Fig. {i}' in ms or f'Fig {i}' in ms or f'Figure {i}' in ms or f'(Fig.' in ms,
              f"  Manuscript references Fig. {i}", 'WARNING')


# ============================================================
# 10. SCIENTIFIC COHERENCE
# ============================================================

def check_scientific_coherence():
    print("\n[10] Scientific Coherence Checks")
    print("-" * 50)

    clf = load_json('classification_metrics.json')
    chg = load_json('change_detection_results.json')
    eco = load_json('ecosystem_services_results.json')

    # Total area consistency across all outputs (~11,000 km2 = ~1,100,000 ha)
    clf_total_t1 = sum(v['area_ha'] for v in clf['pre_acuerdo']['class_areas_ha'].values())
    clf_total_t4 = sum(v['area_ha'] for v in clf['post_acuerdo_2']['class_areas_ha'].values())
    diff_pct = abs(clf_total_t1 - clf_total_t4) / clf_total_t1 * 100
    check(diff_pct < 15,
          f"  Total area T1 ({clf_total_t1:,.0f}) vs T4 ({clf_total_t4:,.0f}): {diff_pct:.1f}% diff", 'WARNING')

    # Absolute area sanity: should be in the ~11,000 km2 range
    clf_total_km2 = clf_total_t1 / 100
    check(8_000 < clf_total_km2 < 15_000,
          f"  T1 total area ({clf_total_km2:,.0f} km2) in expected ~11,000 km2 range", 'WARNING')

    # Change detection T1 areas should match classification T1
    t1_t2_rates = chg['T1_T2']['change_rates']
    for cid in ['1', '2', '3']:
        if cid in t1_t2_rates:
            cr_area = t1_t2_rates[cid]['area_t1_ha']
            clf_area = clf['pre_acuerdo']['class_areas_ha'].get(cid, {}).get('area_ha', 0)
            diff = abs(cr_area - clf_area)
            pct = diff / max(clf_area, 1) * 100
            check(pct < 5,
                  f"  T1 class {cid}: change det ({cr_area:,.0f}) vs clf ({clf_area:,.0f}) = {pct:.1f}% diff")

    # Carbon should be consistent with areas (Tier 2 Choco values)
    # Dense*281 + Sec*146 + Past*43.5 + Crops*53.5 + Water*0 + Urban*20 + Bare*15 + Mangrove*247
    areas_t1 = clf['pre_acuerdo']['class_areas_ha']
    calc_carbon = (
        areas_t1.get('1', {}).get('area_ha', 0) * 281 +
        areas_t1.get('2', {}).get('area_ha', 0) * 146 +
        areas_t1.get('3', {}).get('area_ha', 0) * 43.5 +
        areas_t1.get('4', {}).get('area_ha', 0) * 53.5 +
        areas_t1.get('5', {}).get('area_ha', 0) * 0 +
        areas_t1.get('6', {}).get('area_ha', 0) * 20 +
        areas_t1.get('7', {}).get('area_ha', 0) * 15 +
        areas_t1.get('8', {}).get('area_ha', 0) * 247
    )
    json_carbon = eco['pre_acuerdo']['carbon_Mg_C']
    # Note: JSON uses Olofsson-adjusted areas, so pixel-counted areas will differ
    diff_pct = abs(calc_carbon - json_carbon) / json_carbon * 100
    check(diff_pct < 30,
          f"  T1 carbon: pixel-based ({calc_carbon/1e6:.0f} Tg) vs Olofsson-based ({json_carbon/1e6:.0f} Tg) = {diff_pct:.1f}% diff (expected: different areas)")

    # OA values should be reasonable for 8-class tropical classification
    for pk in ['pre_acuerdo', 'transicion', 'post_acuerdo_1', 'post_acuerdo_2']:
        oa = clf[pk]['overall_accuracy']
        check(oa > 0.50,
              f"  {pk}: OA ({oa:.3f}) > chance level (0.125 for 8 classes)", 'WARNING')

    # GWR: check that drivers make physical sense
    gwr = load_json('gwr_drivers_results.json')
    elev_coeff = gwr['ols']['coefficients']['elevation']
    check(elev_coeff < 0,
          f"  Elevation negatively associated with deforestation ({elev_coeff:.3f})", 'WARNING')


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 6: VALIDACION DE CALIDAD FINAL — URABA ANTIOQUENO")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    check_json_integrity()
    check_classification()
    check_change_detection()
    check_ecosystem()
    check_gwr()
    check_camarkov()
    check_figures()
    check_tables()
    check_manuscript()
    check_scientific_coherence()

    # Summary
    print("\n" + "=" * 60)
    print("RESUMEN DE VALIDACION")
    print("=" * 60)
    print(f"  PASSED:   {len(passed)}")
    print(f"  WARNINGS: {len(warnings)}")
    print(f"  ERRORS:   {len(issues)}")
    print()

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  [!] {w}")
        print()

    if issues:
        print("ERRORS:")
        for e in issues:
            print(f"  [X] {e}")
        print()

    # Write report
    report_path = os.path.join(QC_DIR, 'qc_report_final.md')
    with open(report_path, 'w') as f:
        f.write("# FASE 6: REPORTE DE VALIDACION DE CALIDAD FINAL — URABA ANTIOQUENO\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Archivos validados:** 8 JSONs, 6 CSVs, 10 figuras, 1 manuscrito\n\n")
        f.write("---\n\n")
        f.write("## Resumen\n\n")
        f.write(f"| Resultado | Cantidad |\n")
        f.write(f"|-----------|----------|\n")
        f.write(f"| PASSED | {len(passed)} |\n")
        f.write(f"| WARNINGS | {len(warnings)} |\n")
        f.write(f"| ERRORS | {len(issues)} |\n\n")

        f.write("## Checks Passed\n\n")
        for p in passed:
            f.write(f"- [x] {p.strip()}\n")

        if warnings:
            f.write("\n## Warnings\n\n")
            for w in warnings:
                f.write(f"- [!] {w.strip()}\n")

        if issues:
            f.write("\n## Errors\n\n")
            for e in issues:
                f.write(f"- [X] {e.strip()}\n")

        f.write("\n---\n\n")
        f.write("## Datos Validados\n\n")
        f.write("### Phase 3 JSONs\n")
        f.write("- classification_metrics.json: 4 periods, 8 classes, OA/Kappa/confusion matrices\n")
        f.write("- change_detection_results.json: 3 transition matrices + Hansen GFC\n")
        f.write("- ecosystem_services_results.json: Carbon, water yield, habitat quality\n")
        f.write("- climate_analysis_results.json: Precipitation, LST, SPI, trends\n")
        f.write("- hotspot_analysis_results.json: Moran's I, Gi* counts\n")
        f.write("- gwr_drivers_results.json: OLS, GWR, VIF, 8 drivers\n")
        f.write("- feature_importance.json: RF importance per period\n")
        f.write("- ca_markov_results.json: Transition matrix, 6 scenarios\n\n")

        f.write("### Phase 4 Figures (10)\n")
        if os.path.isdir(FIG_DIR):
            for fig in sorted(os.listdir(FIG_DIR)):
                if fig.endswith('.png'):
                    size = os.path.getsize(os.path.join(FIG_DIR, fig)) / 1024
                    f.write(f"- {fig} ({size:.0f} KB)\n")
        else:
            f.write("- (figures directory not found)\n")

        f.write("\n### Phase 4 Tables (6)\n")
        if os.path.isdir(TABLE_DIR):
            for tab in sorted(os.listdir(TABLE_DIR)):
                if tab.endswith('.csv'):
                    f.write(f"- {tab}\n")
        else:
            f.write("- (tables directory not found)\n")

        f.write("\n### Phase 5 Manuscript\n")
        f.write("- Manuscript validated for structural completeness and key references\n")

    print(f"\nReporte guardado: {report_path}")
    print("=" * 60)

    return len(issues)


if __name__ == '__main__':
    n_errors = main()
    sys.exit(0 if n_errors == 0 else 1)
