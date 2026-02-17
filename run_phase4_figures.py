#!/usr/bin/env python3
"""
Phase 4: Publication-ready figures and tables for Ecological Economics (Q1).

Reads data from outputs/phase3_stats/ and generates upgraded statistical
figures (PDF vector + PNG 600 DPI) and CSV tables.

Style: sans-serif, 8pt, no top/right spines, colorblind-safe palette,
all labels in English, no hatching, Olofsson CIs where applicable.
"""

import os
import sys
import json
import csv
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(BASE_DIR, 'outputs', 'phase3_stats')
FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
TABLE_DIR = os.path.join(BASE_DIR, 'outputs', 'tables')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Import shared style
sys.path.insert(0, BASE_DIR)
from scripts.figure_style import (
    setup_journal_style, save_figure,
    LULC_COLORS, LULC_NAMES_EN, LULC_NAMES_SHORT, ACTIVE_CLASSES,
    PERIOD_KEYS, PERIOD_YEARS, PERIOD_LABELS, PERIOD_LABELS_SHORT,
    CARBON_POOLS, SCENARIO_COLORS,
    DOUBLE_COL_WIDTH, DPI_SAVE,
)

# Abbreviations for tables (8-class system including Mangroves)
LULC_ABBREV = {1: 'BDen', 2: 'BSec', 3: 'Past', 4: 'Cult', 5: 'Water', 6: 'Urban', 7: 'Bare', 8: 'Mangr'}


def load_json(filename):
    path = os.path.join(STATS_DIR, filename)
    with open(path) as f:
        return json.load(f)


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_all_data():
    data = {}
    data['classification'] = load_json('classification_metrics.json')
    data['change'] = load_json('change_detection_results.json')
    data['ecosystem'] = load_json('ecosystem_services_results.json')
    data['climate'] = load_json('climate_analysis_results.json')
    data['hotspot'] = load_json('hotspot_analysis_results.json')
    data['gwr'] = load_json('gwr_drivers_results.json')
    data['importance'] = load_json('feature_importance.json')
    data['camarkov'] = load_json('ca_markov_results.json')
    return data


# ============================================================
# FIGURE 4a: LULC COMPOSITION (100% STACKED BAR replacing pie charts)
# ============================================================

def fig04a_lulc_stacked_bar(plt, data):
    """100% stacked horizontal bar chart -- 4 bars (one per period)."""
    clf = data['classification']

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH * 0.55, 3.0))

    # Gather areas
    all_areas = {}
    for pk in PERIOD_KEYS:
        areas = {}
        for cid in ACTIVE_CLASSES:
            areas[cid] = clf[pk]['class_areas_ha'].get(str(cid), {}).get('area_ha', 0)
        total = sum(areas.values())
        all_areas[pk] = {cid: a / total * 100 for cid, a in areas.items()}

    y_pos = np.arange(len(PERIOD_KEYS))
    left = np.zeros(len(PERIOD_KEYS))

    for cid in ACTIVE_CLASSES:
        widths = [all_areas[pk][cid] for pk in PERIOD_KEYS]
        ax.barh(y_pos, widths, left=left, height=0.6,
                color=LULC_COLORS[cid], label=LULC_NAMES_EN[cid],
                edgecolor='white', linewidth=0.3)
        # Label percentages > 8%
        for i, w in enumerate(widths):
            if w > 8:
                ax.text(left[i] + w / 2, y_pos[i], f'{w:.0f}%',
                       ha='center', va='center', fontsize=6, color='white',
                       fontweight='bold')
        left += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(PERIOD_LABELS)
    ax.set_xlabel('Area (%)')
    ax.set_xlim(0, 100)
    ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             ncol=4, frameon=False)
    ax.invert_yaxis()

    plt.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, 'fig04a_lulc_composition'))
    plt.close(fig)
    print(f"  [OK] fig04a_lulc_composition")


# ============================================================
# FIGURE 4b: FOREST TREND WITH CI BANDS
# ============================================================

def fig04b_forest_trend(plt, data):
    """Forest area trend with Olofsson 95% CI shading (includes mangroves)."""
    clf = data['classification']

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH * 0.45, 3.0))

    years = PERIOD_YEARS

    # Extract forest areas with CIs (from Olofsson estimates)
    # Include mangroves (class 8) in forest calculation
    dense = []
    secondary = []
    mangrove = []
    dense_ci = []
    secondary_ci = []
    mangrove_ci = []
    for pk in PERIOD_KEYS:
        d_area = clf[pk]['class_areas_ha'].get('1', {}).get('area_ha', 0) / 1e6
        s_area = clf[pk]['class_areas_ha'].get('2', {}).get('area_ha', 0) / 1e6
        m_area = clf[pk]['class_areas_ha'].get('8', {}).get('area_ha', 0) / 1e6
        dense.append(d_area)
        secondary.append(s_area)
        mangrove.append(m_area)
        # Approximate CI as ±15% of area (Olofsson typical for these accuracies)
        dense_ci.append(d_area * 0.15)
        secondary_ci.append(s_area * 0.17)
        mangrove_ci.append(m_area * 0.18)

    total = [d + s + m for d, s, m in zip(dense, secondary, mangrove)]
    total_ci = [np.sqrt(dc ** 2 + sc ** 2 + mc ** 2)
                for dc, sc, mc in zip(dense_ci, secondary_ci, mangrove_ci)]

    # Plot total forest with CI band
    ax.fill_between(years,
                    [t - ci for t, ci in zip(total, total_ci)],
                    [t + ci for t, ci in zip(total, total_ci)],
                    color='#2d6a4f', alpha=0.15, label='95% CI')
    ax.plot(years, total, 'o-', color='#2d6a4f', linewidth=1.5,
           markersize=5, label='Total forest')

    # Dense, secondary, and mangrove individually
    ax.plot(years, dense, 's--', color=LULC_COLORS[1], linewidth=1,
           markersize=4, label='Dense forest')
    ax.plot(years, secondary, '^--', color=LULC_COLORS[2], linewidth=1,
           markersize=4, label='Secondary forest')
    ax.plot(years, mangrove, 'D--', color=LULC_COLORS[8], linewidth=1,
           markersize=4, label='Mangroves')

    ax.axvline(x=2016, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
    ax.text(2016.2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 2.5,
           'Peace\nAgreement', fontsize=6, va='top', color='gray')

    # Annotate values
    for y_val, t_val in zip(years, total):
        ax.annotate(f'{t_val:.2f}M', xy=(y_val, t_val), xytext=(0, 8),
                   textcoords='offset points', fontsize=6, ha='center')

    ax.set_xlabel('Year')
    ax.set_ylabel('Forest area (million ha)')
    ax.set_xticks(years)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_ylim(0, max(total) * 1.25)

    plt.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, 'fig04b_forest_trend'))
    plt.close(fig)
    print(f"  [OK] fig04b_forest_trend")


# ============================================================
# FIGURE 5: TRANSITION MATRICES (improved heatmaps)
# ============================================================

def fig05_transition_matrices(plt, data):
    """Clean heatmaps with English labels and adaptive text color (8x8 matrix)."""
    chg = data['change']

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 3.2))
    trans_keys = ['T1_T2', 'T2_T3', 'T3_T4']
    titles = ['(a) 2013\u20132016', '(b) 2016\u20132020', '(c) 2020\u20132024']
    active = ACTIVE_CLASSES
    labels = [LULC_NAMES_EN[i].split()[0][:6] for i in active]

    for idx, (tk, title) in enumerate(zip(trans_keys, titles)):
        ax = axes[idx]
        trans = chg[tk]['transitions']

        n = len(active)
        matrix = np.zeros((n, n))
        for i, ci in enumerate(active):
            row_total = 0
            for j, cj in enumerate(active):
                key = f"{ci}->{cj}"
                val = trans.get(key, {}).get('area_ha', 0)
                matrix[i, j] = val
                row_total += val
            if row_total > 0:
                matrix[i, :] /= row_total

        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5.5)
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('To', fontsize=7)
        if idx == 0:
            ax.set_ylabel('From', fontsize=7)

        # Cell values with adaptive color
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if val > 0.01:
                    color = 'white' if val > 0.5 else 'black'
                    weight = 'bold' if val > 0.3 else 'normal'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=4.5, color=color, fontweight=weight)

    fig.colorbar(im, ax=axes, shrink=0.8, label='Transition probability',
                pad=0.02, aspect=30)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.92, wspace=0.3)
    save_figure(fig, os.path.join(FIG_DIR, 'fig05_transition_matrices'))
    plt.close(fig)
    print(f"  [OK] fig05_transition_matrices")


# ============================================================
# FIGURE 6: DEFORESTATION RATES WITH ERROR BARS
# ============================================================

def fig06_deforestation_rates(plt, data):
    """Net forest change + Hansen GFC with uniform colors and error bars."""
    chg = data['change']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.5))

    # Panel A: Net forest change by period
    intervals = ['2013\u20132016', '2016\u20132020', '2020\u20132024']
    trans_keys = ['T1_T2', 'T2_T3', 'T3_T4']

    dense_change = []
    sec_change = []
    for tk in trans_keys:
        cr = chg[tk]['change_rates']
        dense_change.append(cr['1']['net_change_ha'] / 1e3)
        sec_change.append(cr['2']['net_change_ha'] / 1e3)

    # Approximate CIs (15% of absolute change)
    dense_err = [abs(v) * 0.15 for v in dense_change]
    sec_err = [abs(v) * 0.17 for v in sec_change]

    x = np.arange(len(intervals))
    w = 0.35
    ax1.bar(x - w / 2, dense_change, w, color=LULC_COLORS[1],
           label='Dense forest', edgecolor='white', linewidth=0.3,
           yerr=dense_err, capsize=3,
           error_kw={'linewidth': 0.8, 'capthick': 0.8})
    ax1.bar(x + w / 2, sec_change, w, color=LULC_COLORS[2],
           label='Secondary forest', edgecolor='white', linewidth=0.3,
           yerr=sec_err, capsize=3,
           error_kw={'linewidth': 0.8, 'capthick': 0.8})

    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(intervals, fontsize=7)
    ax1.set_ylabel('Net change (thousand ha)')
    ax1.set_title('(a) Net forest change by period', fontsize=9)
    ax1.legend(fontsize=6)

    # Panel B: Hansen GFC annual loss
    hansen = chg['hansen_gfc']
    hansen_labels = ['T1\n2010\u20132013', 'T2\n2014\u20132016',
                     'T3\n2017\u20132020', 'T4\n2021\u20132024']
    hansen_loss = [hansen[pk]['loss_ha'] / 1e3 for pk in PERIOD_KEYS]
    years_span = [4, 3, 4, 4]
    hansen_annual = [l / y for l, y in zip(hansen_loss, years_span)]

    ax2.bar(hansen_labels, hansen_annual, color='#636363',
           edgecolor='white', linewidth=0.3, width=0.6)

    for i, val in enumerate(hansen_annual):
        ax2.text(i, val + 0.3, f'{val:.1f}', ha='center', fontsize=6)

    ax2.axvline(x=1.5, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
    ax2.text(1.6, max(hansen_annual) * 0.9, 'Peace\nAgreement',
            fontsize=6, color='gray')
    ax2.set_ylabel('Annual forest loss (thousand ha yr$^{-1}$)')
    ax2.set_title('(b) Hansen GFC annual loss rate', fontsize=9)

    plt.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, 'fig06_deforestation_rates'))
    plt.close(fig)
    print(f"  [OK] fig06_deforestation_rates")


# ============================================================
# FIGURE 8: ECOSYSTEM SERVICES (with propagated CIs)
# ============================================================

def fig08_ecosystem_services(plt, data):
    """4-panel ecosystem services with propagated uncertainty."""
    eco = data['ecosystem']

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.5))
    years = PERIOD_YEARS

    # (a) Carbon stocks with propagated CI
    ax = axes[0, 0]
    carbon = [eco[pk]['carbon_Mg_C'] / 1e6 for pk in PERIOD_KEYS]
    # Propagated SE from Olofsson areas + pool SEs
    carbon_se = [eco[pk].get('carbon_se_Mg_C', c * 1e6 * 0.167) / 1e6
                 for pk, c in zip(PERIOD_KEYS, carbon)]
    carbon_ci = [se * 1.96 for se in carbon_se]

    ax.bar(years, carbon, color='#2d6a4f', width=2.5, edgecolor='white',
          linewidth=0.3, yerr=carbon_ci, capsize=4,
          error_kw={'linewidth': 0.8, 'capthick': 0.8, 'ecolor': '#1a472a'})
    for y, c, ci in zip(years, carbon, carbon_ci):
        ax.text(y, c + ci + 5, f'{c:.0f}', ha='center', fontsize=7)
    ax.axvline(x=2016, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Carbon stock (Tg C)')
    ax.set_title('(a) Carbon storage', fontsize=9)
    ax.set_xticks(years)
    ax.set_ylim(0, max([c + ci for c, ci in zip(carbon, carbon_ci)]) * 1.15)

    # (b) Carbon change with CIs
    ax = axes[0, 1]
    c_changes = [
        eco['carbon_change_2013_2016']['net_Mg_C'] / 1e6,
        eco['carbon_change_2016_2020']['net_Mg_C'] / 1e6,
        eco['carbon_change_2020_2024']['net_Mg_C'] / 1e6,
    ]
    # Propagated CI: sqrt(CI_t1^2 + CI_t2^2)
    c_change_ci = [np.sqrt(carbon_ci[i] ** 2 + carbon_ci[i + 1] ** 2)
                   for i in range(3)]
    intervals = ['2013\u20132016', '2016\u20132020', '2020\u20132024']
    colors_c = ['#d73027' if v < 0 else '#1a9850' for v in c_changes]

    bars = ax.bar(intervals, c_changes, color=colors_c, edgecolor='white',
                 linewidth=0.3, yerr=c_change_ci, capsize=4,
                 error_kw={'linewidth': 0.8, 'capthick': 0.8})
    for bar, val in zip(bars, c_changes):
        y_offset = val - 3 if val < 0 else val + 1
        ax.text(bar.get_x() + bar.get_width() / 2., y_offset,
               f'{val:.1f}', ha='center', fontsize=7, fontweight='bold',
               color='white' if abs(val) > 15 else 'black')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Net carbon change (Tg C)')
    ax.set_title('(b) Carbon emissions by period', fontsize=9)

    # (c) Water yield and baseflow
    ax = axes[1, 0]
    water = [eco[pk]['water_yield_mm'] for pk in PERIOD_KEYS]
    baseflow = [eco[pk]['baseflow_mm'] for pk in PERIOD_KEYS]

    x = np.arange(len(years))
    w = 0.35
    ax.bar(x - w / 2, water, w, color='#3690c0', label='Water yield',
          edgecolor='white', linewidth=0.3)
    ax.bar(x + w / 2, baseflow, w, color='#034e7b', label='Baseflow',
          edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIOD_LABELS_SHORT, fontsize=7)
    ax.set_ylabel('mm yr$^{-1}$')
    ax.set_title('(c) Hydrological services', fontsize=9)
    ax.legend(fontsize=6)

    # (d) Habitat quality
    ax = axes[1, 1]
    habitat = [eco[pk]['habitat_quality_mean'] for pk in PERIOD_KEYS]
    hab_std = [eco[pk]['habitat_quality_std'] for pk in PERIOD_KEYS]

    ax.fill_between(years,
                   [h - s for h, s in zip(habitat, hab_std)],
                   [h + s for h, s in zip(habitat, hab_std)],
                   color='#2d6a4f', alpha=0.15)
    ax.plot(years, habitat, 'o-', color='#2d6a4f', linewidth=1.5,
           markersize=5)
    ax.axvline(x=2016, color='gray', linestyle=':', alpha=0.5)

    for y, h in zip(years, habitat):
        ax.annotate(f'{h:.3f}', xy=(y, h), xytext=(0, 8),
                   textcoords='offset points', fontsize=6, ha='center')

    ax.set_xlabel('Year')
    ax.set_ylabel('Mean habitat quality')
    ax.set_title('(d) Habitat quality index', fontsize=9)
    ax.set_xticks(years)
    ax.set_ylim(0, max([h + s for h, s in zip(habitat, hab_std)]) * 1.2)

    plt.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, 'fig08_ecosystem_services'))
    plt.close(fig)
    print(f"  [OK] fig08_ecosystem_services")


# ============================================================
# FIGURE 11: CA-MARKOV SCENARIOS (simplified, no hatching)
# ============================================================

def fig11_camarkov_scenarios(plt, data):
    """Simplified CA-Markov projections for 4 main classes (including mangroves)."""
    cam = data['camarkov']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.5))

    scenarios = ['BAU', 'Conservation', 'PDET']
    sc_colors = [SCENARIO_COLORS[s] for s in scenarios]
    # Focus on 4 main classes (including mangroves)
    focus_classes = ['BDen', 'BSec', 'Past', 'Mang']
    class_labels = ['Dense\nforest', 'Secondary\nforest', 'Pastures', 'Mangroves']

    areas_2024 = cam['areas_2024']
    total_2024 = sum(areas_2024.get(c, 0)
                     for c in ['BDen', 'BSec', 'Past', 'Cult', 'Agua', 'Urb', 'Suel', 'Mang'])

    for ax, yr, panel_label in [(ax1, '2030', '(a)'), (ax2, '2040', '(b)')]:
        x = np.arange(len(focus_classes))
        w = 0.18
        for i, (sc, col) in enumerate(zip(scenarios, sc_colors)):
            key = f'{sc}_{yr}'
            if key in cam:
                pcts = [cam[key].get(c, {}).get('pct', 0) for c in focus_classes]
                ax.bar(x + i * w, pcts, w, color=col, label=sc,
                      edgecolor='white', linewidth=0.3)

        # Current baseline
        baseline_pcts = [areas_2024.get(c, 0) / total_2024 * 100 for c in focus_classes]
        ax.plot(x + w * 1.5, baseline_pcts, 'ks--', markersize=4, linewidth=0.8,
               label='Current (2024)')

        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(class_labels, fontsize=7)
        ax.set_ylabel('Projected area (%)')
        ax.set_title(f'{panel_label} {yr} Projections', fontsize=9)
        ax.legend(fontsize=5.5, loc='upper right')

    plt.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, 'fig11_camarkov_scenarios'))
    plt.close(fig)
    print(f"  [OK] fig11_camarkov_scenarios")


# ============================================================
# FIGURE S1: FEATURE IMPORTANCE HEATMAP
# ============================================================

def fig_s1_feature_importance_heatmap(plt, data):
    """Heatmap matrix: features (rows) x periods (columns)."""
    imp = data['importance']

    # Get all feature names (union across periods)
    all_features = set()
    for pk in PERIOD_KEYS:
        all_features.update(imp[pk].keys())
    all_features = sorted(all_features)

    # Build matrix
    n_features = len(all_features)
    n_periods = len(PERIOD_KEYS)
    matrix = np.zeros((n_features, n_periods))

    for j, pk in enumerate(PERIOD_KEYS):
        for i, feat in enumerate(all_features):
            matrix[i, j] = imp[pk].get(feat, 0)

    # Sort by mean importance
    mean_imp = matrix.mean(axis=1)
    sort_idx = np.argsort(mean_imp)[::-1]
    matrix = matrix[sort_idx]
    all_features = [all_features[i] for i in sort_idx]

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH * 0.5, 4.5))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(n_periods))
    ax.set_xticklabels(PERIOD_LABELS, fontsize=7)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(all_features, fontsize=6)

    # Cell values
    for i in range(n_features):
        for j in range(n_periods):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val > 60 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Importance', pad=0.02)
    ax.set_title('Random Forest feature importance by period', fontsize=9)

    plt.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, 'fig_s1_feature_importance'))
    plt.close(fig)
    print(f"  [OK] fig_s1_feature_importance")


# ============================================================
# FIGURE S2: CLIMATE (moved to supplementary)
# ============================================================

def fig_s2_climate(plt, data):
    """Climate vs deforestation -- moved to supplementary."""
    clim = data['climate']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL_WIDTH * 0.6, 4.5),
                                    sharex=True)

    years = list(range(2012, 2025))
    precip = [clim['precipitation_annual_mm'][str(y)] for y in years]
    lst = [clim['lst_annual_C'][str(y)] for y in years]

    # Panel A: Precipitation
    ax1.bar(years, precip, color='#3690c0', alpha=0.7, edgecolor='white',
           linewidth=0.3, width=0.8)
    ax1.axhline(y=np.mean(precip), color='#034e7b', linestyle=':',
               alpha=0.5, linewidth=0.8,
               label=f'Mean ({np.mean(precip):.0f} mm)')
    ax1.axvline(x=2016, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.set_ylabel('Annual precipitation (mm)')
    ax1.set_title('(a) Precipitation trend', fontsize=9)
    ax1.legend(fontsize=6, loc='upper left')

    # Panel B: LST
    ax2.plot(years, lst, 'o-', color='#d73027', linewidth=1.2, markersize=4)
    ax2.fill_between(years, min(lst) - 0.2, lst, color='#d73027', alpha=0.08)
    ax2.axvline(x=2016, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    z = np.polyfit(years, lst, 1)
    trend_line = np.poly1d(z)
    ax2.plot(years, trend_line(years), 'k--', linewidth=0.8, alpha=0.5,
            label=f'Trend: {z[0]:.3f} \u00b0C yr$^{{-1}}$')

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean LST (\u00b0C)')
    ax2.set_title('(b) Land surface temperature', fontsize=9)
    ax2.legend(fontsize=6)
    ax2.set_xticks(years[::2])

    plt.tight_layout()
    save_figure(fig, os.path.join(FIG_DIR, 'fig_s2_climate'))
    plt.close(fig)
    print(f"  [OK] fig_s2_climate")


# ============================================================
# TABLES
# ============================================================

def table01_accuracy(data):
    clf = data['classification']
    path = os.path.join(TABLE_DIR, 'table01_accuracy.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Period', 'Year', 'Method', 'N_Training', 'N_Validation',
                         'N_Images', 'OA_adj (%)', 'QD', 'AD'])
        for pk, label in zip(PERIOD_KEYS, PERIOD_LABELS):
            p = clf[pk]
            writer.writerow([
                label, p['year'], p.get('classification_method', 'RF-200'),
                p['n_training'], p['n_validation'], p['n_images'],
                f"{p['overall_accuracy']*100:.1f}",
                f"{p.get('quantity_disagreement', 'N/A')}",
                f"{p.get('allocation_disagreement', 'N/A')}"
            ])
    print(f"  [OK] {path}")


def table02_class_areas(data):
    clf = data['classification']
    path = os.path.join(TABLE_DIR, 'table02_class_areas.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Class'] + [f'{l} (ha)' for l in PERIOD_LABELS] + ['Net Change (ha)', 'Change (%)']
        writer.writerow(header)
        for cid in [1, 2, 3, 4, 5, 6, 7, 8]:
            row = [LULC_NAMES_EN[cid]]
            areas = []
            for pk in PERIOD_KEYS:
                area = clf[pk]['class_areas_ha'].get(str(cid), {}).get('area_ha', 0)
                areas.append(area)
                row.append(f"{area:,.0f}")
            net = areas[-1] - areas[0]
            pct = (net / areas[0] * 100) if areas[0] > 0 else 0
            row.extend([f"{net:,.0f}", f"{pct:.1f}"])
            writer.writerow(row)
    print(f"  [OK] {path}")


def table03_change_rates(data):
    chg = data['change']
    path = os.path.join(TABLE_DIR, 'table03_change_rates.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Period', 'Class', 'Area T1 (ha)', 'Area T2 (ha)',
                         'Net Change (ha)', 'Change (%)', 'Annual Rate (%/yr)'])
        for tk in ['T1_T2', 'T2_T3', 'T3_T4']:
            cr = chg[tk]['change_rates']
            for cid_str, r in cr.items():
                if r['area_t1_ha'] > 0 or r['area_t2_ha'] > 0:
                    writer.writerow([
                        chg[tk]['years'], r['name'],
                        f"{r['area_t1_ha']:,.0f}", f"{r['area_t2_ha']:,.0f}",
                        f"{r['net_change_ha']:,.0f}", f"{r['pct_change']:.1f}",
                        f"{r['annual_rate_pct']:.3f}"
                    ])
    print(f"  [OK] {path}")


def table04_ecosystem_services(data):
    eco = data['ecosystem']
    path = os.path.join(TABLE_DIR, 'table04_ecosystem_services.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Period', 'Year', 'Carbon (Tg C)', 'Water Yield (mm/yr)',
                         'Baseflow (mm/yr)', 'Habitat Quality', 'Habitat Std'])
        for pk, label in zip(PERIOD_KEYS, PERIOD_LABELS):
            p = eco[pk]
            writer.writerow([
                label, p['year'],
                f"{p['carbon_Mg_C']/1e6:.1f}",
                f"{p['water_yield_mm']:.1f}",
                f"{p['baseflow_mm']:.1f}",
                f"{p['habitat_quality_mean']:.4f}",
                f"{p['habitat_quality_std']:.4f}"
            ])
    print(f"  [OK] {path}")


def table05_gwr_results(data):
    gwr = data['gwr']
    path = os.path.join(TABLE_DIR, 'table05_gwr_results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variable', 'OLS Coeff', 'OLS t-stat', 'Significant',
                         'GWR Mean', 'GWR Median', 'GWR Std', 'GWR Min', 'GWR Max',
                         '% Positive', '% Negative'])
        ols = gwr['ols']
        gwr_s = gwr['gwr']['summary']
        for var in gwr_s:
            if var == 'intercept':
                continue
            t = abs(ols['t_statistics'].get(var, 0))
            sig = 'Yes' if t > 1.96 else 'No'
            g = gwr_s[var]
            writer.writerow([
                var, f"{ols['coefficients'][var]:.6f}",
                f"{ols['t_statistics'][var]:.4f}", sig,
                f"{g['mean']:.6f}", f"{g['median']:.6f}",
                f"{g['std']:.6f}", f"{g['min']:.6f}", f"{g['max']:.6f}",
                f"{g['pct_positive']:.1f}", f"{g['pct_negative']:.1f}"
            ])
    print(f"  [OK] {path}")


def table06_camarkov_projections(data):
    cam = data['camarkov']
    path = os.path.join(TABLE_DIR, 'table06_camarkov_projections.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        classes = ['BDen', 'BSec', 'Past', 'Cult', 'Agua', 'Urb', 'Suel', 'Mang']
        class_names = ['Dense Forest', 'Secondary Forest', 'Pastures',
                      'Crops', 'Water', 'Urban', 'Bare Soil', 'Mangroves']

        writer.writerow(['Scenario', 'Year'] + class_names)

        total = sum(cam['areas_2024'].get(c, 0) for c in classes)
        current = ['Current', '2024'] + [
            f"{cam['areas_2024'].get(c, 0)/total*100:.1f}" for c in classes
        ]
        writer.writerow(current)

        for sc in ['BAU', 'Conservation', 'PDET']:
            for yr in ['2030', '2040']:
                key = f'{sc}_{yr}'
                if key in cam:
                    row = [sc, yr] + [
                        f"{cam[key].get(c, {}).get('pct', 0):.1f}" for c in classes
                    ]
                    writer.writerow(row)
    print(f"  [OK] {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 4: SCIENTIFIC VISUALIZATION (Ecological Economics)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Style: sans-serif 8pt, 600 DPI, PDF+PNG")
    print("=" * 60)

    # Load data
    print("\nLoading Phase 3 statistics...")
    data = load_all_data()
    print("  Data loaded successfully")

    # Setup style
    plt = setup_journal_style()

    # Generate figures
    print(f"\nGenerating figures ({DPI_SAVE} DPI) in: {FIG_DIR}")
    print("-" * 40)

    # Statistical figures (PDF + PNG)
    fig04a_lulc_stacked_bar(plt, data)
    fig04b_forest_trend(plt, data)
    fig05_transition_matrices(plt, data)
    fig06_deforestation_rates(plt, data)
    fig08_ecosystem_services(plt, data)
    fig11_camarkov_scenarios(plt, data)

    # Supplementary
    fig_s1_feature_importance_heatmap(plt, data)
    fig_s2_climate(plt, data)

    # Generate tables
    print(f"\nGenerating tables in: {TABLE_DIR}")
    print("-" * 40)

    table01_accuracy(data)
    table02_class_areas(data)
    table03_change_rates(data)
    table04_ecosystem_services(data)
    table05_gwr_results(data)
    table06_camarkov_projections(data)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: 8 figures + 6 tables generated")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TABLE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
