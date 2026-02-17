#!/usr/bin/env python3
"""
Phase 3: Publication-Quality Veredal Choropleth Maps for aastex701 Preprint.

Generates 6 Q1-quality vector choropleth maps at vereda level using
pre-computed zonal statistics from scripts/17_veredal_zonal_stats.py.

Style: White background, percentile-binned divergent color scales, thick
administrative borders, clean legends — matching West Bengal–Odisha reference.

Maps generated:
  1. fig01_study_area_choropleth.png      - Study area + LULC 2024 dominant class
  2. fig02_lulc_choropleth_4panel.png     - Dominant LULC per period (2x2)
  3. fig03_deforestation_choropleth.png   - Deforestation intensity (percentile)
  4. fig06_hotspot_choropleth.png         - Gi* Z-scores per vereda
  5. fig07_carbon_choropleth.png          - Carbon stock change T1->T4
  6. fig10_gwr_choropleth_4panel.png      - GWR local coefficients (2x2)

Requires: geopandas, matplotlib, numpy, scipy
Input:  data/map_exports/veredal_stats.gpkg  (from script 17)
        data/map_exports/departments_clip.gpkg
        data/map_exports/municipalities_clip.gpkg

Usage:
    python scripts/14_generate_cartographic_maps.py
"""

import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.figure_style import (
    setup_journal_style, LULC_COLORS, LULC_NAMES_EN, ACTIVE_CLASSES,
    DOUBLE_COL_WIDTH, DPI_SAVE, BBOX,
    DEFORESTATION_COLORS_9, CARBON_CHANGE_COLORS_9,
    GWR_COEFF_COLORS_9, HOTSPOT_ZSCORE_COLORS_9,
    make_percentile_cmap, compute_percentile_bins, assign_percentile_classes,
    create_percentile_legend, add_scalebar, add_north_arrow,
    format_coord_labels, save_map_figure, label_with_halo,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DATA_DIR = os.path.join(BASE_DIR, 'data', 'map_exports')
FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Data files
VEREDAL_GPKG = os.path.join(MAP_DATA_DIR, 'veredal_stats.gpkg')
DEPTS_GPKG = os.path.join(MAP_DATA_DIR, 'departments_clip.gpkg')
MUNIS_GPKG = os.path.join(MAP_DATA_DIR, 'municipalities_clip.gpkg')
COLOMBIA_GEOJSON = os.path.join(MAP_DATA_DIR, 'colombia_outline.geojson')

# Key cities for context panel
CITIES = {
    'Bogotá': (-74.1, 4.6),
    'Medellín': (-75.6, 6.25),
    'Cartagena': (-75.5, 10.4),
}

# Municipality labels (Urabá Antioqueño)
MUNICIPALITIES = {
    'Apartadó': (-76.63, 7.88),
    'Turbo': (-76.73, 8.10),
    'Chigorodó': (-76.68, 7.67),
    'Carepa': (-76.65, 7.76),
    'Necoclí': (-76.78, 8.43),
    'Mutatá': (-76.44, 7.24),
    'Dabeiba': (-76.26, 7.00),
    'Riosucio': (-77.12, 7.44),
}

# Department labels
DEPT_LABELS = {
    'Antioquia': (-76.3, 7.8),
    'Chocó': (-77.0, 7.3),
}

# Colombia simplified outline (fallback)
COL_LON = [-77, -75.5, -73, -72, -67, -67, -70, -71.5, -73.5, -76, -77.5, -79, -77]
COL_LAT = [1.5, -0.5, -1.5, -3.5, -1, 2, 4.5, 7, 11, 11, 8, 2, 1.5]


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load all GeoPackage data files."""
    data = {}

    if os.path.isfile(VEREDAL_GPKG):
        print(f"  Loading veredas: {VEREDAL_GPKG}")
        data['veredas'] = gpd.read_file(VEREDAL_GPKG)
        print(f"    -> {len(data['veredas'])} veredas, "
              f"{len(data['veredas'].columns)} columns")
    else:
        print(f"  [WARN] Veredal stats not found: {VEREDAL_GPKG}")
        print("         Run scripts/17_veredal_zonal_stats.py first.")
        return None

    if os.path.isfile(DEPTS_GPKG):
        data['depts'] = gpd.read_file(DEPTS_GPKG)
        print(f"  Loaded {len(data['depts'])} departments")
    else:
        data['depts'] = None
        print("  [WARN] Department boundaries not found")

    if os.path.isfile(MUNIS_GPKG):
        data['munis'] = gpd.read_file(MUNIS_GPKG)
        print(f"  Loaded {len(data['munis'])} municipalities")
    else:
        data['munis'] = None
        print("  [WARN] Municipality boundaries not found")

    if os.path.isfile(COLOMBIA_GEOJSON):
        data['colombia'] = gpd.read_file(COLOMBIA_GEOJSON)
    else:
        data['colombia'] = None

    return data


def _add_admin_overlays(ax, depts, munis):
    """Add department (thick) and municipality (thin) boundary overlays."""
    if depts is not None:
        depts.boundary.plot(ax=ax, color='black', linewidth=2.0)
    if munis is not None:
        munis.boundary.plot(ax=ax, color='#333333', linewidth=0.5)


def _setup_map_axes(ax, title=None, show_all_spines=True):
    """Configure axes for a choropleth map panel."""
    ax.set_facecolor('white')
    ax.set_xlim(BBOX[0], BBOX[2])
    ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_aspect('equal')
    format_coord_labels(ax)
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    if show_all_spines:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)


def _add_map_furniture(ax):
    """Add scale bar and north arrow."""
    add_scalebar(ax, BBOX[0] + 0.1, BBOX[1] + 0.15, length_km=50, fontsize=6)
    add_north_arrow(ax, BBOX[2] - 0.15, BBOX[3] - 0.3, fontsize=8)


# ============================================================
# MAP 1: STUDY AREA CHOROPLETH
# ============================================================

def map01_study_area(data):
    """Study area with (a) Colombia context and (b) LULC 2024 dominant class."""
    print("\n--- Generating Map 1: Study Area Choropleth ---")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.8]})

    veredas = data['veredas']
    depts = data['depts']
    munis = data['munis']

    # --- Panel (a): Colombia context ---
    ax1 = axes[0]
    ax1.set_facecolor('white')

    if data['colombia'] is not None:
        data['colombia'].plot(ax=ax1, color='#E8E8E8', edgecolor='black',
                              linewidth=0.6)
    else:
        ax1.fill(COL_LON, COL_LAT, color='#E8E8E8', edgecolor='black',
                 linewidth=0.6)

    # Study area rectangle
    rect = mpatches.Rectangle(
        (BBOX[0], BBOX[1]), BBOX[2] - BBOX[0], BBOX[3] - BBOX[1],
        linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.25
    )
    ax1.add_patch(rect)
    ax1.annotate('Urabá\nAntioqueño', xy=(-76.5, 7.9), fontsize=7,
                 ha='center', va='center', fontweight='bold', color='darkred')

    for city, (lon, lat) in CITIES.items():
        ax1.plot(lon, lat, 'ko', markersize=2.5)
        ax1.annotate(city, xy=(lon, lat), xytext=(4, 2),
                     textcoords='offset points', fontsize=5.5)

    ax1.set_xlim(-80, -66)
    ax1.set_ylim(-5, 13)
    ax1.set_title('(a) Location in Colombia', fontsize=9)
    format_coord_labels(ax1)
    ax1.set_aspect('equal')

    # --- Panel (b): Dominant LULC 2024 choropleth ---
    ax2 = axes[1]
    ax2.set_facecolor('white')

    col_name = 'dominant_lulc_T4'
    if col_name in veredas.columns:
        # Build categorical color map
        colors_mapped = []
        for val in veredas[col_name]:
            if np.isnan(val):
                colors_mapped.append('white')
            else:
                colors_mapped.append(LULC_COLORS.get(int(val), '#cccccc'))
        veredas_plot = veredas.copy()
        veredas_plot['_color'] = colors_mapped

        veredas_plot.plot(ax=ax2, color=veredas_plot['_color'],
                         edgecolor='#cccccc', linewidth=0.05)

    _add_admin_overlays(ax2, depts, munis)

    # Department labels
    for dept, (lon, lat) in DEPT_LABELS.items():
        ax2.text(lon, lat, dept, fontsize=7, fontstyle='italic',
                 ha='center', va='center', color='#333333',
                 fontweight='bold', alpha=0.7)

    # Municipality labels
    for mun, (lon, lat) in MUNICIPALITIES.items():
        ax2.plot(lon, lat, 'ks', markersize=2)
        ax2.text(lon + 0.04, lat + 0.04, mun, fontsize=5,
                 ha='left', va='bottom', color='#333333')

    _add_map_furniture(ax2)

    # Area annotation
    ax2.text(BBOX[0] + 0.05, BBOX[1] + 0.45,
             r'${\sim}$11,664 km$^2$' + '\n15 municipalities\n2 departments',
             fontsize=6, va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.85, edgecolor='#cccccc'))

    # LULC legend
    legend_elements = [mpatches.Patch(facecolor=LULC_COLORS[c], edgecolor='black',
                                       linewidth=0.5, label=LULC_NAMES_EN[c])
                       for c in ACTIVE_CLASSES]
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='#999999',
                                           linewidth=0.5, label='No data'))
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=6,
               frameon=True, framealpha=0.95, edgecolor='#cccccc')

    _setup_map_axes(ax2, '(b) Dominant LULC 2024')

    plt.tight_layout()
    save_map_figure(fig, os.path.join(FIG_DIR, 'fig01_study_area_choropleth'))
    plt.close(fig)


# ============================================================
# MAP 2: LULC 4-PANEL CHOROPLETH
# ============================================================

def map02_lulc_4panel(data):
    """2x2 dominant LULC choropleth per period."""
    print("\n--- Generating Map 2: LULC 4-Panel Choropleth ---")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 8.0))
    axes = axes.flatten()

    veredas = data['veredas']
    depts = data['depts']
    munis = data['munis']

    periods = [
        ('dominant_lulc_T1', '(a) T1: 2013'),
        ('dominant_lulc_T2', '(b) T2: 2016'),
        ('dominant_lulc_T3', '(c) T3: 2020'),
        ('dominant_lulc_T4', '(d) T4: 2024'),
    ]

    for idx, (col_name, title) in enumerate(periods):
        ax = axes[idx]
        ax.set_facecolor('white')

        if col_name in veredas.columns:
            colors_mapped = []
            for val in veredas[col_name]:
                if np.isnan(val):
                    colors_mapped.append('white')
                else:
                    colors_mapped.append(LULC_COLORS.get(int(val), '#cccccc'))
            veredas_plot = veredas.copy()
            veredas_plot['_color'] = colors_mapped
            veredas_plot.plot(ax=ax, color=veredas_plot['_color'],
                             edgecolor='#cccccc', linewidth=0.05)
        else:
            ax.text(0.5, 0.5, f'Column {col_name}\nnot available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='gray')

        _add_admin_overlays(ax, depts, munis)
        _setup_map_axes(ax, title)

    # Shared legend at bottom
    legend_elements = [mpatches.Patch(facecolor=LULC_COLORS[c], edgecolor='black',
                                       linewidth=0.5, label=LULC_NAMES_EN[c])
                       for c in ACTIVE_CLASSES]
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='#999999',
                                           linewidth=0.5, label='No data'))
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=6, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.01), edgecolor='0.8')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_map_figure(fig, os.path.join(FIG_DIR, 'fig02_lulc_choropleth_4panel'))
    plt.close(fig)


# ============================================================
# MAP 3: DEFORESTATION INTENSITY CHOROPLETH
# ============================================================

def map03_deforestation(data):
    """Deforestation rate (%) T1->T4 with percentile-based 9-class binning."""
    print("\n--- Generating Map 3: Deforestation Intensity Choropleth ---")

    veredas = data['veredas']
    depts = data['depts']
    munis = data['munis']

    col_name = 'defor_rate_pct'
    if col_name not in veredas.columns:
        print("  [WARN] Deforestation rate column not found. Skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH, 5.5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    values = veredas[col_name].values.astype(float)
    cmap_colors = DEFORESTATION_COLORS_9
    k = 9

    # Compute percentile bins
    clean = values[~np.isnan(values)]
    if len(clean) > 0:
        bins = compute_percentile_bins(clean, k)
        classes = assign_percentile_classes(values, bins)

        colors_mapped = []
        for c, v in zip(classes, values):
            if np.isnan(v):
                colors_mapped.append('white')
            else:
                colors_mapped.append(cmap_colors[min(c, k - 1)])

        veredas_plot = veredas.copy()
        veredas_plot['_color'] = colors_mapped
        veredas_plot.plot(ax=ax, color=veredas_plot['_color'],
                         edgecolor='#cccccc', linewidth=0.05)

    _add_admin_overlays(ax, depts, munis)
    _add_map_furniture(ax)
    _setup_map_axes(ax, 'Deforestation rate 2013\u20132024 (%)')

    # Percentile legend
    create_percentile_legend(ax, values, cmap_colors, k,
                             title='Deforestation rate (%)',
                             loc='lower right', fontsize=5.5)

    plt.tight_layout()
    save_map_figure(fig, os.path.join(FIG_DIR, 'fig03_deforestation_choropleth'))
    plt.close(fig)


# ============================================================
# MAP 4: HOTSPOT Gi* CHOROPLETH
# ============================================================

def map04_hotspot(data):
    """Hotspot Gi* Z-scores interpolated to vereda level."""
    print("\n--- Generating Map 4: Hotspot Gi* Choropleth ---")

    veredas = data['veredas']
    depts = data['depts']
    munis = data['munis']

    col_name = 'hotspot_zscore'
    if col_name not in veredas.columns:
        print("  [WARN] Hotspot Z-score column not found. Skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH, 5.5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    values = veredas[col_name].values.astype(float)
    cmap_colors = HOTSPOT_ZSCORE_COLORS_9
    k = 9

    clean = values[~np.isnan(values)]
    if len(clean) > 0:
        # Use fixed significance thresholds for Gi* Z-scores
        # instead of percentiles for interpretability
        bins = np.array([-2.576, -1.96, -1.645, -0.5, 0.5, 1.645, 1.96, 2.576])
        classes = np.digitize(values, bins, right=True)

        colors_mapped = []
        for c, v in zip(classes, values):
            if np.isnan(v):
                colors_mapped.append('white')
            else:
                colors_mapped.append(cmap_colors[min(c, k - 1)])

        veredas_plot = veredas.copy()
        veredas_plot['_color'] = colors_mapped
        veredas_plot.plot(ax=ax, color=veredas_plot['_color'],
                         edgecolor='#cccccc', linewidth=0.05)

    _add_admin_overlays(ax, depts, munis)
    _add_map_furniture(ax)
    _setup_map_axes(ax, 'Deforestation hotspots (Getis-Ord Gi*)')

    # Fixed significance-level legend
    sig_labels = [
        'Coldspot 99% (Z < -2.58)',
        'Coldspot 95% (-2.58 to -1.96)',
        'Coldspot 90% (-1.96 to -1.65)',
        'Moderate cold (-1.65 to -0.5)',
        'Not significant (-0.5 to 0.5)',
        'Moderate hot (0.5 to 1.65)',
        'Hotspot 90% (1.65 to 1.96)',
        'Hotspot 95% (1.96 to 2.58)',
        'Hotspot 99% (Z > 2.58)',
    ]
    patches = [mpatches.Patch(facecolor=c, edgecolor='#999999', linewidth=0.3,
                               label=l) for c, l in zip(cmap_colors, sig_labels)]
    patches.append(mpatches.Patch(facecolor='white', edgecolor='#999999',
                                   linewidth=0.3, label='No data'))
    ax.legend(handles=patches, title='Gi* Z-score', loc='lower right',
              fontsize=5, title_fontsize=6, frameon=True, framealpha=0.95,
              edgecolor='#cccccc', handlelength=1.2, handleheight=0.8,
              labelspacing=0.3)

    plt.tight_layout()
    save_map_figure(fig, os.path.join(FIG_DIR, 'fig06_hotspot_choropleth'))
    plt.close(fig)


# ============================================================
# MAP 5: CARBON CHANGE CHOROPLETH
# ============================================================

def map05_carbon(data):
    """Carbon stock change (Mg C/ha) T1->T4 with divergent 9-class binning."""
    print("\n--- Generating Map 5: Carbon Change Choropleth ---")

    veredas = data['veredas']
    depts = data['depts']
    munis = data['munis']

    col_name = 'carbon_change_MgC_ha'
    if col_name not in veredas.columns:
        print("  [WARN] Carbon change column not found. Skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH, 5.5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    values = veredas[col_name].values.astype(float)
    cmap_colors = CARBON_CHANGE_COLORS_9
    k = 9

    clean = values[~np.isnan(values)]
    if len(clean) > 0:
        # Divergent bins centered at zero
        vmax = max(abs(np.nanpercentile(clean, 5)),
                   abs(np.nanpercentile(clean, 95)))
        if vmax == 0:
            vmax = 100
        bins = np.linspace(-vmax, vmax, k + 1)[1:-1]
        classes = np.digitize(values, bins, right=True)

        colors_mapped = []
        for c, v in zip(classes, values):
            if np.isnan(v):
                colors_mapped.append('white')
            else:
                colors_mapped.append(cmap_colors[min(c, k - 1)])

        veredas_plot = veredas.copy()
        veredas_plot['_color'] = colors_mapped
        veredas_plot.plot(ax=ax, color=veredas_plot['_color'],
                         edgecolor='#cccccc', linewidth=0.05)

    _add_admin_overlays(ax, depts, munis)
    _add_map_furniture(ax)
    _setup_map_axes(ax, r'Carbon stock change 2013$\rightarrow$2024')

    # Divergent legend
    if len(clean) > 0:
        bin_edges = np.concatenate([[-vmax], bins, [vmax]])
        labels = []
        for i in range(k):
            labels.append(f'{bin_edges[i]:.0f} to {bin_edges[i+1]:.0f}')
        patches = [mpatches.Patch(facecolor=c, edgecolor='#999999',
                                   linewidth=0.3, label=l)
                   for c, l in zip(cmap_colors, labels)]
        patches.append(mpatches.Patch(facecolor='white', edgecolor='#999999',
                                       linewidth=0.3, label='No data'))
        ax.legend(handles=patches,
                  title=r'Carbon change (Mg C ha$^{-1}$)',
                  loc='lower right', fontsize=5, title_fontsize=6,
                  frameon=True, framealpha=0.95, edgecolor='#cccccc',
                  handlelength=1.2, handleheight=0.8, labelspacing=0.3)

    plt.tight_layout()
    save_map_figure(fig, os.path.join(FIG_DIR, 'fig07_carbon_choropleth'))
    plt.close(fig)


# ============================================================
# MAP 6: GWR COEFFICIENT 4-PANEL CHOROPLETH
# ============================================================

def map06_gwr_coefficients(data):
    """2x2 panel of GWR local coefficients interpolated to vereda level."""
    print("\n--- Generating Map 6: GWR Coefficients 4-Panel Choropleth ---")

    veredas = data['veredas']
    depts = data['depts']
    munis = data['munis']

    # Find GWR beta columns
    gwr_cols = [c for c in veredas.columns if c.startswith('gwr_beta_')]
    if not gwr_cols:
        print("  [WARN] No GWR beta columns found. Skipping.")
        return

    # Select top 4 variables (or all if fewer)
    # Prefer these names if available
    preferred = ['gwr_beta_elevation', 'gwr_beta_lst',
                 'gwr_beta_dist_rivers', 'gwr_beta_dist_roads']
    target_cols = [c for c in preferred if c in gwr_cols]
    # Fill remaining with other available columns
    for c in gwr_cols:
        if c not in target_cols:
            target_cols.append(c)
    target_cols = target_cols[:4]

    panel_labels = {
        'gwr_beta_elevation': r'Elevation ($\beta$)',
        'gwr_beta_lst': r'LST ($\beta$)',
        'gwr_beta_dist_rivers': r'Dist. rivers ($\beta$)',
        'gwr_beta_dist_roads': r'Dist. roads ($\beta$)',
    }

    n_panels = len(target_cols)
    ncols = 2
    nrows = (n_panels + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL_WIDTH, 3.2 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap_colors = GWR_COEFF_COLORS_9
    k = 9

    for idx, col_name in enumerate(target_cols):
        ax = axes[idx]
        ax.set_facecolor('white')

        values = veredas[col_name].values.astype(float)
        clean = values[~np.isnan(values)]

        if len(clean) > 0:
            # Divergent bins centered at zero
            vmax = max(abs(np.nanpercentile(clean, 5)),
                       abs(np.nanpercentile(clean, 95)))
            if vmax == 0:
                vmax = 1
            bins = np.linspace(-vmax, vmax, k + 1)[1:-1]
            classes = np.digitize(values, bins, right=True)

            colors_mapped = []
            for c, v in zip(classes, values):
                if np.isnan(v):
                    colors_mapped.append('white')
                else:
                    colors_mapped.append(cmap_colors[min(c, k - 1)])

            veredas_plot = veredas.copy()
            veredas_plot['_color'] = colors_mapped
            veredas_plot.plot(ax=ax, color=veredas_plot['_color'],
                             edgecolor='#cccccc', linewidth=0.05)

            # Colorbar-style legend
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            sm = ScalarMappable(cmap='RdBu_r', norm=Normalize(-vmax, vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
            cbar.set_label('Local coefficient', fontsize=6)
            cbar.ax.tick_params(labelsize=5)

        _add_admin_overlays(ax, depts, munis)

        letter = chr(97 + idx)
        label = panel_labels.get(col_name, col_name.replace('gwr_beta_', ''))
        _setup_map_axes(ax, f'({letter}) {label}')

    # Hide extra axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_map_figure(fig, os.path.join(FIG_DIR, 'fig10_gwr_choropleth_4panel'))
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    from datetime import datetime
    print("=" * 60)
    print("CARTOGRAPHIC MAP GENERATION (Veredal Choropleth)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Data source: {MAP_DATA_DIR}")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    # Setup style
    setup_journal_style()

    # Load data
    print("\nLoading data...")
    data = load_data()
    if data is None:
        print("\nERROR: Cannot proceed without veredal stats. Exiting.")
        sys.exit(1)

    # Generate all 6 choropleth maps
    map01_study_area(data)
    map02_lulc_4panel(data)
    map03_deforestation(data)
    map04_hotspot(data)
    map05_carbon(data)
    map06_gwr_coefficients(data)

    print(f"\n{'=' * 60}")
    print("CHOROPLETH MAP GENERATION COMPLETE")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
