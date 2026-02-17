"""
Shared figure style module for aastex701 preprint / Q1 journal format.
Provides: rcParams, colorblind-safe palettes, size constants, helper functions,
choropleth map rendering utilities.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


# ============================================================
# SIZE CONSTANTS (aastex701 preprint / Elsevier double-column)
# ============================================================

SINGLE_COL_WIDTH = 3.54  # 90 mm
DOUBLE_COL_WIDTH = 7.48  # 190 mm
MAX_HEIGHT = 9.45  # 240 mm

DPI_SAVE = 600
DPI_DISPLAY = 150

# ============================================================
# STUDY AREA BOUNDING BOX
# ============================================================

BBOX = [-77.2, 7.0, -75.8, 8.9]  # [xmin, ymin, xmax, ymax] Uraba Antioqueno

# ============================================================
# COLORBLIND-SAFE LULC PALETTE
# ============================================================

LULC_COLORS = {
    1: '#1b7837',  # Dense forest - dark green
    2: '#7fbf7b',  # Secondary forest - light green
    3: '#e6ab02',  # Pastures - amber/gold
    4: '#ff8c00',  # Crops - orange
    5: '#3690c0',  # Water - blue
    6: '#878787',  # Urban - gray
    7: '#a6611a',  # Bare soil - brown
    8: '#2E8B57',  # Mangroves - sea green
}

LULC_COLORS_HEX = LULC_COLORS  # alias

LULC_NAMES_EN = {
    1: 'Dense forest',
    2: 'Secondary forest',
    3: 'Pastures',
    4: 'Crops',
    5: 'Water',
    6: 'Urban',
    7: 'Bare soil',
    8: 'Mangroves',
}

LULC_NAMES_SHORT = {
    1: 'Dense\nforest',
    2: 'Secondary\nforest',
    3: 'Pastures',
    4: 'Crops',
    5: 'Water',
    6: 'Urban',
    7: 'Bare\nsoil',
    8: 'Mangroves',
}

# Active classes (8-class system for Uraba including Mangroves)
ACTIVE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]

# ============================================================
# PERIOD DEFINITIONS
# ============================================================

PERIOD_KEYS = ['pre_acuerdo', 'transicion', 'post_acuerdo_1', 'post_acuerdo_2']
PERIOD_YEARS = [2013, 2016, 2020, 2024]
PERIOD_LABELS = ['T1 (2013)', 'T2 (2016)', 'T3 (2020)', 'T4 (2024)']
PERIOD_LABELS_SHORT = ['T1\n2013', 'T2\n2016', 'T3\n2020', 'T4\n2024']

# ============================================================
# CARBON POOLS (Tier 2, Mg C/ha)
# ============================================================

CARBON_POOLS = {
    1: {'total': 281, 'se': 27.5},   # Dense forest Choco (155+39+65+22)
    2: {'total': 146, 'se': 19.7},   # Secondary forest (65+16+55+10)
    3: {'total': 43.5, 'se': 8.3},   # Pastures
    4: {'total': 53.5, 'se': 9.7},   # Crops (banana/palm higher: 12+3+38+0.5)
    5: {'total': 0, 'se': 0},        # Water
    6: {'total': 20, 'se': 5.1},     # Urban
    7: {'total': 15, 'se': 4.0},     # Bare soil
    8: {'total': 247, 'se': 44.5},   # Mangroves blue carbon (90+25+120+12)
}

# ============================================================
# SCENARIO COLORS
# ============================================================

SCENARIO_COLORS = {
    'BAU': '#d73027',
    'Conservation': '#1a9850',
    'PDET': '#4575b4',
}

# ============================================================
# DIVERGENT COLORMAPS FOR MAPS
# ============================================================

FOREST_CHANGE_COLORS = {
    'loss': '#d73027',
    'gain': '#1a9850',
    'stable_forest': '#006400',
    'non_forest': '#f0f0f0',
}

HOTSPOT_COLORS = {
    'hot_99': '#d73027',
    'hot_95': '#f46d43',
    'hot_90': '#fdae61',
    'not_sig': 'none',  # transparent
    'cold_90': '#abd9e9',
    'cold_95': '#74add1',
    'cold_99': '#4575b4',
}

# ============================================================
# CHOROPLETH COLOR SCALES (Percentile-based, 9 classes)
# ============================================================

# Deforestation intensity: RdYlGn divergent (red=high deforestation, green=low)
DEFORESTATION_COLORS_9 = [
    '#1a9850',  # Bin 1: <= P10 (lowest deforestation / forest gain)
    '#66bd63',  # Bin 2: P10-P20
    '#a6d96a',  # Bin 3: P20-P30
    '#d9ef8b',  # Bin 4: P30-P40
    '#ffffbf',  # Bin 5: P40-P60 (neutral)
    '#fee08b',  # Bin 6: P60-P70
    '#fdae61',  # Bin 7: P70-P80
    '#f46d43',  # Bin 8: P80-P90
    '#d73027',  # Bin 9: > P90 (highest deforestation)
]

# Carbon change: RdYlGn divergent centered at zero
CARBON_CHANGE_COLORS_9 = DEFORESTATION_COLORS_9  # same divergent scale

# GWR coefficients: RdBu_r divergent centered at zero
GWR_COEFF_COLORS_9 = [
    '#b2182b',  # strong negative
    '#d6604d',
    '#f4a582',
    '#fddbc7',
    '#f7f7f7',  # neutral (near zero)
    '#d1e5f0',
    '#92c5de',
    '#4393c3',
    '#2166ac',  # strong positive
]

# Hotspot Gi* Z-scores: RdYlBu_r
HOTSPOT_ZSCORE_COLORS_9 = [
    '#4575b4',  # strong coldspot
    '#74add1',
    '#abd9e9',
    '#e0f3f8',
    '#ffffbf',  # not significant
    '#fee090',
    '#fdae61',
    '#f46d43',
    '#d73027',  # strong hotspot
]


def make_percentile_cmap(color_list):
    """Create a ListedColormap from a list of hex colors."""
    return mcolors.ListedColormap(color_list)


def compute_percentile_bins(values, k=9):
    """Compute percentile bin edges for k classes.

    Returns bin edges suitable for np.digitize or mapclassify.
    For k=9, uses: P10, P20, P30, P40, P60, P70, P80, P90.
    """
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return np.linspace(0, 1, k + 1)
    percentiles = [10, 20, 30, 40, 60, 70, 80, 90]
    bins = np.percentile(clean, percentiles)
    return bins


def assign_percentile_classes(values, bins):
    """Assign values to percentile bin classes (0 to k-1)."""
    return np.digitize(values, bins, right=True)


# ============================================================
# MATPLOTLIB RCPARAMS SETUP
# ============================================================

def setup_journal_style():
    """Configure matplotlib rcParams for aastex701 preprint (Times font)."""
    plt.rcParams.update({
        # Font settings — Times to match aastex701
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'legend.title_fontsize': 8,

        # Spine and axis styling
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'axes.grid': False,

        # Tick styling
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Line and marker
        'lines.linewidth': 1.2,
        'lines.markersize': 4,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Save settings
        'figure.dpi': DPI_DISPLAY,
        'savefig.dpi': DPI_SAVE,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.format': 'pdf',

        # Math
        'mathtext.default': 'regular',
        'mathtext.fontset': 'stix',
    })
    return plt


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def label_with_halo(ax, x, y, text, fontsize=7, color='white',
                    halo_color='black', halo_width=2, **kwargs):
    """Add text with stroke halo for legibility over imagery."""
    txt = ax.text(x, y, text, fontsize=fontsize, color=color,
                  path_effects=[
                      pe.withStroke(linewidth=halo_width, foreground=halo_color)
                  ], **kwargs)
    return txt


def add_scalebar(ax, lon, lat, length_km=50, fontsize=7):
    """Add a simple scale bar to a map axis (approximate for low latitudes)."""
    deg_per_km = 1.0 / 111.32  # approximate at equator
    bar_length_deg = length_km * deg_per_km

    ax.plot([lon, lon + bar_length_deg], [lat, lat],
            'k-', linewidth=2, transform=ax.transData)
    ax.plot([lon, lon], [lat - 0.02, lat + 0.02],
            'k-', linewidth=1.5)
    ax.plot([lon + bar_length_deg, lon + bar_length_deg],
            [lat - 0.02, lat + 0.02], 'k-', linewidth=1.5)
    ax.text(lon + bar_length_deg / 2, lat - 0.06,
            f'{length_km} km', ha='center', va='top', fontsize=fontsize)


def add_north_arrow(ax, x, y, fontsize=10):
    """Add a north arrow to a map axis (in data coordinates)."""
    ax.annotate('N', xy=(x, y + 0.05), fontsize=fontsize,
                fontweight='bold', ha='center', va='bottom')
    ax.annotate('', xy=(x, y + 0.05), xytext=(x, y - 0.1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))


def save_figure(fig, filepath, also_png=True, also_pdf=True):
    """Save figure in PDF (vector) and/or PNG (raster) formats."""
    import os
    base, _ = os.path.splitext(filepath)

    if also_pdf:
        fig.savefig(base + '.pdf')
        print(f"  [OK] {base}.pdf")

    if also_png:
        fig.savefig(base + '.png', dpi=DPI_SAVE)
        print(f"  [OK] {base}.png")


def save_map_figure(fig, filepath):
    """Save map figure (with raster content) as PNG only at 600 DPI."""
    import os
    base, _ = os.path.splitext(filepath)
    fig.savefig(base + '.png', dpi=DPI_SAVE, facecolor='white')
    print(f"  [OK] {base}.png")
    # Also save PDF for non-raster elements
    fig.savefig(base + '.pdf', dpi=300)
    print(f"  [OK] {base}.pdf")


def format_coord_labels(ax, x_prefix='', y_prefix=''):
    """Format axis tick labels as geographic coordinates."""
    from matplotlib.ticker import FuncFormatter

    def lon_formatter(x, pos):
        if x < 0:
            return f'{abs(x):.1f}\u00b0W'
        return f'{x:.1f}\u00b0E'

    def lat_formatter(y, pos):
        if y < 0:
            return f'{abs(y):.1f}\u00b0S'
        return f'{y:.1f}\u00b0N'

    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))


# ============================================================
# CHOROPLETH MAP HELPERS
# ============================================================

def create_percentile_legend(ax, values, color_list, k=9, title='',
                             loc='lower right', fontsize=6):
    """Create a percentile-based legend with colored patches.

    Parameters
    ----------
    ax : matplotlib Axes
    values : array-like, data values (used to compute percentile ranges)
    color_list : list of hex colors (length k)
    k : int, number of bins
    title : str, legend title
    loc : str, legend location
    fontsize : int, font size
    """
    clean = values[~np.isnan(values)] if hasattr(values, '__len__') else values
    if len(clean) == 0:
        return

    bins = compute_percentile_bins(clean, k)

    labels = []
    labels.append(f'<= {bins[0]:.1f} (P10)')
    for i in range(len(bins) - 1):
        p_lo = [10, 20, 30, 40, 60, 70, 80][i]
        p_hi = [20, 30, 40, 60, 70, 80, 90][i]
        labels.append(f'{bins[i]:.1f} -- {bins[i+1]:.1f} (P{p_lo}--P{p_hi})')
    labels.append(f'> {bins[-1]:.1f} (P90)')

    patches = [mpatches.Patch(facecolor=c, edgecolor='#999999', linewidth=0.3,
                              label=l) for c, l in zip(color_list, labels)]

    # Add no-data patch
    patches.append(mpatches.Patch(facecolor='white', edgecolor='#999999',
                                  linewidth=0.3, label='No data'))

    leg = ax.legend(handles=patches, title=title, loc=loc, fontsize=fontsize,
                    title_fontsize=fontsize + 1, frameon=True, framealpha=0.95,
                    edgecolor='#cccccc', handlelength=1.2, handleheight=0.8,
                    labelspacing=0.3)
    return leg


def render_choropleth(
    gdf, column, cmap_colors, k=9, legend_title='',
    dept_gdf=None, muni_gdf=None, hydro_gdf=None,
    figsize=(7.5, 5.5), output_path=None, title=None,
    scheme='percentile', center_at_zero=False,
):
    """Render a publication-quality choropleth map.

    Parameters
    ----------
    gdf : GeoDataFrame with vereda geometries + stat column
    column : str, column name to plot
    cmap_colors : list of hex colors for the bins
    k : int, number of color classes
    legend_title : str
    dept_gdf : GeoDataFrame, department boundaries overlay
    muni_gdf : GeoDataFrame, municipality boundaries overlay
    hydro_gdf : GeoDataFrame, optional river overlay
    figsize : tuple
    output_path : str, path to save
    title : str, optional figure title
    scheme : str, 'percentile' or 'divergent_zero'
    center_at_zero : bool, if True center colormap at zero
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    values = gdf[column].values.astype(float)
    cmap = make_percentile_cmap(cmap_colors)

    if scheme == 'percentile':
        bins = compute_percentile_bins(values, k)
        classes = assign_percentile_classes(values, bins)
    elif scheme == 'divergent_zero' and center_at_zero:
        vmax = max(abs(np.nanpercentile(values, 5)),
                   abs(np.nanpercentile(values, 95)))
        bins = np.linspace(-vmax, vmax, k + 1)[1:-1]
        classes = assign_percentile_classes(values, bins)
    else:
        bins = compute_percentile_bins(values, k)
        classes = assign_percentile_classes(values, bins)

    # Assign colors
    colors_mapped = [cmap_colors[min(c, k - 1)] if not np.isnan(v) else 'white'
                     for c, v in zip(classes, values)]

    gdf_plot = gdf.copy()
    gdf_plot['_color'] = colors_mapped

    # Plot veredas with assigned colors
    gdf_plot.plot(ax=ax, color=gdf_plot['_color'],
                  edgecolor='#cccccc', linewidth=0.1)

    # Missing data veredas
    nan_mask = np.isnan(values)
    if nan_mask.any():
        gdf_plot[nan_mask].plot(ax=ax, color='white',
                                edgecolor='#999999', linewidth=0.3)

    # Department boundaries (thick black)
    if dept_gdf is not None:
        dept_gdf.boundary.plot(ax=ax, color='black', linewidth=2.0)

    # Municipality boundaries (thin gray)
    if muni_gdf is not None:
        muni_gdf.boundary.plot(ax=ax, color='#333333', linewidth=0.5)

    # Optional rivers
    if hydro_gdf is not None:
        hydro_gdf.plot(ax=ax, color='#4a90d9', linewidth=0.8, alpha=0.7)

    # Clean axes
    ax.set_xlim(BBOX[0], BBOX[2])
    ax.set_ylim(BBOX[1], BBOX[3])
    format_coord_labels(ax)
    ax.set_aspect('equal')

    # Scale bar and north arrow
    add_scalebar(ax, BBOX[0] + 0.1, BBOX[1] + 0.15, length_km=50, fontsize=6)
    add_north_arrow(ax, BBOX[2] - 0.15, BBOX[3] - 0.3, fontsize=8)

    # Title
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Legend
    create_percentile_legend(ax, values, cmap_colors, k, legend_title)

    # Remove top/right spines for map
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.5)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)

    plt.tight_layout()

    if output_path:
        save_map_figure(fig, output_path)

    return fig, ax
