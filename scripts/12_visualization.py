"""
12_visualization.py
===================
Fase 4: Visualizacion cientifica para publicacion Q1.

Genera todas las figuras del paper:
- Mapas LULC multi-temporales (4 periodos)
- Mapas de cambio con matrices de transicion
- Graficas de tendencias temporales
- Mapas de hotspots y servicios ecosistemicos
- Coeficientes GWR espacializados
- Escenarios futuros CA-Markov

Todas las figuras: 300 DPI, formato journal-ready.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, LULC_CLASSES, CARBON_POOLS

# Colores de clases LULC para matplotlib
LULC_COLORS = {k: v['color'] for k, v in LULC_CLASSES.items()}
LULC_NAMES = {k: v['name'] for k, v in LULC_CLASSES.items()}


# ============================================================
# CONFIGURACION GLOBAL DE FIGURAS
# ============================================================

def setup_figure_style():
    """
    Configura estilo de matplotlib para publicacion Q1.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })

    return plt


# ============================================================
# FIGURA 1: AREA DE ESTUDIO
# ============================================================

def plot_study_area(output_dir):
    """
    Mapa del area de estudio con municipios, rios principales y contexto regional.
    """
    plt = setup_figure_style()
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5),
                             gridspec_kw={'width_ratios': [1, 2]})

    # Panel A: Colombia con ubicacion del area de estudio
    ax1 = axes[0]
    ax1.set_xlim(-82, -66)
    ax1.set_ylim(-5, 14)
    ax1.set_title('(a) Ubicacion en Colombia')
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    # Placeholder: area de estudio como rectangulo
    rect = mpatches.Rectangle((-75.0, 6.0), 1.5, 2.0,
                               linewidth=2, edgecolor='red',
                               facecolor='red', alpha=0.3)
    ax1.add_patch(rect)
    ax1.text(-74.25, 7.0, 'MM', ha='center', fontsize=8, fontweight='bold')

    # Panel B: Detalle del area de estudio
    ax2 = axes[1]
    ax2.set_xlim(-75.1, -73.4)
    ax2.set_ylim(5.9, 8.1)
    ax2.set_title('(b) Magdalena Medio - Area de estudio')
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    path = os.path.join(output_dir, 'fig01_study_area.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 2: MAPAS LULC 4 PERIODOS
# ============================================================

def plot_lulc_maps(lulc_arrays, output_dir):
    """
    Panel 2x2 con mapas LULC clasificados para los 4 periodos.

    Args:
        lulc_arrays: dict {period_key: numpy array 2D}
    """
    plt = setup_figure_style()
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = [LULC_COLORS[i] for i in range(1, 8)]
    cmap = ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 8))
    labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, (period_key, period_info) in enumerate(PERIODS.items()):
        ax = axes[idx // 2, idx % 2]
        year = period_info['map_year']
        title = f"{labels[idx]} {period_info['label']}"

        if period_key in lulc_arrays:
            im = ax.imshow(lulc_arrays[period_key], cmap=cmap, norm=norm,
                          interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Columna (pixel)')
        ax.set_ylabel('Fila (pixel)')

    # Leyenda comun
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=LULC_COLORS[i],
                                      label=LULC_NAMES[i])
                       for i in range(1, 8)]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    path = os.path.join(output_dir, 'fig02_lulc_maps.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 3: AREAS POR CLASE Y PERIODO
# ============================================================

def plot_area_trends(areas_by_period, output_dir):
    """
    Grafica de barras agrupadas + lineas de tendencia por clase LULC.

    Args:
        areas_by_period: dict {period_key: {class_id: area_ha}}
    """
    plt = setup_figure_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4))

    years = [PERIODS[k]['map_year'] for k in PERIODS]
    class_ids = list(range(1, 8))

    # Panel A: barras apiladas
    bottom = np.zeros(len(years))
    for cid in class_ids:
        values = [areas_by_period.get(pk, {}).get(cid, 0) for pk in PERIODS]
        ax1.bar(years, values, bottom=bottom, color=LULC_COLORS[cid],
                label=LULC_NAMES[cid], width=2)
        bottom += np.array(values)

    ax1.set_xlabel('Anio')
    ax1.set_ylabel('Area (ha)')
    ax1.set_title('(a) Distribucion de cobertura')
    ax1.legend(fontsize=7, loc='upper right')

    # Panel B: cambio neto bosque (clases 1+2)
    forest_areas = []
    for pk in PERIODS:
        f1 = areas_by_period.get(pk, {}).get(1, 0)
        f2 = areas_by_period.get(pk, {}).get(2, 0)
        forest_areas.append(f1 + f2)

    ax2.plot(years, forest_areas, 'g-o', linewidth=2, markersize=8, label='Bosque total')
    ax2.fill_between(years, forest_areas, alpha=0.2, color='green')
    ax2.axvline(x=2016, color='red', linestyle='--', alpha=0.7, label='Acuerdo de paz')
    ax2.set_xlabel('Anio')
    ax2.set_ylabel('Area de bosque (ha)')
    ax2.set_title('(b) Tendencia cobertura forestal')
    ax2.legend(fontsize=8)

    plt.tight_layout()

    path = os.path.join(output_dir, 'fig03_area_trends.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 4: MATRICES DE TRANSICION
# ============================================================

def plot_transition_matrices(matrices, output_dir):
    """
    Heatmaps de matrices de transicion (3 intervalos).

    Args:
        matrices: dict {interval_key: numpy array (7,7)}
    """
    plt = setup_figure_style()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5))
    class_labels = [LULC_NAMES[i][:8] for i in range(1, 8)]
    titles = ['(a) 2013-2016', '(b) 2016-2020', '(c) 2020-2024']

    for idx, (key, matrix) in enumerate(matrices.items()):
        ax = axes[idx]
        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(class_labels, fontsize=7)
        ax.set_title(titles[idx], fontsize=10)
        ax.set_xlabel('Destino')
        if idx == 0:
            ax.set_ylabel('Origen')

        # Valores en celdas
        for i in range(7):
            for j in range(7):
                val = matrix[i, j]
                if val > 0.01:
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=5, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label='Probabilidad')
    plt.tight_layout()

    path = os.path.join(output_dir, 'fig04_transition_matrices.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 5: TASAS DE DEFORESTACION
# ============================================================

def plot_deforestation_rates(rates_by_period, output_dir):
    """
    Tasas anuales de deforestacion por periodo con barras de error.
    """
    plt = setup_figure_style()

    fig, ax = plt.subplots(figsize=(5, 4))

    periods = list(rates_by_period.keys())
    rates = [rates_by_period[p]['annual_rate'] for p in periods]
    labels = [f"{PERIODS[p]['map_year']}" for p in periods]

    colors = ['#2ecc71' if r < 1 else '#e74c3c' if r > 2 else '#f39c12' for r in rates]

    bars = ax.bar(labels, rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Acuerdo de paz')
    ax.set_ylabel('Tasa de deforestacion (%/anio)')
    ax.set_xlabel('Periodo')
    ax.set_title('Tasas anuales de deforestacion')

    plt.tight_layout()

    path = os.path.join(output_dir, 'fig05_deforestation_rates.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 6: HOTSPOTS Gi*
# ============================================================

def plot_hotspot_maps(z_scores, coordinates, output_dir):
    """
    Mapas de hotspots/coldspots Gi* para deforestacion.
    """
    plt = setup_figure_style()
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4))
    titles = ['(a) 2013-2016', '(b) 2016-2020', '(c) 2020-2024']
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    for idx, (key, zscores) in enumerate(z_scores.items()):
        ax = axes[idx]
        coords = coordinates[key]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=zscores,
                       cmap='RdYlBu_r', norm=norm, s=10, edgecolors='none')
        ax.set_title(titles[idx], fontsize=10)
        ax.set_xlabel('Longitud')
        if idx == 0:
            ax.set_ylabel('Latitud')

    fig.colorbar(sc, ax=axes, shrink=0.8, label='Gi* Z-score')
    plt.tight_layout()

    path = os.path.join(output_dir, 'fig06_hotspots.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 7: CARBONO Y SERVICIOS ECOSISTEMICOS
# ============================================================

def plot_ecosystem_services(carbon_data, habitat_data, output_dir):
    """
    Panel con carbono almacenado, cambio de carbono y calidad de habitat.
    """
    plt = setup_figure_style()

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))

    # (a) Carbono total por periodo
    ax = axes[0, 0]
    ax.set_title('(a) Carbono almacenado')

    # (b) Perdida de carbono por transicion
    ax = axes[0, 1]
    ax.set_title('(b) Cambio neto de carbono')

    # (c) Calidad de habitat
    ax = axes[1, 0]
    ax.set_title('(c) Calidad de habitat 2024')

    # (d) Tendencia de servicios
    ax = axes[1, 1]
    ax.set_title('(d) Tendencia de servicios ecosistemicos')

    plt.tight_layout()

    path = os.path.join(output_dir, 'fig07_ecosystem_services.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 8: GWR COEFICIENTES
# ============================================================

def plot_gwr_coefficients(gwr_betas, coordinates, variable_names, output_dir):
    """
    Mapas de coeficientes GWR espacializados.
    """
    plt = setup_figure_style()

    n_vars = min(len(variable_names), 6)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()

    for idx in range(n_vars):
        ax = axes[idx]
        betas = gwr_betas[:, idx + 1]  # +1 por intercepto
        sc = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                       c=betas, cmap='RdYlBu_r', s=5)
        ax.set_title(variable_names[idx], fontsize=9)
        plt.colorbar(sc, ax=ax, shrink=0.8)

    # Ocultar axes sobrantes
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Coeficientes GWR espacializados', fontsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, 'fig08_gwr_coefficients.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 9: ESCENARIOS FUTUROS
# ============================================================

def plot_future_scenarios(scenarios_2030, scenarios_2040, output_dir):
    """
    Comparacion de escenarios CA-Markov: BAU, Conservation, PDET.
    """
    plt = setup_figure_style()

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 6))

    scenario_names = ['BAU', 'Conservation', 'PDET']
    row_titles = ['2030', '2040']

    for row, (year, scenarios) in enumerate(
        [(2030, scenarios_2030), (2040, scenarios_2040)]
    ):
        for col, name in enumerate(scenario_names):
            ax = axes[row, col]
            label = f"({'abcdef'[row * 3 + col]}) {name} {year}"
            ax.set_title(label, fontsize=9)

    plt.tight_layout()

    path = os.path.join(output_dir, 'fig09_future_scenarios.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# FIGURA 10: CLIMA Y DEFORESTACION
# ============================================================

def plot_climate_deforestation(precip_series, lst_series, defor_series, output_dir):
    """
    Series temporales de clima vs deforestacion.
    """
    plt = setup_figure_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)

    years = range(2012, 2025)

    # Panel A: Precipitacion + deforestacion
    ax1.set_title('(a) Precipitacion y deforestacion')
    ax1_twin = ax1.twinx()
    ax1.set_ylabel('Precipitacion (mm/anio)', color='blue')
    ax1_twin.set_ylabel('Deforestacion (ha/anio)', color='red')

    # Panel B: LST + deforestacion
    ax2.set_title('(b) Temperatura superficial y deforestacion')
    ax2.set_xlabel('Anio')
    ax2.axvline(x=2016, color='gray', linestyle='--', alpha=0.5, label='Acuerdo de paz')

    plt.tight_layout()

    path = os.path.join(output_dir, 'fig10_climate_deforestation.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ============================================================
# TABLAS AUXILIARES
# ============================================================

def generate_accuracy_table(metrics_by_period, output_dir):
    """
    Genera tabla de accuracy (CSV) para el paper.
    """
    import csv

    path = os.path.join(output_dir, 'table01_accuracy.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Period', 'Year', 'OA (%)', 'Kappa',
                         'N_Training', 'N_Validation'])
        for pk, metrics in metrics_by_period.items():
            writer.writerow([
                PERIODS[pk]['label'],
                metrics.get('year', ''),
                f"{metrics.get('overall_accuracy', 0) * 100:.1f}",
                f"{metrics.get('kappa', 0):.4f}",
                metrics.get('n_training', ''),
                metrics.get('n_validation', ''),
            ])
    print(f"  Tabla guardada: {path}")


def generate_change_rates_table(rates, output_dir):
    """
    Genera tabla de tasas de cambio por clase y periodo.
    """
    import csv

    path = os.path.join(output_dir, 'table02_change_rates.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Transition', 'Class', 'Area_T1 (ha)', 'Area_T2 (ha)',
                         'Net_Change (ha)', 'Annual_Rate (%/yr)'])
        for trans_key, class_rates in rates.items():
            for cid, r in class_rates.items():
                writer.writerow([
                    trans_key, r['name'], r['area_t1_ha'], r['area_t2_ha'],
                    r['net_change_ha'], r['annual_rate_pct'],
                ])
    print(f"  Tabla guardada: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 4: VISUALIZACION CIENTIFICA")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'figures'
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDirectorio de figuras: {output_dir}")
    print("\nFiguras planificadas:")

    figures = {
        'fig01': 'Area de estudio con municipios y contexto regional',
        'fig02': 'Mapas LULC 4 periodos (2x2 panel)',
        'fig03': 'Tendencias de area por clase + bosque total',
        'fig04': 'Matrices de transicion (3 heatmaps)',
        'fig05': 'Tasas de deforestacion por periodo',
        'fig06': 'Hotspots Gi* de deforestacion (3 mapas)',
        'fig07': 'Servicios ecosistemicos (carbono + habitat)',
        'fig08': 'Coeficientes GWR espacializados',
        'fig09': 'Escenarios futuros CA-Markov (2x3)',
        'fig10': 'Series temporales clima vs deforestacion',
    }

    for fig_id, desc in figures.items():
        print(f"  {fig_id}: {desc}")

    print("\nTablas planificadas:")
    tables = {
        'table01': 'Accuracy assessment (OA, Kappa, F1 por clase)',
        'table02': 'Tasas de cambio por clase y periodo',
        'table03': 'Metricas de fragmentacion por periodo',
        'table04': 'Carbono almacenado y perdido por periodo',
        'table05': 'Resultados GWR (OLS vs GWR, coeficientes)',
        'table06': 'Proyecciones CA-Markov (areas por escenario)',
    }

    for tab_id, desc in tables.items():
        print(f"  {tab_id}: {desc}")

    print("\nEspecificaciones:")
    print("  Formato: PNG 300 DPI")
    print("  Fuente: Times New Roman / serif")
    print("  Ancho maximo: 190 mm (full page) / 90 mm (single column)")
    print("  Colormap: RdYlBu_r (hotspots), YlOrRd (transiciones)")

    # Guardar configuracion
    viz_config = {
        'figures': figures,
        'tables': tables,
        'style': {
            'font_family': 'serif (Times New Roman)',
            'font_size': 10,
            'dpi': 300,
            'format': 'PNG + PDF',
            'colorblind_safe': True,
        },
    }

    config_path = os.path.join(output_dir, 'visualization_config.json')
    with open(config_path, 'w') as f:
        json.dump(viz_config, f, indent=2)

    print(f"\nConfiguracion guardada en: {config_path}")
    print("\nNota: Las figuras se generan con datos reales de los scripts anteriores.")
    print("Ejecutar: python 12_visualization.py --generate-all")

    return viz_config


if __name__ == '__main__':
    config = main()
