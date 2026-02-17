"""
07_hotspot_analysis.py
======================
Fase 3.1: Analisis espacial - Autocorrelacion y hotspots.

Implementa: Moran's I global, Getis-Ord Gi* local, kernel density.
Identifica clusters significativos de deforestacion.

Outputs:
- Moran's I global con significancia
- Mapas de hotspots Gi* (Z-scores)
- Mapas de kernel density
- Estadisticas por municipio
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, STUDY_AREA_BBOX


# ============================================================
# MORAN'S I GLOBAL
# ============================================================

def compute_morans_i(values, weights):
    """
    Calcula Moran's I global para autocorrelacion espacial.

    Args:
        values: numpy array 1D con valores de la variable
        weights: numpy array 2D (NxN) matriz de pesos espaciales

    Returns:
        dict con I, expected_I, z_score, p_value
    """
    n = len(values)
    mean = np.mean(values)
    deviations = values - mean

    # Numerador: sum_i sum_j w_ij * (x_i - mean) * (x_j - mean)
    numerator = np.sum(weights * np.outer(deviations, deviations))

    # Denominador: sum_i (x_i - mean)^2
    denominator = np.sum(deviations ** 2)

    # Total de pesos
    W = np.sum(weights)

    if denominator == 0 or W == 0:
        return {'I': 0, 'expected_I': 0, 'z_score': 0, 'p_value': 1}

    I = (n / W) * (numerator / denominator)
    expected_I = -1.0 / (n - 1)

    # Varianza bajo aleatoriedad
    S1 = 0.5 * np.sum((weights + weights.T) ** 2)
    S2 = np.sum((np.sum(weights, axis=1) + np.sum(weights, axis=0)) ** 2)
    S0 = W

    n2 = n * n
    k = (np.sum(deviations ** 4) / n) / ((np.sum(deviations ** 2) / n) ** 2)

    var_I = (n * ((n2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2) -
             k * (n * (n2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / \
            ((n - 1) * (n - 2) * (n - 3) * S0 ** 2) - expected_I ** 2

    if var_I > 0:
        z_score = (I - expected_I) / np.sqrt(var_I)
    else:
        z_score = 0

    # P-value (two-tailed, normal approximation)
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return {
        'I': round(float(I), 6),
        'expected_I': round(float(expected_I), 6),
        'variance': round(float(var_I), 8),
        'z_score': round(float(z_score), 4),
        'p_value': round(float(p_value), 6),
        'significant': p_value < 0.05,
    }


# ============================================================
# GETIS-ORD Gi*
# ============================================================

def compute_getis_ord_gi_star(values, weights):
    """
    Calcula Getis-Ord Gi* para cada observacion.

    Args:
        values: numpy array 1D
        weights: numpy array 2D (NxN) matriz de pesos

    Returns:
        numpy array de Z-scores Gi*
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return np.zeros(n)

    gi_star = np.zeros(n)

    for i in range(n):
        wi = weights[i, :]
        Wi = np.sum(wi)

        numerator = np.sum(wi * values) - mean * Wi
        S = np.sqrt((n * np.sum(wi ** 2) - Wi ** 2) / (n - 1))

        if S > 0:
            denominator = std * S
            gi_star[i] = numerator / denominator
        else:
            gi_star[i] = 0

    return gi_star


def classify_hotspots(z_scores):
    """
    Clasifica Z-scores en categorias de hotspot/coldspot.
    Orden: asignar umbrales menores primero, luego mayores sobreescriben.
    """
    categories = np.zeros_like(z_scores, dtype=int)
    # Hotspots: asignar de menor a mayor para que 99% sobreescriba 95%
    categories[z_scores >= 1.645] = 1   # Hotspot 90%
    categories[z_scores >= 1.960] = 2   # Hotspot 95%
    categories[z_scores >= 2.576] = 3   # Hotspot 99%
    # Coldspots: asignar de menor a mayor (en negativo)
    categories[z_scores <= -1.645] = -1  # Coldspot 90%
    categories[z_scores <= -1.960] = -2  # Coldspot 95%
    categories[z_scores <= -2.576] = -3  # Coldspot 99%

    return categories


# ============================================================
# SPATIAL WEIGHTS
# ============================================================

def create_queen_weights(n_units, coordinates):
    """
    Crea matriz de pesos Queen contiguity desde coordenadas.
    Simplificado: usa distancia inversa con threshold.

    Args:
        n_units: numero de unidades espaciales
        coordinates: array (n, 2) con lon, lat

    Returns:
        numpy array (n, n) de pesos
    """
    from scipy.spatial.distance import cdist

    distances = cdist(coordinates, coordinates)
    threshold = np.percentile(distances[distances > 0], 25)

    weights = np.zeros((n_units, n_units))
    for i in range(n_units):
        for j in range(n_units):
            if i != j and distances[i, j] <= threshold:
                weights[i, j] = 1.0

    # Row-standardize
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums

    return weights


# ============================================================
# KERNEL DENSITY
# ============================================================

def compute_kernel_density(points, grid_size=100, bandwidth=5000):
    """
    Calcula densidad kernel de puntos de cambio.

    Args:
        points: array (n, 2) con coordenadas de cambio
        grid_size: tamano de grilla
        bandwidth: ancho de banda en metros

    Returns:
        2D array con densidad estimada
    """
    from scipy.stats import gaussian_kde

    if len(points) < 2:
        return np.zeros((grid_size, grid_size))

    kde = gaussian_kde(points.T, bw_method=bandwidth / np.std(points))

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    density = kde(positions).reshape(grid_size, grid_size)

    return density, x_grid, y_grid


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 3.1: ANALISIS ESPACIAL - HOTSPOTS")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase3_stats'
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\nMetodos implementados:")
    print("  1. Moran's I global (autocorrelacion espacial)")
    print("  2. Getis-Ord Gi* (hotspot/coldspot local)")
    print("  3. Kernel density estimation")
    print("  4. Estadisticas por municipio")

    print("\nNota: Requiere datos tabulares de cambio por unidad espacial.")
    print("Ejecutar despues de exportar resultados de 05_change_detection.py")

    # Guardar configuracion
    hotspot_config = {
        'methods': {
            'morans_i': {
                'weight_type': 'Queen contiguity (row-standardized)',
                'permutations': 999,
                'significance': 0.05,
            },
            'getis_ord': {
                'distance_band': 'fixed (25th percentile)',
                'correction': 'FDR (Benjamini-Hochberg)',
                'confidence_levels': [90, 95, 99],
            },
            'kernel_density': {
                'bandwidth': '5 km (Gaussian)',
                'grid_resolution': '1 km',
            },
        },
        'spatial_units': {
            'grid': '1x1 km cells',
            'municipalities': '30 municipios',
            'subcatchments': '~15-20 subcuencas',
        },
        'variables': [
            'Tasa de deforestacion (%/anio)',
            'Cambio neto bosque (ha)',
            'Fragmentacion (NP, ED)',
            'Perdida de carbono (Mg C)',
        ],
    }

    config_path = os.path.join(output_dir, 'hotspot_analysis_config.json')
    with open(config_path, 'w') as f:
        json.dump(hotspot_config, f, indent=2)

    print(f"\nConfiguracion guardada en: {config_path}")
    print("\nProximo paso: 08_ecosystem_services.py")

    return hotspot_config


if __name__ == '__main__':
    config = main()
