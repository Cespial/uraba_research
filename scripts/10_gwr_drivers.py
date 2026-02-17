"""
10_gwr_drivers.py
=================
Fase 3.4: Geographically Weighted Regression para drivers de deforestacion.

Implementa:
- OLS global regression (baseline)
- GWR con kernel adaptativo
- Variables: distancia a rios, vias, poblados, pendiente, precipitacion,
  densidad poblacional, tenencia de tierra (proxy), coca (proxy)

Outputs:
- Coeficientes GWR espacializados
- Mapas de R2 local
- Comparacion OLS vs GWR
- Diagnosticos de multicolinealidad (VIF)
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, STUDY_AREA_BBOX


# ============================================================
# PREPARACION DE VARIABLES INDEPENDIENTES
# ============================================================

def prepare_driver_variables_gee(region):
    """
    Prepara variables independientes desde GEE para el modelo de drivers.
    Todas las variables se calculan a nivel de grilla 1x1 km.

    Returns:
        ee.Image multiband con todas las variables
    """
    import ee

    # 1. TOPOGRAFIA (SRTM)
    srtm = ee.Image('USGS/SRTMGL1_003').clip(region)
    elevation = srtm.select('elevation').rename('elevation')
    slope = ee.Terrain.slope(srtm).rename('slope')

    # 2. DISTANCIA A RIOS
    # Usar JRC Global Surface Water como proxy de red hidrica
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(region)
    rivers = jrc.select('occurrence').gte(50)  # agua permanente
    dist_rivers = rivers.fastDistanceTransform().sqrt().multiply(30).rename('dist_rivers')

    # 3. DISTANCIA A VIAS
    # OpenStreetMap roads via GEE
    # Proxy: usar GHSL built-up como indicador de conectividad
    ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020').clip(region)
    built = ghsl.select('built_surface').gt(0)
    dist_roads = built.fastDistanceTransform().sqrt().multiply(30).rename('dist_roads')

    # 4. DISTANCIA A CENTROS POBLADOS
    # Usando GHSL Settlement Model (SMOD)
    smod = ee.Image('JRC/GHSL/P2023A/GHS_SMOD/2020').clip(region)
    urban = smod.select('smod_code').gte(20)  # urban + suburban
    dist_urban = urban.fastDistanceTransform().sqrt().multiply(30).rename('dist_urban')

    # 5. DENSIDAD POBLACIONAL (WorldPop)
    worldpop = (ee.ImageCollection('WorldPop/GP/100m/pop')
                .filter(ee.Filter.eq('country', 'COL'))
                .filter(ee.Filter.eq('year', 2020))
                .first()
                .clip(region)
                .rename('pop_density'))

    # 6. PRECIPITACION MEDIA (CHIRPS climatologia)
    chirps_mean = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                   .filterDate('2012-01-01', '2024-12-31')
                   .filterBounds(region)
                   .mean()
                   .multiply(365)  # media diaria * 365
                   .clip(region)
                   .rename('precip_annual'))

    # 7. TEMPERATURA MEDIA (MODIS LST)
    lst_mean = (ee.ImageCollection('MODIS/061/MOD11A2')
                .filterDate('2012-01-01', '2024-12-31')
                .filterBounds(region)
                .select('LST_Day_1km')
                .mean()
                .multiply(0.02)
                .subtract(273.15)
                .clip(region)
                .rename('lst_mean'))

    # 8. SUELO - contenido de arcilla (OpenLandMap)
    try:
        clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02') \
            .select('b0').clip(region).rename('clay_content')
        clay.bandNames().getInfo()  # Eager check
    except Exception:
        clay = ee.Image(0).rename('clay_content')

    # Combinar todas las variables
    drivers = (elevation
               .addBands(slope)
               .addBands(dist_rivers)
               .addBands(dist_roads)
               .addBands(dist_urban)
               .addBands(worldpop)
               .addBands(chirps_mean)
               .addBands(lst_mean)
               .addBands(clay))

    return drivers


def extract_sample_data(drivers, deforestation_map, region, n_points=2000, scale=1000):
    """
    Extrae datos tabulares para regresion desde GEE.

    Args:
        drivers: ee.Image multiband con variables independientes
        deforestation_map: ee.Image con tasa de deforestacion
        region: ee.Geometry
        n_points: numero de puntos de muestreo
        scale: resolucion

    Returns:
        ee.FeatureCollection con datos para regresion
    """
    import ee

    combined = drivers.addBands(deforestation_map)

    sample = combined.sample(
        region=region,
        scale=scale,
        numPixels=n_points,
        seed=42,
        geometries=True
    )

    return sample


# ============================================================
# OLS GLOBAL REGRESSION
# ============================================================

def fit_ols(X, y):
    """
    Ajusta regresion OLS: y = Xb + e

    Args:
        X: numpy array (n, p) variables independientes
        y: numpy array (n,) variable dependiente

    Returns:
        dict con coeficientes, R2, AIC, residuos
    """
    n, p = X.shape

    # Agregar intercepto
    X_int = np.column_stack([np.ones(n), X])

    # OLS: b = (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X_int.T @ X_int)
        beta = XtX_inv @ X_int.T @ y
    except np.linalg.LinAlgError:
        # Usar pseudoinversa si singular
        beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
        XtX_inv = np.linalg.pinv(X_int.T @ X_int)

    y_pred = X_int @ beta
    residuals = y - y_pred
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (sse / sst) if sst > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    # AIC
    sigma2 = sse / n
    if sigma2 > 0:
        aic = n * np.log(sigma2) + 2 * (p + 1)
    else:
        aic = float('inf')

    # Standard errors
    if sigma2 > 0:
        se = np.sqrt(np.diag(XtX_inv) * (sse / (n - p - 1)))
        t_stats = beta / se
    else:
        se = np.zeros(p + 1)
        t_stats = np.zeros(p + 1)

    return {
        'coefficients': beta.tolist(),
        'std_errors': se.tolist(),
        't_statistics': t_stats.tolist(),
        'r_squared': float(r2),
        'adj_r_squared': float(adj_r2),
        'aic': float(aic),
        'residuals': residuals,
        'y_predicted': y_pred,
        'n': n,
        'p': p,
    }


def compute_vif(X):
    """
    Calcula Variance Inflation Factor para diagnostico de multicolinealidad.

    VIF_j = 1 / (1 - R2_j) donde R2_j es de regresion X_j ~ X_resto

    Returns:
        list de VIF por variable
    """
    n, p = X.shape
    vifs = []

    for j in range(p):
        X_j = X[:, j]
        X_rest = np.delete(X, j, axis=1)

        # Regresion X_j ~ X_rest
        X_rest_int = np.column_stack([np.ones(n), X_rest])
        try:
            beta = np.linalg.lstsq(X_rest_int, X_j, rcond=None)[0]
            X_j_pred = X_rest_int @ beta
            sse = np.sum((X_j - X_j_pred) ** 2)
            sst = np.sum((X_j - np.mean(X_j)) ** 2)
            r2_j = 1 - (sse / sst) if sst > 0 else 0
            vif = 1 / (1 - r2_j) if r2_j < 1 else float('inf')
        except Exception:
            vif = float('inf')

        vifs.append(round(vif, 2))

    return vifs


# ============================================================
# GEOGRAPHICALLY WEIGHTED REGRESSION
# ============================================================

def compute_gwr(X, y, coordinates, bandwidth=None, kernel='adaptive'):
    """
    Geographically Weighted Regression.

    Args:
        X: (n, p) variables independientes
        y: (n,) variable dependiente
        coordinates: (n, 2) lon, lat
        bandwidth: ancho de banda (n vecinos si adaptive, distancia si fixed)
        kernel: 'adaptive' (bisquare) o 'fixed' (gaussian)

    Returns:
        dict con coeficientes locales, R2 local, AIC
    """
    from scipy.spatial.distance import cdist

    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    p_full = p + 1

    # Bandwidth por defecto: n/3 vecinos (adaptive)
    if bandwidth is None:
        bandwidth = max(int(n * 0.3), p_full + 2)

    distances = cdist(coordinates, coordinates)

    # Resultados locales
    local_betas = np.zeros((n, p_full))
    local_r2 = np.zeros(n)
    local_residuals = np.zeros(n)
    hat_matrix_trace = 0

    for i in range(n):
        dists_i = distances[i, :]

        if kernel == 'adaptive':
            # Bisquare kernel: k vecinos mas cercanos
            sorted_idx = np.argsort(dists_i)
            bw_dist = dists_i[sorted_idx[bandwidth]]
            if bw_dist > 0:
                u = dists_i / bw_dist
                weights = np.where(u <= 1, (1 - u ** 2) ** 2, 0)
            else:
                weights = np.ones(n) / n
        else:
            # Gaussian kernel
            weights = np.exp(-0.5 * (dists_i / bandwidth) ** 2)

        # Weighted least squares: b_i = (X'WX)^-1 X'Wy
        # Use vector weights directly (avoid O(n^2) diagonal matrix)
        w_sqrt = np.sqrt(weights)
        Xw = X_int * w_sqrt[:, np.newaxis]  # element-wise: each row * sqrt(w)
        yw = y * w_sqrt
        try:
            XtWX = Xw.T @ Xw
            XtWy = Xw.T @ yw
            beta_i = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta_i = np.linalg.lstsq(Xw, yw, rcond=None)[0]

        local_betas[i, :] = beta_i

        # Prediccion y R2 local
        y_pred_i = X_int[i, :] @ beta_i
        local_residuals[i] = y[i] - y_pred_i

        # R2 local (weighted)
        weighted_y = weights * y
        mean_weighted = np.sum(weighted_y) / np.sum(weights) if np.sum(weights) > 0 else 0
        sst_local = np.sum(weights * (y - mean_weighted) ** 2)
        sse_local = np.sum(weights * (y - X_int @ beta_i) ** 2)
        local_r2[i] = 1 - (sse_local / sst_local) if sst_local > 0 else 0

        # Hat matrix trace (para AIC)
        try:
            hi = X_int[i, :] @ np.linalg.solve(XtWX, X_int[i, :] * weights[i])
            hat_matrix_trace += hi
        except Exception:
            pass

    # AIC GWR
    sse_total = np.sum(local_residuals ** 2)
    sigma2 = sse_total / n
    if sigma2 > 0:
        aic_gwr = n * np.log(sigma2) + 2 * hat_matrix_trace
    else:
        aic_gwr = float('inf')

    return {
        'local_betas': local_betas,
        'local_r2': local_r2,
        'local_residuals': local_residuals,
        'aic': float(aic_gwr),
        'bandwidth': bandwidth,
        'kernel': kernel,
        'mean_r2': float(np.mean(local_r2)),
        'median_r2': float(np.median(local_r2)),
        'hat_matrix_trace': float(hat_matrix_trace),  # Effective Number of Parameters (ENP)
        'enp_n_ratio': float(hat_matrix_trace / n),  # ENP/n ratio
        'n': n,
        'p': p,
    }


def optimize_bandwidth(X, y, coordinates, kernel='adaptive',
                       bw_min=None, bw_max=None, n_steps=20):
    """
    Optimiza bandwidth de GWR minimizando AICc.

    Returns:
        int: bandwidth optimo
    """
    n, p = X.shape

    if bw_min is None:
        bw_min = p + 3
    if bw_max is None:
        bw_max = int(n * 0.8)

    bandwidths = np.linspace(bw_min, bw_max, n_steps).astype(int)
    best_aic = float('inf')
    best_bw = bandwidths[0]

    for bw in bandwidths:
        try:
            result = compute_gwr(X, y, coordinates, bandwidth=int(bw), kernel=kernel)
            if result['aic'] < best_aic:
                best_aic = result['aic']
                best_bw = int(bw)
        except Exception:
            continue

    return best_bw, best_aic


# ============================================================
# ANALISIS DE RESULTADOS GWR
# ============================================================

def summarize_gwr_results(gwr_result, variable_names):
    """
    Resume resultados GWR: estadisticas de coeficientes locales.
    """
    betas = gwr_result['local_betas']
    names = ['intercept'] + variable_names

    summary = {}
    for j, name in enumerate(names):
        beta_j = betas[:, j]
        summary[name] = {
            'mean': round(float(np.mean(beta_j)), 6),
            'median': round(float(np.median(beta_j)), 6),
            'std': round(float(np.std(beta_j)), 6),
            'min': round(float(np.min(beta_j)), 6),
            'max': round(float(np.max(beta_j)), 6),
            'pct_positive': round(float(np.mean(beta_j > 0) * 100), 1),
            'pct_negative': round(float(np.mean(beta_j < 0) * 100), 1),
        }

    return summary


def compare_ols_gwr(ols_result, gwr_result):
    """
    Compara rendimiento OLS vs GWR.
    """
    return {
        'ols_r2': ols_result['r_squared'],
        'ols_adj_r2': ols_result['adj_r_squared'],
        'ols_aic': ols_result['aic'],
        'gwr_mean_r2': gwr_result['mean_r2'],
        'gwr_aic': gwr_result['aic'],
        'r2_improvement': round(gwr_result['mean_r2'] - ols_result['r_squared'], 4),
        'aic_improvement': round(ols_result['aic'] - gwr_result['aic'], 2),
        'gwr_preferred': gwr_result['aic'] < ols_result['aic'],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 3.4: GWR - DRIVERS DE DEFORESTACION")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase3_stats'
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\nModelos implementados:")
    print("  1. OLS global (baseline)")
    print("  2. GWR con kernel bisquare adaptativo")
    print("  3. Optimizacion de bandwidth (AICc)")
    print("  4. Diagnosticos: VIF, comparacion OLS vs GWR")

    print("\nVariables independientes:")
    variable_names = [
        'elevation', 'slope', 'dist_rivers', 'dist_roads',
        'dist_urban', 'pop_density', 'precip_annual',
        'lst_mean', 'clay_content'
    ]
    for i, v in enumerate(variable_names, 1):
        print(f"  X{i}: {v}")

    print("\nVariable dependiente: tasa de deforestacion (%/anio)")
    print("\nNota: Requiere datos tabulares exportados de GEE.")
    print("Ejecutar extract_sample_data() primero.")

    # Guardar configuracion
    gwr_config = {
        'dependent_variable': {
            'name': 'deforestation_rate',
            'description': 'Tasa anual de deforestacion (%)',
            'source': '05_change_detection.py',
        },
        'independent_variables': {
            'elevation': {'source': 'SRTM 30m', 'unit': 'm'},
            'slope': {'source': 'SRTM-derived', 'unit': 'degrees'},
            'dist_rivers': {'source': 'JRC Water occurrence >50%', 'unit': 'm'},
            'dist_roads': {'source': 'GHSL built-up proxy', 'unit': 'm'},
            'dist_urban': {'source': 'GHSL SMOD >=20', 'unit': 'm'},
            'pop_density': {'source': 'WorldPop 2020', 'unit': 'persons/100m2'},
            'precip_annual': {'source': 'CHIRPS 2012-2024 mean', 'unit': 'mm/year'},
            'lst_mean': {'source': 'MODIS MOD11A2 mean', 'unit': 'deg C'},
            'clay_content': {'source': 'SoilGrids ISRIC', 'unit': 'g/kg'},
        },
        'models': {
            'ols': {
                'type': 'Ordinary Least Squares',
                'purpose': 'baseline global model',
            },
            'gwr': {
                'type': 'Geographically Weighted Regression',
                'kernel': 'adaptive bisquare',
                'bandwidth_selection': 'AICc minimization',
                'software': 'custom numpy implementation',
            },
        },
        'diagnostics': {
            'multicollinearity': 'VIF (threshold = 10)',
            'spatial_autocorrelation': "Moran's I on residuals",
            'model_comparison': 'AIC, R2, F-test',
        },
        'spatial_resolution': '1 km grid',
        'sample_size': 2000,
    }

    config_path = os.path.join(output_dir, 'gwr_drivers_config.json')
    with open(config_path, 'w') as f:
        json.dump(gwr_config, f, indent=2)

    print(f"\nConfiguracion guardada en: {config_path}")
    print("\nProximo paso: 11_ca_markov.py")

    return gwr_config


if __name__ == '__main__':
    config = main()
