"""
09_climate_analysis.py
======================
Fase 3.3: Analisis climatico y tendencias.

Implementa:
- Series temporales CHIRPS (precipitacion 2012-2024)
- MODIS LST (temperatura superficial)
- Indices de sequia (SPI, SPEI proxy)
- Mann-Kendall trends
- Correlacion clima-deforestacion

Outputs:
- Tendencias de precipitacion y temperatura
- Mapas de anomalias climaticas
- Correlaciones clima-LULCC
"""

import ee
import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, STUDY_AREA_BBOX
from scripts.utils import get_study_area


# ============================================================
# PRECIPITACION (CHIRPS)
# ============================================================

def get_annual_precipitation(region, start_year=2012, end_year=2024):
    """
    Obtiene precipitacion anual de CHIRPS para cada anio.

    Returns:
        ee.ImageCollection con precipitacion anual (mm)
    """
    def annual_precip(year):
        year = ee.Number(year)
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        annual = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .filterBounds(region)
                  .sum()
                  .clip(region)
                  .rename('precipitation')
                  .set('year', year)
                  .set('system:time_start', start.millis()))
        return annual

    years = ee.List.sequence(start_year, end_year)
    return ee.ImageCollection(years.map(annual_precip))


def get_monthly_precipitation(region, year):
    """
    Obtiene precipitacion mensual para un anio dado.

    Returns:
        ee.ImageCollection con 12 imagenes mensuales
    """
    def monthly_precip(month):
        month = ee.Number(month)
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        monthly = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                   .filterDate(start, end)
                   .filterBounds(region)
                   .sum()
                   .clip(region)
                   .rename('precipitation')
                   .set('month', month)
                   .set('system:time_start', start.millis()))
        return monthly

    months = ee.List.sequence(1, 12)
    return ee.ImageCollection(months.map(monthly_precip))


def compute_precipitation_anomalies(region, start_year=2012, end_year=2024):
    """
    Calcula anomalias de precipitacion respecto al promedio climatologico.
    Anomalia = (P_anual - P_media) / P_std
    """
    annual_col = get_annual_precipitation(region, start_year, end_year)

    # Climatologia (media y std)
    mean_precip = annual_col.mean().rename('precip_mean')
    std_precip = annual_col.reduce(ee.Reducer.stdDev()).rename('precip_std')

    def calc_anomaly(image):
        anomaly = image.subtract(mean_precip).divide(std_precip).rename('precip_anomaly')
        return anomaly.copyProperties(image, ['year', 'system:time_start'])

    anomalies = annual_col.map(calc_anomaly)

    return anomalies, mean_precip, std_precip


# ============================================================
# TEMPERATURA SUPERFICIAL (MODIS LST)
# ============================================================

def get_annual_lst(region, start_year=2012, end_year=2024):
    """
    Obtiene temperatura superficial anual de MODIS MOD11A2.

    Returns:
        ee.ImageCollection con LST media anual (grados C)
    """
    def annual_temp(year):
        year = ee.Number(year)
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        lst = (ee.ImageCollection('MODIS/061/MOD11A2')
               .filterDate(start, end)
               .filterBounds(region)
               .select('LST_Day_1km')
               .mean()
               .multiply(0.02)
               .subtract(273.15)  # Kelvin a Celsius
               .clip(region)
               .rename('LST')
               .set('year', year)
               .set('system:time_start', start.millis()))
        return lst

    years = ee.List.sequence(start_year, end_year)
    return ee.ImageCollection(years.map(annual_temp))


def compute_lst_anomalies(region, start_year=2012, end_year=2024):
    """
    Calcula anomalias de temperatura superficial.
    """
    annual_col = get_annual_lst(region, start_year, end_year)
    mean_lst = annual_col.mean().rename('lst_mean')
    std_lst = annual_col.reduce(ee.Reducer.stdDev()).rename('lst_std')

    def calc_anomaly(image):
        anomaly = image.subtract(mean_lst).divide(std_lst).rename('lst_anomaly')
        return anomaly.copyProperties(image, ['year', 'system:time_start'])

    anomalies = annual_col.map(calc_anomaly)
    return anomalies, mean_lst, std_lst


# ============================================================
# INDICES DE SEQUIA
# ============================================================

def compute_spi(region, year, reference_start=2000, reference_end=2020):
    """
    Standardized Precipitation Index (SPI) simplificado.
    SPI = (P - P_media) / P_std

    Usa periodo de referencia largo para climatologia.
    """
    # Precipitacion del anio de interes
    p_year = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterDate(f'{year}-01-01', f'{year}-12-31')
              .filterBounds(region)
              .sum()
              .clip(region))

    # Climatologia de referencia
    def annual_p(y):
        y = ee.Number(y)
        start = ee.Date.fromYMD(y, 1, 1)
        end = ee.Date.fromYMD(y, 12, 31)
        return (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                .filterDate(start, end)
                .filterBounds(region)
                .sum()
                .clip(region)
                .set('system:time_start', start.millis()))

    ref_years = ee.List.sequence(reference_start, reference_end)
    ref_col = ee.ImageCollection(ref_years.map(annual_p))
    ref_mean = ref_col.mean()
    ref_std = ref_col.reduce(ee.Reducer.stdDev())

    spi = p_year.subtract(ref_mean).divide(ref_std).rename('SPI')
    return spi


def compute_drought_frequency(region, start_year=2012, end_year=2024, threshold=-1,
                              reference_start=2000, reference_end=2020):
    """
    Calcula frecuencia de condiciones de sequia (SPI < threshold).
    Usa loop Python en lugar de ee.List.map() para evitar problemas
    con compute_spi que necesita year como int Python.
    """
    total_years = end_year - start_year + 1
    drought_images = []

    for year in range(start_year, end_year + 1):
        spi = compute_spi(region, year, reference_start, reference_end)
        drought = spi.lt(threshold).rename('drought')
        drought_images.append(drought)

    drought_col = ee.ImageCollection(drought_images)
    frequency = drought_col.sum().divide(total_years).rename('drought_frequency')

    return frequency


# ============================================================
# MANN-KENDALL TREND TEST
# ============================================================

def compute_mann_kendall_trend(image_collection, region, scale=1000):
    """
    Calcula tendencia Mann-Kendall pixel-wise usando ee.Reducer.kendallsCorrelation.

    Args:
        image_collection: ee.ImageCollection con serie temporal
        region: ee.Geometry
        scale: resolucion para calculos

    Returns:
        ee.Image con tau y p-value
    """
    # Agregar banda de tiempo (anos como fraccion)
    def add_time(image):
        date = ee.Date(image.get('system:time_start'))
        year_frac = date.get('year').add(date.get('month').subtract(1).divide(12))
        time_band = ee.Image(year_frac).float().rename('time')
        return image.addBands(time_band)

    col_with_time = image_collection.map(add_time)

    # Kendall's correlation: tau entre tiempo y variable
    trend = col_with_time.select(['time', col_with_time.first().bandNames().get(0)]).reduce(
        ee.Reducer.kendallsCorrelation()
    )

    return trend


def compute_sen_slope(values, times):
    """
    Calcula pendiente de Sen (median of pairwise slopes).
    Para uso local con numpy.

    Args:
        values: array de valores
        times: array de tiempos

    Returns:
        float: pendiente de Sen
    """
    n = len(values)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if times[j] != times[i]:
                slopes.append((values[j] - values[i]) / (times[j] - times[i]))

    if slopes:
        return np.median(slopes)
    return 0.0


# ============================================================
# CORRELACION CLIMA-DEFORESTACION
# ============================================================

def correlate_climate_lulcc(precip_collection, lst_collection,
                            forest_loss_collection, region, scale=1000):
    """
    Correlaciona variables climaticas con perdida de bosque.

    Usa Pearson correlation pixel-wise.
    """
    # Correlacion precipitacion - forest loss
    precip_loss = precip_collection.select('precipitation') \
        .combine(forest_loss_collection)

    corr_precip = precip_loss.reduce(ee.Reducer.pearsonsCorrelation())

    # Correlacion LST - forest loss
    lst_loss = lst_collection.select('LST') \
        .combine(forest_loss_collection)

    corr_lst = lst_loss.reduce(ee.Reducer.pearsonsCorrelation())

    return {
        'precip_loss_correlation': corr_precip,
        'lst_loss_correlation': corr_lst,
    }


def compute_climate_by_lulc_class(lulc_map, precip, lst, region):
    """
    Calcula estadisticas climaticas por clase LULC.
    Util para entender como el cambio de uso afecta el microclima.
    """
    stats_by_class = {}
    for class_id, class_info in {
        1: 'Bosque denso', 2: 'Bosque secundario', 3: 'Pasturas',
        4: 'Cultivos', 5: 'Agua', 6: 'Urbano', 7: 'Suelo desnudo'
    }.items():
        class_mask = lulc_map.eq(class_id)
        masked_lst = lst.updateMask(class_mask)

        lst_stats = masked_lst.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=region,
            scale=1000,
            maxPixels=1e13
        )

        stats_by_class[class_id] = {
            'name': class_info,
            'lst_stats': lst_stats,
        }

    return stats_by_class


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 3.3: ANALISIS CLIMATICO Y TENDENCIAS")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    region = get_study_area()
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase3_stats'
    )
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # PRECIPITACION
    # ============================================================
    print("\nSECCION 1: PRECIPITACION (CHIRPS)")
    print("-" * 40)

    print("  Calculando precipitacion anual 2012-2024...")
    annual_precip = get_annual_precipitation(region, 2012, 2024)
    print(f"  Coleccion: {annual_precip.size().getInfo()} imagenes anuales")

    print("  Calculando anomalias...")
    anomalies, mean_p, std_p = compute_precipitation_anomalies(region, 2012, 2024)

    # Tendencia Mann-Kendall
    print("  Calculando tendencia Mann-Kendall...")
    precip_trend = compute_mann_kendall_trend(annual_precip, region)
    print("  Tendencia calculada.")

    # ============================================================
    # TEMPERATURA SUPERFICIAL
    # ============================================================
    print(f"\n{'=' * 50}")
    print("SECCION 2: TEMPERATURA SUPERFICIAL (MODIS LST)")
    print("=" * 50)

    print("  Calculando LST anual 2012-2024...")
    annual_lst = get_annual_lst(region, 2012, 2024)

    print("  Calculando anomalias de LST...")
    lst_anomalies, mean_lst, std_lst = compute_lst_anomalies(region, 2012, 2024)

    print("  Calculando tendencia Mann-Kendall LST...")
    lst_trend = compute_mann_kendall_trend(annual_lst, region)

    # ============================================================
    # INDICES DE SEQUIA
    # ============================================================
    print(f"\n{'=' * 50}")
    print("SECCION 3: INDICES DE SEQUIA")
    print("=" * 50)

    for year in [2013, 2016, 2020, 2024]:
        print(f"  SPI {year}: calculando...")
        spi = compute_spi(region, year)

    print("  Frecuencia de sequia 2012-2024: calculando...")
    drought_freq = compute_drought_frequency(region, 2012, 2024, threshold=-1)

    # ============================================================
    # GUARDAR CONFIG
    # ============================================================
    climate_config = {
        'precipitation': {
            'source': 'CHIRPS Daily v2.0',
            'resolution': '0.05 deg (~5.5 km)',
            'period': '2012-2024',
            'variables': ['annual total', 'monthly', 'anomalies', 'trend'],
        },
        'temperature': {
            'source': 'MODIS MOD11A2 v061',
            'resolution': '1 km',
            'period': '2012-2024',
            'variables': ['annual mean LST day', 'anomalies', 'trend'],
        },
        'drought': {
            'index': 'SPI (Standardized Precipitation Index)',
            'reference_period': '2000-2020',
            'threshold_drought': -1.0,
            'threshold_severe': -1.5,
        },
        'trend_test': {
            'method': 'Mann-Kendall (kendallsCorrelation)',
            'slope': 'Sen slope',
            'significance': 0.05,
        },
        'correlation': {
            'method': 'Pearson pixel-wise',
            'variables': ['precipitation vs forest loss', 'LST vs forest loss'],
        },
    }

    config_path = os.path.join(output_dir, 'climate_analysis_config.json')
    with open(config_path, 'w') as f:
        json.dump(climate_config, f, indent=2)

    print(f"\nConfiguracion guardada en: {config_path}")
    print("\nProximo paso: 10_gwr_drivers.py")

    return climate_config


if __name__ == '__main__':
    config = main()
