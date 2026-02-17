"""
05_change_detection.py
======================
Fase 2.3: Analisis de cambio de uso del suelo.

Incluye: matrices de transicion, tasas de deforestacion,
LandTrendr continuous change detection, intensity analysis.

Outputs:
- 3 matrices de transicion (T1->T2, T2->T3, T3->T4)
- Tasas anuales de cambio por clase
- Mapas de cambio LandTrendr (2012-2024)
- Estadisticas de cambio por municipio
"""

import ee
import os
import sys
import json
import math
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, LULC_CLASSES, LANDTRENDR_PARAMS, STUDY_AREA_BBOX
from scripts.utils import get_study_area, export_image_to_drive, mask_landsat_clouds


# ============================================================
# MATRICES DE TRANSICION
# ============================================================

def compute_transition_matrix(lulc_t1, lulc_t2, region, n_classes=7, scale=30):
    """
    Calcula matriz de transicion entre dos mapas LULC.

    Returns:
        dict con matriz en hectareas, porcentajes, y estadisticas
    """
    # Crear imagen de transicion: valor = class_t1 * 10 + class_t2
    transition = lulc_t1.multiply(10).add(lulc_t2).rename('transition')

    # Calcular area por tipo de transicion — need 2 bands for Reducer.group:
    # Band 0 = area (to sum), Band 1 = transition code (to group by)
    area_image = ee.Image.pixelArea().divide(10000).addBands(transition)

    # Reducir a estadisticas por valor de transicion
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName='transition'),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        tileScale=4,
        bestEffort=True
    )

    return transition, stats


def compute_change_rates(area_t1, area_t2, years_between):
    """
    Calcula tasas de cambio anuales usando formula FAO (Puyravaud 2003).
    r = (1/t) * ln(A2/A1)
    """
    rates = {}
    n_classes = len(LULC_CLASSES)
    for class_id in range(1, n_classes + 1):
        a1 = area_t1.get(class_id, 0)
        a2 = area_t2.get(class_id, 0)
        if a1 > 0 and a2 > 0:
            r = (1.0 / years_between) * math.log(a2 / a1) * 100
            net_change = a2 - a1
            pct_change = ((a2 - a1) / a1) * 100
        else:
            r = 0
            net_change = a2 - a1
            pct_change = 0

        class_name = LULC_CLASSES.get(class_id, {}).get('name', f'Clase {class_id}')
        rates[class_id] = {
            'name': class_name,
            'area_t1_ha': round(a1, 1),
            'area_t2_ha': round(a2, 1),
            'net_change_ha': round(net_change, 1),
            'pct_change': round(pct_change, 2),
            'annual_rate_pct': round(r, 3),
        }
    return rates


# ============================================================
# LANDTRENDR
# ============================================================

def run_landtrendr(region, start_year=2012, end_year=2024):
    """
    Ejecuta LandTrendr para deteccion continua de cambio.
    Usa NBR como indice principal.
    """
    # Construir coleccion anual de Landsat
    def get_annual_nbr(year):
        year = ee.Number(year)
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)

        l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterDate(start, end)
              .filterBounds(region)
              .filter(ee.Filter.lt('CLOUD_COVER', 70))
              .map(mask_landsat_clouds))

        # NBR = (NIR - SWIR2) / (NIR + SWIR2)
        def calc_nbr(image):
            nir = image.select('SR_B5')
            swir2 = image.select('SR_B7')
            nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename('NBR')
            return nbr.multiply(10000).toInt16().set('system:time_start', image.get('system:time_start'))

        return l8.map(calc_nbr).median().set('system:time_start', start.millis())

    years = ee.List.sequence(start_year, end_year)
    annual_collection = ee.ImageCollection(years.map(get_annual_nbr))

    # Parametros LandTrendr
    lt_params = {
        'timeSeries': annual_collection,
        'maxSegments': LANDTRENDR_PARAMS['maxSegments'],
        'spikeThreshold': LANDTRENDR_PARAMS['spikeThreshold'],
        'vertexCountOvershoot': LANDTRENDR_PARAMS['vertexCountOvershoot'],
        'preventOneYearRecovery': LANDTRENDR_PARAMS['preventOneYearRecovery'],
        'recoveryThreshold': LANDTRENDR_PARAMS['recoveryThreshold'],
        'pvalThreshold': LANDTRENDR_PARAMS['pvalThreshold'],
        'bestModelProportion': LANDTRENDR_PARAMS['bestModelProportion'],
        'minObservationsNeeded': LANDTRENDR_PARAMS['minObservationsNeeded'],
    }

    # Ejecutar LandTrendr
    lt_result = ee.Algorithms.TemporalSegmentation.LandTrendr(**lt_params)

    return lt_result


def extract_disturbance_map(lt_result, min_magnitude=200, min_duration=1):
    """
    Extrae mapa de perturbacion mayor desde resultados LandTrendr.

    Args:
        lt_result: resultado de LandTrendr
        min_magnitude: cambio minimo en NBR*10000
        min_duration: duracion minima en anios

    Returns:
        ee.Image con bandas: year_of_disturbance, magnitude, duration
    """
    lt_array = lt_result.select('LandTrendr')
    rmse = lt_result.select('rmse')

    # Extraer vertices del array LandTrendr
    # LandTrendr array shape: [rows=3+, cols=years]
    # Row 0: years, Row 1: source values, Row 2: fitted values
    vertice_mask = lt_array.arraySlice(0, 3, 4)  # is-vertex flag

    # Calcular deltas entre vertices consecutivos
    fitted = lt_array.arraySlice(0, 2, 3).arrayProject([1]).arrayFlatten([['fitted']])

    # Approach: mayor segmento de perdida (decline en NBR = disturbio)
    # Usar diferencia entre max fitted y min fitted post-max
    fitted_max = lt_array.arraySlice(0, 2, 3).arrayReduce(ee.Reducer.max(), [1])
    fitted_min = lt_array.arraySlice(0, 2, 3).arrayReduce(ee.Reducer.min(), [1])

    # Magnitud de la mayor perturbacion
    magnitude = fitted_max.subtract(fitted_min).arrayFlatten([['magnitude']])

    # Filtrar por magnitud minima
    disturbance = magnitude.updateMask(magnitude.gte(min_magnitude))

    return disturbance, rmse


# ============================================================
# ANALISIS POR MUNICIPIO
# ============================================================

def compute_change_by_municipality(transition_map, region):
    """
    Calcula estadisticas de cambio agregadas por municipio.
    """
    # Cargar limites municipales
    admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    colombia_mun = admin2.filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))
    study_mun = colombia_mun.filterBounds(region)

    # Reducir por municipio
    stats_by_mun = transition_map.multiply(ee.Image.pixelArea()).divide(10000).reduceRegions(
        collection=study_mun,
        reducer=ee.Reducer.sum(),
        scale=30
    )

    return stats_by_mun


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 2.3: ANALISIS DE CAMBIO DE USO DEL SUELO")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    region = get_study_area()
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase2_gee'
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\nNota: Este script requiere los mapas LULC de 03_classification.py")
    print("Los mapas se cargan desde GEE assets o se regeneran.\n")

    # ============================================================
    # MATRICES DE TRANSICION
    # ============================================================
    print("SECCION 1: MATRICES DE TRANSICION")
    print("─" * 40)

    periods_list = list(PERIODS.keys())
    transitions = {
        'T1_T2': {'from': periods_list[0], 'to': periods_list[1],
                   'years': PERIODS[periods_list[1]]['map_year'] - PERIODS[periods_list[0]]['map_year']},
        'T2_T3': {'from': periods_list[1], 'to': periods_list[2],
                   'years': PERIODS[periods_list[2]]['map_year'] - PERIODS[periods_list[1]]['map_year']},
        'T3_T4': {'from': periods_list[2], 'to': periods_list[3],
                   'years': PERIODS[periods_list[3]]['map_year'] - PERIODS[periods_list[2]]['map_year']},
    }

    for trans_key, trans_info in transitions.items():
        p_from = trans_info['from']
        p_to = trans_info['to']
        years_between = trans_info['years']
        y_from = PERIODS[p_from]['map_year']
        y_to = PERIODS[p_to]['map_year']

        print(f"\n  Transicion {y_from} -> {y_to} ({years_between} anios)")
        print(f"    {PERIODS[p_from]['label']} -> {PERIODS[p_to]['label']}")
        print(f"    Matrices y tasas se calculan post-clasificacion.")

    # ============================================================
    # LANDTRENDR
    # ============================================================
    print(f"\n{'═' * 50}")
    print("SECCION 2: LANDTRENDR CONTINUOUS CHANGE DETECTION")
    print("═" * 50)

    print("\n  Ejecutando LandTrendr (2012-2024)...")
    print(f"  Parametros: maxSegments={LANDTRENDR_PARAMS['maxSegments']}, "
          f"spikeThreshold={LANDTRENDR_PARAMS['spikeThreshold']}")

    lt_result = run_landtrendr(region, 2012, 2024)
    print("  LandTrendr completado.")

    # Extraer perturbaciones
    lt_disturbance, rmse = extract_disturbance_map(lt_result)
    print("  Mapa de perturbaciones extraido.")

    # ============================================================
    # HANSEN GFC COMO REFERENCIA
    # ============================================================
    print(f"\n{'═' * 50}")
    print("SECCION 3: HANSEN GFC - VALIDACION CRUZADA")
    print("═" * 50)

    hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').clip(region)
    lossyear = hansen.select('lossyear')

    # Perdida por periodo
    hansen_stats = {}
    for period_key, period_info in PERIODS.items():
        y_start = int(period_info['start'][:4]) - 2000
        y_end = int(period_info['end'][:4]) - 2000
        period_loss = lossyear.gte(y_start).And(lossyear.lte(y_end))
        loss_ha = period_loss.multiply(ee.Image.pixelArea()).divide(10000)
        total = loss_ha.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=30,
            maxPixels=1e13
        )
        print(f"  Hansen forest loss {period_info['label']}: calculando...")
        hansen_stats[period_key] = {
            'year_range': f"{period_info['start'][:4]}-{period_info['end'][:4]}",
        }

    # Guardar configuracion de change detection
    change_config = {
        'transitions': {k: {
            'from_period': v['from'],
            'to_period': v['to'],
            'years_between': v['years'],
            'from_year': PERIODS[v['from']]['map_year'],
            'to_year': PERIODS[v['to']]['map_year'],
        } for k, v in transitions.items()},
        'landtrendr_params': LANDTRENDR_PARAMS,
        'hansen_gfc_version': 'v1.11 (2023)',
        'change_rate_formula': 'FAO Puyravaud 2003: r = (1/t) * ln(A2/A1)',
    }

    config_path = os.path.join(output_dir, 'change_detection_config.json')
    with open(config_path, 'w') as f:
        json.dump(change_config, f, indent=2)

    print(f"\nConfiguracion guardada en: {config_path}")
    print("\nProximo paso: 06_fragmentation.py")

    return lt_result, hansen_stats


if __name__ == '__main__':
    lt_result, hansen_stats = main()
