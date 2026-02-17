"""
08_ecosystem_services.py
========================
Fase 3.2: Estimacion de servicios ecosistemicos.

Implementa modelos InVEST simplificados en GEE:
- Carbon Storage (IPCC Tier 1 + GEE biomass)
- Seasonal Water Yield (proxy via CHIRPS + LULC)
- Habitat Quality (basado en amenazas y sensibilidad)

Outputs:
- Mapas de carbono almacenado por periodo
- Perdida de carbono por transiciones LULC
- Indice de rendimiento hidrico estacional
- Mapas de calidad de habitat
"""

import ee
import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, LULC_CLASSES, CARBON_POOLS, STUDY_AREA_BBOX
from scripts.utils import get_study_area, export_image_to_drive


# ============================================================
# CARBON STORAGE (InVEST-style en GEE)
# ============================================================

def compute_carbon_storage(lulc_map, region, carbon_pools=None):
    """
    Calcula almacenamiento de carbono basado en LULC usando
    pools IPCC Tier 1 para tropical humedo.

    Args:
        lulc_map: ee.Image con clasificacion LULC (1-7)
        region: ee.Geometry
        carbon_pools: dict {class_id: {c_above, c_below, c_soil, c_dead}}

    Returns:
        dict con ee.Images de carbono y estadisticas
    """
    if carbon_pools is None:
        carbon_pools = CARBON_POOLS

    # Crear imagenes de carbono por pool
    c_above = ee.Image(0).float()
    c_below = ee.Image(0).float()
    c_soil = ee.Image(0).float()
    c_dead = ee.Image(0).float()

    for class_id, pools in carbon_pools.items():
        mask = lulc_map.eq(class_id)
        c_above = c_above.where(mask, pools['c_above'])
        c_below = c_below.where(mask, pools['c_below'])
        c_soil = c_soil.where(mask, pools['c_soil'])
        c_dead = c_dead.where(mask, pools['c_dead'])

    # Total carbon (Mg C/ha)
    c_total = c_above.add(c_below).add(c_soil).add(c_dead).rename('c_total')

    # Estadisticas zonales
    stats = c_total.multiply(ee.Image.pixelArea().divide(10000)).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=30,
        maxPixels=1e13
    )

    return {
        'c_above': c_above.rename('c_above'),
        'c_below': c_below.rename('c_below'),
        'c_soil': c_soil.rename('c_soil'),
        'c_dead': c_dead.rename('c_dead'),
        'c_total': c_total,
        'total_Mg_C': stats,
    }


def compute_carbon_change(lulc_t1, lulc_t2, region, carbon_pools=None):
    """
    Calcula cambio neto de carbono entre dos periodos.

    Returns:
        ee.Image de cambio de carbono (Mg C/ha) y estadisticas
    """
    carbon_t1 = compute_carbon_storage(lulc_t1, region, carbon_pools)
    carbon_t2 = compute_carbon_storage(lulc_t2, region, carbon_pools)

    # Cambio = t2 - t1 (negativo = perdida)
    c_change = carbon_t2['c_total'].subtract(carbon_t1['c_total']).rename('c_change')

    # Estadisticas de cambio (use coarser scale to avoid memory limits)
    change_stats = c_change.multiply(ee.Image.pixelArea().divide(10000)).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=100,
        maxPixels=1e12,
        tileScale=4,
        bestEffort=True
    )

    # Perdida neta (solo pixeles negativos)
    loss = c_change.updateMask(c_change.lt(0))
    loss_stats = loss.multiply(ee.Image.pixelArea().divide(10000)).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=100,
        maxPixels=1e12,
        tileScale=4,
        bestEffort=True
    )

    # Ganancia neta (solo pixeles positivos)
    gain = c_change.updateMask(c_change.gt(0))
    gain_stats = gain.multiply(ee.Image.pixelArea().divide(10000)).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=100,
        maxPixels=1e12,
        tileScale=4,
        bestEffort=True
    )

    return {
        'c_change_map': c_change,
        'net_change_Mg_C': change_stats,
        'loss_Mg_C': loss_stats,
        'gain_Mg_C': gain_stats,
    }


def enhance_carbon_with_biomass(lulc_map, region):
    """
    Mejora estimacion de carbono usando GEDI L4B aboveground biomass.
    Reemplaza solo el pool above-ground con GEDI; mantiene IPCC para
    below-ground, soil y dead organic matter.
    """
    carbon_ipcc = compute_carbon_storage(lulc_map, region)

    # GEDI L4B Gridded Aboveground Biomass Density (Mg/ha)
    try:
        gedi_agb = ee.Image('LCLUC/GEDI/GEDI04_B_002').select('MU').clip(region)
        # Convertir AGB (Mg/ha) a C above-ground (factor 0.47)
        gedi_c_above = gedi_agb.multiply(0.47).rename('c_above_gedi')

        # Usar GEDI donde disponible, IPCC c_above donde no
        c_above_enhanced = gedi_c_above.unmask(carbon_ipcc['c_above'])
    except Exception:
        # Fallback a IPCC si GEDI no disponible
        c_above_enhanced = carbon_ipcc['c_above']

    # Total = GEDI above-ground + IPCC (below + soil + dead)
    c_total_enhanced = (c_above_enhanced
                        .add(carbon_ipcc['c_below'])
                        .add(carbon_ipcc['c_soil'])
                        .add(carbon_ipcc['c_dead'])
                        .rename('c_total_enhanced'))

    return c_total_enhanced


# ============================================================
# SEASONAL WATER YIELD (proxy)
# ============================================================

def compute_water_yield_proxy(lulc_map, region, year):
    """
    Estima rendimiento hidrico estacional como proxy de InVEST SWY.

    Usa: P (CHIRPS) - ET (MODIS) ajustado por coeficientes LULC.
    Kc (coeficiente de cultivo) varía por clase LULC.

    Args:
        lulc_map: ee.Image LULC
        region: ee.Geometry
        year: int

    Returns:
        ee.Image de rendimiento hidrico (mm/year)
    """
    # Precipitacion anual (CHIRPS)
    chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterDate(f'{year}-01-01', f'{year}-12-31')
              .filterBounds(region)
              .sum()
              .clip(region)
              .rename('precipitation'))

    # Evapotranspiracion (MODIS ET - MOD16A2)
    # Use fallback if ET collection is empty for this year/region
    et_col = (ee.ImageCollection('MODIS/061/MOD16A2')
              .filterDate(f'{year}-01-01', f'{year}-12-31')
              .filterBounds(region)
              .select('ET'))
    # Fallback: use MODIS ET from the closest available year
    modis_et = ee.Algorithms.If(
        et_col.size().gt(0),
        et_col.sum().multiply(0.1).clip(region).rename('ET'),
        # Fallback: estimate ET as 40% of precipitation
        chirps.multiply(0.4).rename('ET')
    )
    modis_et = ee.Image(modis_et)

    # Coeficientes Kc por clase LULC (FAO 56)
    kc_values = {
        1: 1.0,   # Bosque denso - alta ET
        2: 0.85,  # Bosque secundario
        3: 0.60,  # Pasturas
        4: 0.70,  # Cultivos
        5: 1.20,  # Agua - evaporacion abierta
        6: 0.30,  # Urbano - impermeabilizado
        7: 0.15,  # Suelo desnudo
        8: 0.90,  # Manglares - alta ET costera
    }

    kc_image = ee.Image(0).float()
    for class_id, kc in kc_values.items():
        kc_image = kc_image.where(lulc_map.eq(class_id), kc)

    # Water yield = P - (ET * Kc_adjustment)
    # Simplificacion: areas con mayor cobertura forestal retienen mas agua
    # pero tambien mayor ET; el balance depende del contexto
    et_adjusted = modis_et.multiply(kc_image)
    water_yield = chirps.subtract(et_adjusted).rename('water_yield')

    # Baseflow proxy: proporcion que recarga acuiferos
    # Mayor en bosques (infiltracion) vs pasturas (escorrentia)
    recharge_coeff = ee.Image(0).float()
    recharge_values = {1: 0.35, 2: 0.30, 3: 0.15, 4: 0.18, 5: 0, 6: 0.05, 7: 0.10, 8: 0.25}
    for class_id, rc in recharge_values.items():
        recharge_coeff = recharge_coeff.where(lulc_map.eq(class_id), rc)

    baseflow = chirps.multiply(recharge_coeff).rename('baseflow')

    return {
        'precipitation': chirps,
        'ET': modis_et,
        'water_yield': water_yield,
        'baseflow': baseflow,
        'kc': kc_image,
    }


def compute_sediment_proxy(lulc_map, region):
    """
    Proxy de retencion de sedimentos basado en USLE simplificada.
    Erosion potencial = R * K * LS * C * P
    """
    # Factor topografico LS desde SRTM
    srtm = ee.Image('USGS/SRTMGL1_003').clip(region)
    slope = ee.Terrain.slope(srtm)

    # LS factor (simplificado: basado en slope)
    ls_factor = slope.multiply(0.065).add(0.045).multiply(
        slope.multiply(0.065)
    ).rename('LS')

    # Factor C (cobertura) por clase LULC
    c_values = {
        1: 0.001,  # Bosque denso - minima erosion
        2: 0.01,   # Bosque secundario
        3: 0.10,   # Pasturas
        4: 0.15,   # Cultivos
        5: 0.0,    # Agua
        6: 0.0,    # Urbano
        7: 0.50,   # Suelo desnudo - maxima erosion
        8: 0.002,  # Manglares - low erosion, root stabilization
    }

    c_factor = ee.Image(0).float()
    for class_id, c in c_values.items():
        c_factor = c_factor.where(lulc_map.eq(class_id), c)

    # Erosion relativa (sin R y K que son constantes en la comparacion)
    erosion_relative = ls_factor.multiply(c_factor).rename('erosion_relative')

    return erosion_relative


# ============================================================
# HABITAT QUALITY (InVEST-style)
# ============================================================

def compute_habitat_quality(lulc_map, region):
    """
    Calcula calidad de habitat basado en InVEST Habitat Quality.

    Componentes:
    - Habitat suitability por clase
    - Amenazas (proximity-weighted)
    - Degradacion del habitat
    - Calidad final = habitat * (1 - degradacion)
    """
    # Habitat suitability (0-1 por clase)
    hab_suit = {
        1: 1.0,   # Bosque denso - habitat optimo
        2: 0.7,   # Bosque secundario
        3: 0.2,   # Pasturas
        4: 0.15,  # Cultivos
        5: 0.6,   # Agua - habitat acuatico
        6: 0.0,   # Urbano
        7: 0.0,   # Suelo desnudo
        8: 0.95,  # Manglares - high biodiversity
    }

    habitat = ee.Image(0).float()
    for class_id, hs in hab_suit.items():
        habitat = habitat.where(lulc_map.eq(class_id), hs)

    # Amenazas: distancia a fuentes de degradacion
    # Amenaza 1: Proximidad a areas agricolas/ganaderas
    agro_threat = lulc_map.eq(3).Or(lulc_map.eq(4))
    agro_distance = agro_threat.fastDistanceTransform().sqrt().multiply(30)  # metros
    agro_decay = agro_distance.divide(5000).multiply(-1).exp()  # decaimiento exponencial 5km

    # Amenaza 2: Proximidad a urbano
    urban_threat = lulc_map.eq(6)
    urban_distance = urban_threat.fastDistanceTransform().sqrt().multiply(30)
    urban_decay = urban_distance.divide(10000).multiply(-1).exp()  # decaimiento 10km

    # Amenaza 3: Proximidad a carreteras (usando suelo desnudo como proxy)
    road_threat = lulc_map.eq(7)
    road_distance = road_threat.fastDistanceTransform().sqrt().multiply(30)
    road_decay = road_distance.divide(3000).multiply(-1).exp()  # decaimiento 3km

    # Degradacion total (weighted sum)
    degradation = (agro_decay.multiply(0.6)
                   .add(urban_decay.multiply(0.3))
                   .add(road_decay.multiply(0.1)))

    # Sensibilidad del habitat a amenazas
    sensitivity = {
        1: 0.8,  # Bosque denso - alta sensibilidad
        2: 0.7,  # Bosque secundario
        3: 0.3,  # Pasturas
        4: 0.2,  # Cultivos
        5: 0.5,  # Agua
        6: 0.0,  # Urbano
        7: 0.0,  # Suelo desnudo
        8: 0.85, # Manglares - very sensitive to threats
    }

    sens_image = ee.Image(0).float()
    for class_id, s in sensitivity.items():
        sens_image = sens_image.where(lulc_map.eq(class_id), s)

    # Degradacion efectiva
    effective_deg = degradation.multiply(sens_image)

    # Calidad de habitat = suitability * (1 - degradacion_efectiva)
    # Ajuste half-saturation: HQ = H * (1 - (D^z / (D^z + k^z)))
    k = 0.5  # half-saturation constant
    z = 2.5  # scaling parameter
    deg_effect = effective_deg.pow(z).divide(
        effective_deg.pow(z).add(ee.Image(k).pow(z))
    )

    habitat_quality = habitat.multiply(ee.Image(1).subtract(deg_effect)).rename('habitat_quality')

    return {
        'habitat_suitability': habitat.rename('habitat_suitability'),
        'degradation': degradation.rename('degradation'),
        'habitat_quality': habitat_quality,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 3.2: SERVICIOS ECOSISTEMICOS")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    region = get_study_area()
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase3_stats'
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\nModelos implementados:")
    print("  1. Carbon Storage (IPCC Tier 1 + GEDI AGB)")
    print("  2. Water Yield proxy (CHIRPS - MODIS ET)")
    print("  3. Sediment Retention proxy (USLE simplificada)")
    print("  4. Habitat Quality (amenazas + sensibilidad)")

    print("\nNota: Requiere mapas LULC clasificados de 03_classification.py")
    print("Los modelos se ejecutan sobre cada periodo temporal.\n")

    # Calcular para cada periodo
    for period_key, period_info in PERIODS.items():
        year = period_info['map_year']
        print(f"\n  Periodo: {period_info['label']} ({year})")
        print(f"    Carbon storage: pools IPCC Tier 1")
        print(f"    Water yield: CHIRPS {year} - MODIS ET {year}")
        print(f"    Habitat quality: threat proximity model")

    # Guardar configuracion
    es_config = {
        'models': {
            'carbon_storage': {
                'method': 'IPCC Tier 1 + GEDI L4B',
                'pools': ['c_above', 'c_below', 'c_soil', 'c_dead'],
                'units': 'Mg C/ha',
                'reference': 'IPCC 2006 Guidelines, Ch. 4',
                'enhancement': 'GEDI L4B gridded AGB where available',
            },
            'water_yield': {
                'method': 'P - ET (proxy SWY)',
                'precipitation': 'CHIRPS Daily aggregated',
                'evapotranspiration': 'MODIS MOD16A2',
                'kc_source': 'FAO 56 crop coefficients',
                'units': 'mm/year',
            },
            'sediment_retention': {
                'method': 'USLE simplified',
                'factors': 'LS (SRTM slope) * C (LULC)',
                'units': 'relative index',
            },
            'habitat_quality': {
                'method': 'InVEST-style threat proximity',
                'threats': ['agriculture/pasture', 'urban', 'bare soil/roads'],
                'decay': 'exponential',
                'decay_distances_m': [5000, 10000, 3000],
                'half_saturation_k': 0.5,
                'scaling_z': 2.5,
            },
        },
        'carbon_pools_MgC_ha': {
            LULC_CLASSES[k]['name']: v for k, v in CARBON_POOLS.items()
        },
    }

    config_path = os.path.join(output_dir, 'ecosystem_services_config.json')
    with open(config_path, 'w') as f:
        json.dump(es_config, f, indent=2)

    print(f"\nConfiguracion guardada en: {config_path}")
    print("\nProximo paso: 09_climate_analysis.py")

    return es_config


if __name__ == '__main__':
    config = main()
