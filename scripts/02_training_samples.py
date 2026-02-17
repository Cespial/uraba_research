"""
02_training_samples.py
======================
Fase 2.2: Generacion de muestras de entrenamiento y validacion.
Adapted for Uraba Antioqueno: 8 classes including Mangroves (class 8).

Estrategia: muestreo estratificado aleatorio usando multiples fuentes
de referencia (MapBiomas, Hansen GFC, Global Mangrove Watch, indices espectrales).

Outputs:
- FeatureCollection de training samples por periodo
- FeatureCollection de validation samples (30% hold-out)
- Exportacion a Google Drive como CSV
"""

import ee
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, LULC_CLASSES, STUDY_AREA_BBOX, N_CLASSES
from scripts.utils import get_study_area


# ============================================================
# FUENTES DE REFERENCIA PARA TRAINING
# ============================================================

def get_reference_lulc(year, region):
    """
    Genera mapa de referencia LULC combinando multiples fuentes.
    8 classes for Uraba including Mangroves (class 8) from GMW.
    """
    # --- Hansen GFC v1.12: separar bosque denso ---
    hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
    treecover2000 = hansen.select('treecover2000')
    loss = hansen.select('lossyear')

    # Bosque con >60% cobertura en 2000 y sin perdida hasta 'year'
    # Use gte to include loss IN the target year (forest existed at start of year)
    year_offset = year - 2000
    forest_dense = treecover2000.gte(60).And(
        loss.eq(0).Or(loss.gte(year_offset))
    )

    # Bosque secundario: 30-60% cobertura
    forest_secondary = treecover2000.gte(30).And(treecover2000.lt(60)).And(
        loss.eq(0).Or(loss.gte(year_offset))
    )

    # --- JRC Water: cuerpos de agua ---
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    water = jrc.select('occurrence').gte(50)  # agua permanente >50% del tiempo

    # --- GHSL: areas urbanas ---
    ghsl = ee.Image('JRC/GHSL/P2023A/GHS_SMOD/2020').select('smod_code')
    urban = ghsl.gte(22)  # smod_code >= 22 = semi-dense urban clusters + urban centres only

    # --- Mangrove extent from USGS (Giri et al. 2011) ---
    # Eager test: verify dataset accessibility before incorporating
    has_mangrove = False
    mangrove = ee.Image(0)
    try:
        mf = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS')
        # Eager check — force server evaluation to detect access errors
        mf.size().getInfo()
        mangrove = mf.mosaic().select('1').gte(1)
        has_mangrove = True
    except Exception:
        print("  WARNING: Mangrove dataset unavailable, skipping class 8")

    # --- Combinar en mapa de referencia ---
    # Prioridad: mangroves > agua > urbano > bosque denso > bosque secundario > resto (pasturas default)
    reference = ee.Image(3)  # Default: pasturas
    reference = reference.where(forest_secondary, 2)
    reference = reference.where(forest_dense, 1)
    reference = reference.where(urban, 6)
    reference = reference.where(water, 5)
    # Mangroves override forest in coastal areas
    if has_mangrove:
        reference = reference.where(mangrove, 8)

    return reference.clip(region).rename('lulc_reference')


def get_mapbiomas_reference(year, region):
    """
    Intenta cargar referencia MapBiomas Colombia Collection 2.0.
    Reclasifica a nuestras 7 clases.
    """
    try:
        # MapBiomas Colombia - ajustar asset ID segun disponibilidad
        mapbiomas = ee.Image(
            f'projects/mapbiomas-public/assets/colombia/collection2/mapbiomas_colombia_collection2_integration_v1'
        ).select(f'classification_{year}')

        # Reclasificacion MapBiomas -> 8 clases (Uraba)
        # MapBiomas clases principales:
        # 3=Forest, 4=Savanna, 5=Mangrove, 12=Grassland, 15=Pasture,
        # 18=Agriculture, 21=Mosaic, 24=Urban, 25=Bare, 26=Water, 33=Water
        remap_from = [3, 4, 5, 6, 11, 12, 13, 15, 18, 19, 20, 21, 24, 25, 26, 29, 30, 33]
        remap_to =   [1, 2, 8, 2, 2,  3,  3,  3,  4,  4,  4,  4,  6,  7,  5,  7,  7,  5]

        reclassified = mapbiomas.remap(remap_from, remap_to).rename('lulc_mapbiomas')
        return reclassified.clip(region)
    except Exception as e:
        print(f"  MapBiomas no disponible para {year}: {e}")
        return None


# ============================================================
# MUESTREO ESTRATIFICADO
# ============================================================

def generate_stratified_samples(reference_map, region, n_per_class=500, seed=42):
    """
    Genera muestras estratificadas aleatorias desde mapa de referencia.

    Args:
        reference_map: ee.Image con valores 1-8
        region: ee.Geometry del area de estudio
        n_per_class: numero de puntos por clase
        seed: semilla para reproducibilidad

    Returns:
        ee.FeatureCollection con puntos de entrenamiento
    """
    n_classes = N_CLASSES  # 8 for Uraba (includes Mangroves)
    # Muestreo estratificado
    samples = reference_map.stratifiedSample(
        numPoints=n_per_class,
        classBand='lulc_reference',
        region=region,
        scale=30,
        seed=seed,
        geometries=True,
        classValues=list(range(1, n_classes + 1)),
        classPoints=[n_per_class] * n_classes
    )

    return samples


def split_train_validation(samples, train_fraction=0.7, seed=42):
    """
    Divide samples en entrenamiento (70%) y validacion (30%).
    Usa columna aleatoria para split reproducible.
    """
    samples = samples.randomColumn('random', seed)
    training = samples.filter(ee.Filter.lt('random', train_fraction))
    validation = samples.filter(ee.Filter.gte('random', train_fraction))
    return training, validation


# ============================================================
# EXTRAER VALORES DE COMPOSITE
# ============================================================

def extract_composite_values(samples, composite, label_band='lulc_reference'):
    """
    Extrae valores del composite en las ubicaciones de los puntos de muestreo.
    """
    enriched = composite.sampleRegions(
        collection=samples,
        properties=[label_band],
        scale=30,
        geometries=True
    )
    return enriched


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 2.2: GENERACION DE MUESTRAS DE ENTRENAMIENTO")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    region = get_study_area()
    all_samples = {}
    sample_stats = {}

    for period_key, period_info in PERIODS.items():
        year = period_info['map_year']
        print(f"\n{'─' * 50}")
        print(f"Generando muestras para {year} ({period_info['label']})")
        print(f"{'─' * 50}")

        # 1. Generar mapa de referencia
        print("  Generando mapa de referencia multi-fuente...")
        reference = get_reference_lulc(year, region)

        # 2. Intentar MapBiomas como referencia adicional
        print("  Intentando cargar MapBiomas Colombia...")
        mapbiomas = get_mapbiomas_reference(year, region)
        if mapbiomas is not None:
            print("  MapBiomas cargado - se usara para validacion cruzada")

        # 3. Muestreo estratificado
        n_per_class = 500
        print(f"  Muestreo estratificado: {n_per_class} puntos/clase x {N_CLASSES} clases = {n_per_class * N_CLASSES} total")
        samples = generate_stratified_samples(reference, region, n_per_class, seed=42 + year)

        # 4. Split train/validation
        training, validation = split_train_validation(samples, 0.7, seed=42)

        n_train = training.size().getInfo()
        n_val = validation.size().getInfo()
        print(f"  Training: {n_train} puntos")
        print(f"  Validation: {n_val} puntos")

        all_samples[period_key] = {
            'reference': reference,
            'training': training,
            'validation': validation,
            'mapbiomas': mapbiomas,
        }

        sample_stats[period_key] = {
            'year': year,
            'n_per_class': n_per_class,
            'n_training': n_train,
            'n_validation': n_val,
            'total': n_train + n_val,
        }

    # ============================================================
    # RESUMEN
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE MUESTRAS GENERADAS")
    print("=" * 60)

    for key, stats in sample_stats.items():
        print(f"  {PERIODS[key]['label']}:")
        print(f"    Training: {stats['n_training']} | Validation: {stats['n_validation']} | Total: {stats['total']}")

    # Guardar estadisticas
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase2_gee', 'training_samples_stats.json'
    )
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(sample_stats, f, indent=2)
    print(f"\nEstadisticas guardadas en: {stats_path}")

    print("\nProximo paso: 03_classification.py")
    return all_samples, sample_stats


if __name__ == '__main__':
    all_samples, sample_stats = main()
