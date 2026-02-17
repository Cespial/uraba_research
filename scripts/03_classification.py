"""
03_classification.py
====================
Fase 2.2: Clasificacion supervisada Random Forest en Google Earth Engine.

Clasifica LULC para 4 periodos usando RF con 17 features (6 bandas + 8 indices + 3 topo).
Genera mapas de clasificacion y mapas de probabilidad de clase.

Outputs:
- 4 mapas LULC clasificados (2013, 2016, 2020, 2024)
- 4 mapas de probabilidad (confianza RF)
- Feature importance rankings
- Exportacion a Google Drive
"""

import ee
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, LULC_CLASSES, RF_PARAMS, STUDY_AREA_BBOX
from scripts.utils import (
    create_landsat_composite, create_harmonized_composite,
    get_study_area, export_image_to_drive, print_image_info
)
import importlib as _il
_training_mod = _il.import_module('scripts.02_training_samples')
get_reference_lulc = _training_mod.get_reference_lulc
generate_stratified_samples = _training_mod.generate_stratified_samples
split_train_validation = _training_mod.split_train_validation


# ============================================================
# CLASIFICACION RANDOM FOREST
# ============================================================

def train_random_forest(training_data, feature_bands, label='lulc_reference'):
    """
    Entrena clasificador Random Forest.

    Args:
        training_data: ee.FeatureCollection con valores de bandas y etiquetas
        feature_bands: lista de nombres de bandas para clasificacion
        label: nombre de la columna con etiquetas

    Returns:
        clasificador entrenado
    """
    classifier = ee.Classifier.smileRandomForest(**RF_PARAMS).train(
        features=training_data,
        classProperty=label,
        inputProperties=feature_bands
    )
    return classifier


def classify_image(composite, classifier, feature_bands):
    """
    Aplica clasificador a un composite.
    Retorna mapa clasificado y mapa de probabilidad.
    """
    classified = composite.select(feature_bands).classify(classifier).rename('lulc')

    # Probabilidad de clase (confianza)
    probabilities = (composite.select(feature_bands)
                     .classify(classifier.setOutputMode('MULTIPROBABILITY'))
                     .rename(['prob_1', 'prob_2', 'prob_3', 'prob_4',
                              'prob_5', 'prob_6', 'prob_7']))

    # Mapa de confianza (max probabilidad)
    confidence = probabilities.reduce(ee.Reducer.max()).rename('confidence')

    return classified, confidence, probabilities


def get_feature_importance(classifier, feature_bands):
    """Obtiene importancia de variables del modelo RF."""
    importance = classifier.explain().get('importance')
    return ee.Dictionary(importance)


# ============================================================
# POSTPROCESAMIENTO
# ============================================================

def apply_spatial_filter(classified, kernel_size=3):
    """
    Aplica filtro modal para eliminar pixeles aislados (salt-and-pepper noise).
    """
    kernel = ee.Kernel.square(kernel_size, 'pixels')
    filtered = classified.focal_mode(kernel_size, 'square', 'pixels')
    return filtered.rename('lulc')


def enforce_water_mask(classified, region):
    """
    Fuerza la clase agua usando JRC Global Surface Water para pixeles
    con >80% de ocurrencia temporal.
    """
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(region)
    permanent_water = jrc.select('occurrence').gte(80)
    corrected = classified.where(permanent_water, 5)
    return corrected


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 2.2: CLASIFICACION RANDOM FOREST EN GEE")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    region = get_study_area()
    results = {}
    all_importance = {}

    for period_key, period_info in PERIODS.items():
        year = period_info['map_year']
        start = period_info['start']
        end = period_info['end']

        print(f"\n{'═' * 50}")
        print(f"CLASIFICANDO: {year} ({period_info['label']})")
        print(f"{'═' * 50}")

        # 1. Crear composite
        print("  [1/6] Generando composite...")
        if year >= 2016:
            composite, n_images = create_harmonized_composite(start, end, region)
            feature_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
                             'NDVI', 'EVI', 'NDWI', 'NDBI', 'BSI', 'NBR', 'SAVI', 'MNDWI',
                             'elevation', 'slope', 'aspect']
        else:
            composite, n_images = create_landsat_composite(start, end, region)
            from gee_config import LANDSAT_BANDS
            feature_bands = list(LANDSAT_BANDS.values()) + \
                            ['NDVI', 'EVI', 'NDWI', 'NDBI', 'BSI', 'NBR', 'SAVI', 'MNDWI',
                             'elevation', 'slope', 'aspect']

        print(f"    Imagenes: {n_images.getInfo()}")
        print(f"    Features: {len(feature_bands)} bandas")

        # 2. Generar muestras de entrenamiento
        print("  [2/6] Generando muestras de entrenamiento...")
        reference = get_reference_lulc(year, region)
        samples = generate_stratified_samples(reference, region, n_per_class=500, seed=42 + year)
        training, validation = split_train_validation(samples, 0.7, seed=42)

        # 3. Extraer valores del composite en puntos de muestreo
        print("  [3/6] Extrayendo valores espectrales en puntos...")
        training_data = composite.select(feature_bands).sampleRegions(
            collection=training,
            properties=['lulc_reference'],
            scale=30,
            geometries=True
        )

        validation_data = composite.select(feature_bands).sampleRegions(
            collection=validation,
            properties=['lulc_reference'],
            scale=30,
            geometries=True
        )

        # 4. Entrenar Random Forest
        print(f"  [4/6] Entrenando Random Forest (ntree={RF_PARAMS['numberOfTrees']})...")
        classifier = train_random_forest(training_data, feature_bands)

        # Feature importance
        importance = get_feature_importance(classifier, feature_bands)
        importance_dict = importance.getInfo()
        print("    Feature importance (top 5):")
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp[:5]:
            print(f"      {feat}: {imp:.2f}")
        all_importance[period_key] = importance_dict

        # 5. Clasificar
        print("  [5/6] Clasificando imagen...")
        classified, confidence, probabilities = classify_image(composite, classifier, feature_bands)

        # Postprocesamiento
        classified = apply_spatial_filter(classified, kernel_size=3)
        classified = enforce_water_mask(classified, region)

        # 6. Validacion rapida (accuracy en validation set)
        print("  [6/6] Validacion rapida en hold-out set...")
        validated = validation_data.classify(classifier)
        error_matrix = validated.errorMatrix('lulc_reference', 'classification')
        oa = error_matrix.accuracy()
        kappa = error_matrix.kappa()

        oa_val = oa.getInfo()
        kappa_val = kappa.getInfo()
        print(f"    Overall Accuracy: {oa_val:.4f} ({oa_val*100:.1f}%)")
        print(f"    Kappa: {kappa_val:.4f}")

        # Guardar resultados
        results[period_key] = {
            'year': year,
            'composite': composite,
            'classified': classified,
            'confidence': confidence,
            'probabilities': probabilities,
            'classifier': classifier,
            'training_data': training_data,
            'validation_data': validation_data,
            'error_matrix': error_matrix,
            'overall_accuracy': oa_val,
            'kappa': kappa_val,
            'n_training': training_data.size().getInfo(),
            'n_validation': validation_data.size().getInfo(),
        }

    # ============================================================
    # RESUMEN GENERAL
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE CLASIFICACION")
    print("=" * 60)
    print(f"\n{'Periodo':<30} {'OA':>8} {'Kappa':>8} {'Train':>8} {'Val':>8}")
    print("─" * 62)
    for key, res in results.items():
        label = PERIODS[key]['label'][:28]
        print(f"{label:<30} {res['overall_accuracy']:>7.1%} {res['kappa']:>8.4f} "
              f"{res['n_training']:>8} {res['n_validation']:>8}")

    # Guardar metricas
    metrics = {}
    for key, res in results.items():
        metrics[key] = {
            'year': res['year'],
            'overall_accuracy': res['overall_accuracy'],
            'kappa': res['kappa'],
            'n_training': res['n_training'],
            'n_validation': res['n_validation'],
        }

    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase2_gee', 'classification_metrics.json'
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Guardar feature importance
    imp_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase2_gee', 'feature_importance.json'
    )
    with open(imp_path, 'w') as f:
        json.dump(all_importance, f, indent=2)

    print(f"\nMetricas guardadas en: {metrics_path}")
    print(f"Feature importance guardada en: {imp_path}")

    # Exportar a Drive
    export = input("\nExportar mapas LULC a Google Drive? (s/n): ").strip().lower()
    if export == 's':
        drive_folder = 'magdalena_medio_gee'
        for key, res in results.items():
            year = res['year']
            # Mapa clasificado
            export_image_to_drive(
                image=res['classified'].toByte(),
                description=f'lulc_classified_{year}',
                folder=drive_folder,
                region=region, scale=30
            )
            # Mapa de confianza
            export_image_to_drive(
                image=res['confidence'].toFloat(),
                description=f'lulc_confidence_{year}',
                folder=drive_folder,
                region=region, scale=30
            )
        print(f"\nExportaciones iniciadas. Revisa: https://code.earthengine.google.com/tasks")

    print("\nProximo paso: 04_accuracy_assessment.py")
    return results


if __name__ == '__main__':
    results = main()
