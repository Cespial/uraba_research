"""
06_fragmentation.py
===================
Fase 2.3: Analisis de fragmentacion del paisaje.

Calcula metricas FRAGSTATS a nivel de clase y paisaje para la clase bosque,
comparando periodos pre y post-acuerdo.

Metricas: NP, PD, LPI, ED, COHESION, AI, MESH, ENN_MN

Outputs:
- Metricas de fragmentacion por periodo
- Comparacion temporal de metricas
- Datos para graficas de fragmentacion
"""

import ee
import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, LULC_CLASSES, STUDY_AREA_BBOX
from scripts.utils import get_study_area


# ============================================================
# METRICAS DE FRAGMENTACION (GEE-compatible)
# ============================================================

def compute_patch_metrics_gee(lulc_map, target_class, region, scale=30):
    """
    Calcula metricas de fragmentacion para una clase objetivo usando GEE.

    Args:
        lulc_map: ee.Image con clasificacion LULC
        target_class: int, clase a analizar (1=bosque denso)
        region: ee.Geometry
        scale: resolucion en metros

    Returns:
        dict con metricas de fragmentacion
    """
    # Mascara binaria de la clase objetivo
    class_mask = lulc_map.eq(target_class).selfMask()

    # --- Area total de la clase ---
    class_area = class_mask.multiply(ee.Image.pixelArea()).divide(10000)
    total_area_ha = class_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13
    )

    # --- Area total del paisaje ---
    landscape_area = ee.Image.pixelArea().divide(10000).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13
    )

    # --- Numero de parches (connected components) ---
    patches = class_mask.connectedComponents(
        connectedness=ee.Kernel.plus(1),
        maxSize=1024
    )
    patch_labels = patches.select('labels')

    # Contar parches unicos
    n_patches = patch_labels.reduceRegion(
        reducer=ee.Reducer.countDistinctNonNull(),
        geometry=region,
        scale=scale,
        maxPixels=1e13
    )

    # --- Largest Patch Index (LPI) ---
    # Area de cada parche
    patch_areas = class_mask.multiply(ee.Image.pixelArea()).divide(10000) \
        .addBands(patch_labels)
    max_patch_area = patch_areas.select(0).reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=region,
        scale=scale,
        maxPixels=1e13
    )

    # --- Edge Density (ED) ---
    # Detectar bordes: pixeles de clase adyacentes a pixeles de no-clase
    edges = class_mask.focal_min(1, 'square').neq(class_mask).And(class_mask)
    edge_length = edges.multiply(scale)  # metros de borde por pixel
    total_edge = edge_length.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13
    )

    return {
        'total_area': total_area_ha,
        'landscape_area': landscape_area,
        'n_patches': n_patches,
        'max_patch_area': max_patch_area,
        'total_edge': total_edge,
    }


def compute_fragmentation_local(lulc_array, target_class=1, pixel_size=30):
    """
    Calcula metricas FRAGSTATS desde array numpy (para datos exportados).
    Usa pylandstats si esta disponible, sino calculo manual.

    Args:
        lulc_array: numpy array 2D con clasificacion
        target_class: clase a analizar
        pixel_size: tamanio de pixel en metros

    Returns:
        dict con metricas de fragmentacion
    """
    try:
        import pylandstats as pls
        landscape = pls.Landscape(lulc_array, res=(pixel_size, pixel_size))

        metrics = {
            'NP': landscape.number_of_patches(target_class),
            'PD': landscape.patch_density(target_class),
            'LPI': landscape.largest_patch_index(target_class),
            'TE': landscape.total_edge(target_class),
            'ED': landscape.edge_density(target_class),
            'AREA_MN': landscape.area_mn(target_class),
            'AREA_SD': landscape.area_sd(target_class),
            'COHESION': landscape.patch_cohesion_index(target_class),
            'AI': landscape.aggregation_index(target_class),
        }
        return metrics

    except ImportError:
        # Calculo manual basico
        binary = (lulc_array == target_class).astype(int)
        total_pixels = binary.sum()
        total_area_ha = total_pixels * (pixel_size ** 2) / 10000

        # Deteccion de bordes simple
        from scipy import ndimage
        labeled, n_patches = ndimage.label(binary)
        patch_sizes = ndimage.sum(binary, labeled, range(1, n_patches + 1))

        if n_patches > 0 and total_area_ha > 0:
            landscape_area_ha = lulc_array.size * (pixel_size ** 2) / 10000
            metrics = {
                'NP': int(n_patches),
                'PD': round(n_patches / landscape_area_ha * 100, 4),
                'LPI': round(max(patch_sizes) * (pixel_size ** 2) / 10000 / landscape_area_ha * 100, 4),
                'AREA_MN': round(np.mean(patch_sizes) * (pixel_size ** 2) / 10000, 4),
                'AREA_SD': round(np.std(patch_sizes) * (pixel_size ** 2) / 10000, 4),
                'class_area_ha': round(total_area_ha, 1),
            }
        else:
            metrics = {
                'NP': 0, 'PD': 0, 'LPI': 0,
                'AREA_MN': 0, 'AREA_SD': 0,
                'class_area_ha': 0,
            }

        return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 2.3: ANALISIS DE FRAGMENTACION DEL PAISAJE")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    region = get_study_area()
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase2_gee'
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\nMetricas a calcular:")
    print("  - NP (Number of Patches)")
    print("  - PD (Patch Density)")
    print("  - LPI (Largest Patch Index)")
    print("  - ED (Edge Density)")
    print("  - COHESION (Patch Cohesion Index)")
    print("  - AI (Aggregation Index)")
    print("  - AREA_MN (Mean Patch Area)")
    print("  - ENN_MN (Mean Euclidean Nearest Neighbor)")

    print("\nClases objetivo: Bosque denso (1), Bosque secundario (2)")
    print("Nivel de analisis: Clase + Paisaje")

    print("\nNota: Las metricas se calculan despues de exportar mapas LULC.")
    print("Opciones de computo:")
    print("  a) GEE server-side (aproximado, para parches grandes)")
    print("  b) Local con pylandstats (preciso, requiere export previo)")

    # Guardar configuracion
    frag_config = {
        'target_classes': [1, 2],
        'class_names': ['Bosque denso', 'Bosque secundario'],
        'metrics': ['NP', 'PD', 'LPI', 'ED', 'COHESION', 'AI', 'AREA_MN', 'ENN_MN'],
        'connectivity': '8-cell neighborhood',
        'scale': 30,
        'analysis_levels': ['class', 'landscape'],
        'periods': {k: v['map_year'] for k, v in PERIODS.items()},
        'software': 'pylandstats / scipy.ndimage / GEE connectedComponents',
    }

    config_path = os.path.join(output_dir, 'fragmentation_config.json')
    with open(config_path, 'w') as f:
        json.dump(frag_config, f, indent=2)

    print(f"\nConfiguracion guardada en: {config_path}")
    print("\nProximo paso: 07_hotspot_analysis.py")

    return frag_config


if __name__ == '__main__':
    config = main()
