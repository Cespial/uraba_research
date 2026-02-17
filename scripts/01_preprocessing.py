"""
01_preprocessing.py
===================
Fase 2.1: Adquisicion de datos y preprocesamiento en Google Earth Engine.

Genera composites multi-temporales con cloud masking e indices espectrales
para los 4 periodos de analisis del Magdalena Medio.

Outputs:
- 4 composites Landsat+Sentinel (2013, 2016, 2020, 2024) con 17 bandas
- Metadata de imagenes disponibles por periodo
- Exportacion opcional a Google Drive
"""

import ee
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, COLLECTIONS, STUDY_AREA_BBOX
from scripts.utils import (
    create_landsat_composite,
    create_harmonized_composite,
    get_study_area,
    get_terrain_bands,
    export_image_to_drive,
    print_image_info,
)


def main():
    print("=" * 60)
    print("FASE 2.1: PREPROCESAMIENTO Y COMPOSITES GEE")
    print(f"Fecha de ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    region = get_study_area()
    composites = {}
    metadata = {}

    for period_key, period_info in PERIODS.items():
        print(f"\n{'─' * 50}")
        print(f"Procesando: {period_info['label']}")
        print(f"  Ventana: {period_info['start']} a {period_info['end']}")
        print(f"{'─' * 50}")

        start = period_info['start']
        end = period_info['end']
        map_year = period_info['map_year']

        # Para periodos con Sentinel-2 disponible (>=2016), usar composite armonizado
        if map_year >= 2016:
            print("  Usando composite armonizado Landsat 8/9 + Sentinel-2...")
            composite, n_images = create_harmonized_composite(start, end, region)
        else:
            print("  Usando composite Landsat 8 (Sentinel-2 no disponible)...")
            composite, n_images = create_landsat_composite(start, end, region)

        n_img = n_images.getInfo()
        print(f"  Imagenes disponibles: {n_img}")

        # Verificar bandas
        print_image_info(composite, f"  Composite {map_year}")

        composites[period_key] = composite
        metadata[period_key] = {
            'period': period_key,
            'label': period_info['label'],
            'map_year': map_year,
            'start_date': start,
            'end_date': end,
            'n_images': n_img,
            'bands': composite.bandNames().getInfo(),
            'context': period_info['context'],
        }

    # ============================================================
    # RESUMEN
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE COMPOSITES GENERADOS")
    print("=" * 60)

    for key, meta in metadata.items():
        print(f"\n  {meta['label']}")
        print(f"    Anio mapa: {meta['map_year']}")
        print(f"    Imagenes: {meta['n_images']}")
        print(f"    Bandas: {len(meta['bands'])}")

    # Guardar metadata
    metadata_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'phase2_gee', 'composites_metadata.json'
    )
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata guardada en: {metadata_path}")

    # ============================================================
    # EXPORTAR A DRIVE (opcional)
    # ============================================================
    export = input("\nExportar composites a Google Drive? (s/n): ").strip().lower()
    if export == 's':
        drive_folder = 'magdalena_medio_gee'
        tasks = []
        for key, comp in composites.items():
            year = PERIODS[key]['map_year']
            task = export_image_to_drive(
                image=comp.toFloat(),
                description=f'composite_magdalena_medio_{year}',
                folder=drive_folder,
                region=region,
                scale=30
            )
            tasks.append(task)
        print(f"\n{len(tasks)} tareas de exportacion iniciadas.")
        print(f"Revisa el progreso en: https://code.earthengine.google.com/tasks")

    # ============================================================
    # DATOS AUXILIARES
    # ============================================================
    print("\n" + "=" * 60)
    print("PROCESANDO DATOS AUXILIARES")
    print("=" * 60)

    # Hansen Global Forest Change
    print("\n  Cargando Hansen GFC v1.11...")
    hansen = ee.Image(COLLECTIONS['hansen']).clip(region)
    treecover2000 = hansen.select('treecover2000')
    loss = hansen.select('loss')
    gain = hansen.select('gain')
    lossyear = hansen.select('lossyear')

    # Calcular perdida forestal por periodo
    for period_key, period_info in PERIODS.items():
        year_start = int(period_info['start'][:4]) - 2000
        year_end = int(period_info['end'][:4]) - 2000
        period_loss = lossyear.gte(year_start).And(lossyear.lte(year_end))
        loss_area = period_loss.multiply(ee.Image.pixelArea()).divide(10000)
        total_loss = loss_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=30,
            maxPixels=1e13
        )
        print(f"  Perdida forestal Hansen {period_info['label']}: calculando...")

    # CHIRPS precipitacion media anual
    print("\n  Cargando CHIRPS precipitacion...")
    for year in [2013, 2016, 2020, 2024]:
        chirps_year = (ee.ImageCollection(COLLECTIONS['chirps'])
                       .filterDate(f'{year}-01-01', f'{year}-12-31')
                       .sum()
                       .clip(region))

    # MODIS LST
    print("  Cargando MODIS LST...")
    modis_lst = (ee.ImageCollection(COLLECTIONS['modis_lst'])
                 .filterDate('2012-01-01', '2024-12-31')
                 .filterBounds(region)
                 .select('LST_Day_1km'))

    # WorldPop
    print("  Cargando WorldPop...")
    for year in [2013, 2016, 2020]:
        worldpop = (ee.ImageCollection(COLLECTIONS['worldpop'])
                    .filterDate(f'{year}-01-01', f'{year}-12-31')
                    .first()
                    .clip(region))

    # SRTM + derivados
    print("  Procesando terreno SRTM...")
    terrain = get_terrain_bands().clip(region)

    # JRC Water
    print("  Cargando JRC Global Surface Water...")
    jrc_water = ee.Image(COLLECTIONS['jrc_water']).clip(region)
    water_occurrence = jrc_water.select('occurrence')

    print("\nDatos auxiliares cargados.")

    # ============================================================
    # QUALITY CHECK: Cobertura de pixeles validos
    # ============================================================
    print("\n" + "=" * 60)
    print("QUALITY CHECK: Cobertura de pixeles validos")
    print("=" * 60)

    for key, comp in composites.items():
        year = PERIODS[key]['map_year']
        # Contar pixeles validos (no enmascarados) vs total
        valid_mask = comp.select(0).mask()
        valid_pixels = valid_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=300,  # escala gruesa para rapido
            maxPixels=1e10
        )
        print(f"  Composite {year}: cobertura valida calculando...")

    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print("\nProximo paso: 02_training_samples.py")

    return composites, metadata


if __name__ == '__main__':
    composites, metadata = main()
