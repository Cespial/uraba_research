"""
Funciones auxiliares para el proyecto Magdalena Medio LULCC.
Incluye: cloud masking, indices espectrales, exportacion, visualizacion.
"""

import ee
import os
import sys

# Agregar directorio raiz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import (
    LANDSAT_BANDS, SENTINEL_BANDS, SPECTRAL_INDICES,
    LULC_CLASSES, STUDY_AREA_BBOX
)


# ============================================================
# CLOUD MASKING
# ============================================================

def mask_landsat_clouds(image):
    """Cloud mask para Landsat 8/9 Collection 2 Level 2 usando QA_PIXEL."""
    qa = image.select('QA_PIXEL')
    # Bits: 3=cloud shadow, 4=cloud, 5=cloud shadow (high conf)
    cloud_shadow_bit = 1 << 3
    cloud_bit = 1 << 4
    cirrus_bit = 1 << 2
    mask = (qa.bitwiseAnd(cloud_shadow_bit).eq(0)
            .And(qa.bitwiseAnd(cloud_bit).eq(0))
            .And(qa.bitwiseAnd(cirrus_bit).eq(0)))
    # Aplicar scale factors Landsat C2 L2
    optical = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']) \
        .multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True).updateMask(mask)


def mask_sentinel2_clouds(image):
    """Cloud mask para Sentinel-2 SR usando SCL band."""
    scl = image.select('SCL')
    # SCL: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus, 11=snow
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
            .And(scl.neq(10)).And(scl.neq(11)))
    # Sentinel-2 SR ya esta en reflectancia * 10000
    optical = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']) \
        .multiply(0.0001)
    return image.addBands(optical, overwrite=True).updateMask(mask)


# ============================================================
# INDICES ESPECTRALES
# ============================================================

def add_spectral_indices(image, sensor='landsat'):
    """Agrega indices espectrales a una imagen Landsat o Sentinel-2."""
    if sensor == 'landsat':
        bands = LANDSAT_BANDS
    else:
        bands = SENTINEL_BANDS

    blue = image.select(bands['blue'])
    green = image.select(bands['green'])
    red = image.select(bands['red'])
    nir = image.select(bands['nir'])
    swir1 = image.select(bands['swir1'])
    swir2 = image.select(bands['swir2'])

    # NDVI
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')

    # EVI
    evi = nir.subtract(red).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).multiply(2.5).rename('EVI')

    # NDWI (McFeeters)
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')

    # NDBI
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI')

    # BSI (Bare Soil Index)
    bsi = swir1.add(red).subtract(nir).subtract(blue).divide(
        swir1.add(red).add(nir).add(blue)
    ).rename('BSI')

    # NBR
    nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename('NBR')

    # SAVI (L=0.5)
    savi = nir.subtract(red).divide(nir.add(red).add(0.5)).multiply(1.5).rename('SAVI')

    # MNDWI
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI')

    return image.addBands([ndvi, evi, ndwi, ndbi, bsi, nbr, savi, mndwi])


# ============================================================
# TOPOGRAFIA
# ============================================================

def get_terrain_bands():
    """Obtiene bandas de terreno desde SRTM."""
    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.select('elevation').rename('elevation')
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    return elevation.addBands([slope, aspect])


# ============================================================
# COMPOSITES
# ============================================================

def create_landsat_composite(start_date, end_date, region):
    """Crea composite de mediana Landsat 8/9 con cloud masking e indices."""
    # Landsat 8
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUD_COVER', 70))
          .map(mask_landsat_clouds)
          .map(lambda img: add_spectral_indices(img, 'landsat')))

    # Landsat 9 (disponible desde 2022)
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUD_COVER', 70))
          .map(mask_landsat_clouds)
          .map(lambda img: add_spectral_indices(img, 'landsat')))

    # Merge y mediana
    merged = l8.merge(l9)

    # Bandas de reflectancia + indices
    bands = list(LANDSAT_BANDS.values()) + SPECTRAL_INDICES
    composite = merged.select(bands).median().clip(region)

    # Agregar terreno
    terrain = get_terrain_bands().clip(region)
    composite = composite.addBands(terrain)

    return composite, merged.size()


def create_sentinel2_composite(start_date, end_date, region):
    """Crea composite de mediana Sentinel-2 SR con cloud masking e indices."""
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
          .map(mask_sentinel2_clouds)
          .map(lambda img: add_spectral_indices(img, 'sentinel')))

    bands = list(SENTINEL_BANDS.values()) + SPECTRAL_INDICES
    composite = s2.select(bands).median().clip(region)

    terrain = get_terrain_bands().clip(region)
    composite = composite.addBands(terrain)

    return composite, s2.size()


def create_harmonized_composite(start_date, end_date, region):
    """
    Crea composite armonizado Landsat+Sentinel-2.
    Usa nombres de banda estandarizados para compatibilidad.
    """
    # Renombrar bandas a nombres comunes
    common_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

    # Landsat 8/9
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUD_COVER', 70))
          .map(mask_landsat_clouds))
    l8 = l8.select(list(LANDSAT_BANDS.values()), common_names)
    l8 = l8.map(lambda img: img.toFloat())

    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUD_COVER', 70))
          .map(mask_landsat_clouds))
    l9 = l9.select(list(LANDSAT_BANDS.values()), common_names)
    l9 = l9.map(lambda img: img.toFloat())

    # Sentinel-2 (resample a 30m para compatibilidad)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
          .map(mask_sentinel2_clouds))
    s2 = s2.select(list(SENTINEL_BANDS.values()), common_names)
    s2 = s2.map(lambda img: img.toFloat())

    # Merge todas las colecciones
    merged = l8.merge(l9).merge(s2)

    # Calcular indices con nombres comunes
    def add_indices_common(image):
        nir = image.select('nir')
        red = image.select('red')
        green = image.select('green')
        blue = image.select('blue')
        swir1 = image.select('swir1')
        swir2 = image.select('swir2')

        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        evi = nir.subtract(red).divide(
            nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
        ).multiply(2.5).rename('EVI')
        ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
        ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI')
        bsi = swir1.add(red).subtract(nir).subtract(blue).divide(
            swir1.add(red).add(nir).add(blue)).rename('BSI')
        nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename('NBR')
        savi = nir.subtract(red).divide(nir.add(red).add(0.5)).multiply(1.5).rename('SAVI')
        mndwi = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI')

        return image.addBands([ndvi, evi, ndwi, ndbi, bsi, nbr, savi, mndwi])

    merged = merged.map(add_indices_common)

    # Composite de mediana
    all_bands = common_names + SPECTRAL_INDICES
    composite = merged.select(all_bands).median().clip(region)

    # Agregar terreno
    terrain = get_terrain_bands().clip(region)
    composite = composite.addBands(terrain)

    return composite, merged.size()


# ============================================================
# EXPORTACION
# ============================================================

def export_image_to_drive(image, description, folder, region, scale=30):
    """Exporta imagen GEE a Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        maxPixels=1e13,
        crs='EPSG:4326'
    )
    task.start()
    print(f"Exportando: {description}")
    return task


def export_table_to_drive(fc, description, folder):
    """Exporta FeatureCollection a Google Drive como CSV."""
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=folder,
        fileFormat='CSV'
    )
    task.start()
    print(f"Exportando tabla: {description}")
    return task


# ============================================================
# VISUALIZACION
# ============================================================

def get_lulc_vis_params():
    """Parametros de visualizacion para mapas LULC."""
    colors = [cls['color'] for cls in LULC_CLASSES.values()]
    return {
        'min': 1,
        'max': 7,
        'palette': colors
    }


def get_ndvi_vis_params():
    """Parametros de visualizacion para NDVI."""
    return {
        'min': -0.2,
        'max': 0.9,
        'palette': ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    }


# ============================================================
# AREA DE ESTUDIO
# ============================================================

def get_study_area_from_admin():
    """
    Obtiene area de estudio desde limites administrativos FAO GAUL.
    Alternativa: usar shapefile DANE cargado como asset.
    """
    admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    colombia = admin2.filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))

    # Filtrar municipios clave del Magdalena Medio
    # Nota: los nombres deben coincidir exactamente con FAO GAUL
    key_municipalities = [
        'Barrancabermeja', 'Puerto Wilches', 'Cimitarra',
        'Puerto Berrio', 'Yondo', 'San Pablo',
        'Aguachica', 'Cantagallo', 'Simiti'
    ]

    filters = ee.Filter.inList('ADM2_NAME', key_municipalities)
    study_municipalities = colombia.filter(filters)

    # Si no se encuentran suficientes, usar bbox
    return study_municipalities.geometry().bounds()


def get_study_area():
    """Retorna el area de estudio (bbox o municipios)."""
    return STUDY_AREA_BBOX


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def print_image_info(image, name="Image"):
    """Imprime informacion basica de una imagen GEE."""
    band_names = image.bandNames().getInfo()
    print(f"\n{name}:")
    print(f"  Bandas ({len(band_names)}): {band_names}")


def print_collection_info(collection, name="Collection"):
    """Imprime informacion de una coleccion GEE."""
    size = collection.size().getInfo()
    print(f"\n{name}: {size} imagenes")
    return size
