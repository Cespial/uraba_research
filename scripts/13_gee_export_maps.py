#!/usr/bin/env python3
"""
Phase 1: GEE Data Export Pipeline for Cartographic Maps.

Exports raster and vector data from Google Earth Engine to local GeoTIFF/GeoJSON
files for high-quality cartographic map rendering.

Exports:
  - Sentinel-2 true-color composite (2024) at 100m -> GeoTIFF RGB uint8
  - LULC classifications for T1-T4 at 100m -> GeoTIFF uint8
  - Hansen GFC layers (treecover, loss, gain, lossyear) at 100m -> GeoTIFF
  - SRTM hillshade at 100m -> GeoTIFF uint8
  - Administrative boundaries (municipalities, departments, Colombia) -> GeoJSON

Requires: earthengine-api, geemap (for download helpers)

Usage:
    python scripts/13_gee_export_maps.py
"""

import os
import sys
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import (
    STUDY_AREA_BBOX, PERIODS, COLLECTIONS, LULC_CLASSES,
    RF_PARAMS, LANDSAT_BANDS, SENTINEL_BANDS
)
from scripts.utils import (
    mask_landsat_clouds, mask_sentinel2_clouds,
    create_landsat_composite, create_harmonized_composite,
    get_terrain_bands, get_study_area
)

import ee

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'map_exports')
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPORT_SCALE = 100  # meters
EXPORT_CRS = 'EPSG:4326'

# Study area bounds for clipping
BBOX = [-77.2, 7.0, -75.8, 8.9]
REGION = ee.Geometry.Rectangle(BBOX)


# ============================================================
# SENTINEL-2 TRUE-COLOR COMPOSITE
# ============================================================

def export_sentinel2_composite(region, output_path, year=2024, scale=100):
    """
    Export a Sentinel-2 true-color (RGB) composite for the study area.
    Uses median composite from cloud-masked imagery, stretched to uint8.
    """
    print(f"\n--- Exporting Sentinel-2 composite ({year}) ---")

    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
          .map(mask_sentinel2_clouds))

    n_images = s2.size().getInfo()
    print(f"  Images found: {n_images}")

    # True-color composite (B4=Red, B3=Green, B2=Blue)
    composite = s2.select(['B4', 'B3', 'B2']).median().clip(region)

    # Scale to 0-255 uint8 (reflectance already scaled to 0-1 by mask function)
    # Stretch using percentile 2-98 for contrast
    percentiles = composite.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),
        geometry=region,
        scale=500,
        maxPixels=1e9,
        bestEffort=True
    ).getInfo()

    # Build min/max per band
    bands = ['B4', 'B3', 'B2']
    rgb_stretched = ee.Image.cat([
        composite.select(b)
        .unitScale(
            percentiles.get(f'{b}_p2', 0.0),
            percentiles.get(f'{b}_p98', 0.3)
        )
        .multiply(255).clamp(0, 255).uint8()
        for b in bands
    ]).rename(['red', 'green', 'blue'])

    # Download using geemap or getDownloadURL
    _download_ee_image(rgb_stretched, output_path, region, scale,
                       bands=['red', 'green', 'blue'])
    return output_path


# ============================================================
# LULC CLASSIFICATIONS
# ============================================================

def _get_ndvi_composite(region, year):
    """Get NDVI composite from Sentinel-2 or Landsat for a given year.
    Uses a narrow +-1 year window and limits images to reduce computation."""
    start = f'{year - 1}-06-01'
    end = f'{year + 1}-05-31'

    # Try Sentinel-2 first (available from 2017)
    if year >= 2017:
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(start, end)
              .filterBounds(region)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
              .limit(100))
        ndvi = s2.map(lambda img: img.normalizedDifference(['B8', 'B4'])
                      .rename('ndvi').toFloat()).median()
        return ndvi

    # Landsat 8 for earlier years
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(start, end)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUD_COVER', 30))
          .limit(100))
    ndvi = l8.map(lambda img: img.normalizedDifference(['SR_B5', 'SR_B4'])
                  .rename('ndvi').toFloat()).median()
    return ndvi


def export_lulc_classifications(region, output_dir, scale=100):
    """
    Export LULC classification maps for all 4 periods.
    Uses Hansen GFC + Sentinel-2/Landsat NDVI + JRC + GHSL.
    For T3/T4 uses NDVI to verify current forest state (avoids
    Hansen cumulative loss overcount for recent years).
    """
    print("\n--- Exporting LULC classifications (4 periods) ---")

    periods = {
        'T1_2013': 2013,
        'T2_2016': 2016,
        'T3_2020': 2020,
        'T4_2024': 2024,
    }

    # Load reference datasets once
    hansen = ee.Image(COLLECTIONS['hansen'])
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    ghsl = ee.Image('JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020')

    treecover = hansen.select('treecover2000')
    lossyear = hansen.select('lossyear')
    gain = hansen.select('gain')

    output_paths = {}

    for period_key, year in periods.items():
        print(f"\n  Processing {period_key} ({year})...")
        output_path = os.path.join(output_dir, f'lulc_{period_key}.tif')

        # Year offset for Hansen loss year band (encoded as year - 2000)
        loss_year_offset = year - 2000

        # Forest loss: pixels lost by the target year
        lost_by_year = lossyear.gt(0).And(lossyear.lte(loss_year_offset))

        # Get NDVI composite for current period
        ndvi = _get_ndvi_composite(region, year)

        # Dense forest: combine Hansen baseline with NDVI verification
        # Hansen: treecover2000 >= 60 AND not lost
        hansen_dense = treecover.gte(60).And(lost_by_year.Not())

        # NDVI-based: pixels with NDVI > 0.6 are likely dense forest
        ndvi_dense = ndvi.gt(0.6)

        # For T1/T2, trust Hansen more; for T3/T4, combine with NDVI
        if year <= 2016:
            dense_forest = hansen_dense
            if year >= 2016:
                dense_forest = dense_forest.Or(
                    gain.eq(1).And(lost_by_year.Not()).And(treecover.gte(40))
                )
        else:
            # For T3/T4: forest = Hansen says forest OR NDVI confirms forest
            dense_forest = hansen_dense.Or(ndvi_dense)
            # Also include gain pixels
            dense_forest = dense_forest.Or(
                gain.eq(1).And(treecover.gte(30))
            )

        # Secondary forest: moderate NDVI (0.35-0.6) or Hansen treecover 25-60
        ndvi_secondary = ndvi.gte(0.35).And(ndvi.lt(0.6))
        secondary_forest = (
            treecover.gte(25).And(treecover.lt(60)).And(lost_by_year.Not())
            .Or(ndvi_secondary)
        ).And(dense_forest.Not())

        # Water: JRC occurrence > 80%
        water = jrc.select('occurrence').gt(80)

        # Urban: GHSL SMOD >= 22 (tighter threshold to avoid rural misclassification)
        urban = ghsl.select('smod_code').gte(22)

        # Pastures/agriculture: everything else
        pastures = dense_forest.Not().And(secondary_forest.Not()) \
            .And(water.Not()).And(urban.Not())

        # Build classified image
        classified = ee.Image(0) \
            .where(pastures, 3) \
            .where(water, 5) \
            .where(urban, 6) \
            .where(secondary_forest, 2) \
            .where(dense_forest, 1) \
            .rename('classification') \
            .uint8() \
            .clip(region)

        # Apply focal mode to fill small gaps
        classified = classified.focal_mode(1, 'square', 'pixels') \
            .rename('classification').uint8().clip(region)

        # Download
        _download_ee_image(classified, output_path, region, scale,
                           bands=['classification'])
        output_paths[period_key] = output_path

    return output_paths


def _generate_training_samples(region, year):
    """Generate rule-based training samples using Hansen + JRC + GHSL."""
    hansen = ee.Image(COLLECTIONS['hansen'])
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    ghsl = ee.Image('JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020')

    treecover = hansen.select('treecover2000')
    lossyear = hansen.select('lossyear')

    # Dense forest: treecover >= 60, no loss by target year
    loss_year_offset = year - 2000
    no_loss = lossyear.eq(0).Or(lossyear.gt(loss_year_offset))
    dense_forest = treecover.gte(60).And(no_loss)

    # Secondary forest: treecover 30-60, no loss
    secondary_forest = treecover.gte(30).And(treecover.lt(60)).And(no_loss)

    # Water: JRC occurrence > 80%
    water = jrc.select('occurrence').gt(80)

    # Urban: GHSL SMOD >= 20
    urban = ghsl.select('smod_code').gte(20)

    # Pastures: not forest, not water, not urban, low treecover
    pastures = dense_forest.Not().And(secondary_forest.Not()) \
        .And(water.Not()).And(urban.Not()) \
        .And(treecover.lt(30))

    # Create labeled image
    labeled = ee.Image(0) \
        .where(pastures, 3) \
        .where(water, 5) \
        .where(urban, 6) \
        .where(secondary_forest, 2) \
        .where(dense_forest, 1) \
        .rename('class')

    # Stratified random sampling
    samples = labeled.stratifiedSample(
        numPoints=300,
        classBand='class',
        region=region,
        scale=30,
        seed=42,
        geometries=True
    ).filter(ee.Filter.gt('class', 0))

    return samples


# ============================================================
# HANSEN GLOBAL FOREST CHANGE
# ============================================================

def export_hansen_change(region, output_path, scale=100):
    """Export Hansen GFC layers: treecover2000, loss, gain, lossyear."""
    print("\n--- Exporting Hansen GFC layers ---")

    hansen = ee.Image(COLLECTIONS['hansen'])

    # Select relevant bands
    hansen_export = hansen.select([
        'treecover2000', 'loss', 'gain', 'lossyear'
    ]).clip(region)

    _download_ee_image(hansen_export, output_path, region, scale,
                       bands=['treecover2000', 'loss', 'gain', 'lossyear'])
    return output_path


# ============================================================
# SRTM HILLSHADE
# ============================================================

def export_srtm_hillshade(region, output_path, scale=100):
    """Export SRTM-derived hillshade as uint8."""
    print("\n--- Exporting SRTM hillshade ---")

    dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
    hillshade = ee.Terrain.hillshade(dem).clip(region).uint8()

    _download_ee_image(hillshade, output_path, region, scale,
                       bands=['hillshade'])
    return output_path


# ============================================================
# ADMINISTRATIVE BOUNDARIES
# ============================================================

def export_admin_boundaries(region, output_dir):
    """
    Export administrative boundaries as GeoJSON from FAO GAUL.
    - Level 2 (municipalities) within study area
    - Level 1 (departments) intersecting study area
    - Level 0 (Colombia outline)
    """
    print("\n--- Exporting administrative boundaries ---")

    # Colombia outline
    admin0 = ee.FeatureCollection("FAO/GAUL/2015/level0")
    colombia = admin0.filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))
    _export_vector(colombia, os.path.join(output_dir, 'colombia_outline.geojson'))

    # Departments (level 1) intersecting study area
    admin1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    departments = admin1.filter(ee.Filter.eq('ADM0_NAME', 'Colombia')) \
        .filterBounds(region)
    _export_vector(departments, os.path.join(output_dir, 'departments.geojson'))

    # Municipalities (level 2) within study area
    admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    municipalities = admin2.filter(ee.Filter.eq('ADM0_NAME', 'Colombia')) \
        .filterBounds(region)
    _export_vector(municipalities, os.path.join(output_dir, 'municipalities.geojson'))

    print("  Boundaries exported successfully")


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def _download_ee_image(image, output_path, region, scale, bands=None):
    """
    Download an Earth Engine image to local GeoTIFF.
    Uses geemap if available, falls back to ee.Image.getDownloadURL.
    """
    try:
        import geemap
        print(f"  Downloading via geemap to {output_path}...")
        geemap.download_ee_image(
            image,
            filename=output_path,
            region=region,
            scale=scale,
            crs=EXPORT_CRS
        )
        print(f"  [OK] {output_path}")

    except ImportError:
        print("  geemap not available, using getDownloadURL fallback...")
        _download_via_url(image, output_path, region, scale, bands)

    except Exception as e:
        print(f"  geemap download failed: {e}")
        print("  Attempting getDownloadURL fallback...")
        _download_via_url(image, output_path, region, scale, bands)


def _download_via_url(image, output_path, region, scale, bands=None):
    """Fallback download using getDownloadURL."""
    import urllib.request
    import zipfile
    import tempfile

    params = {
        'scale': scale,
        'crs': EXPORT_CRS,
        'region': region.getInfo()['coordinates'],
        'format': 'GEO_TIFF',
    }
    if bands:
        params['bands'] = bands

    try:
        url = image.getDownloadURL(params)
        print(f"  Download URL obtained, fetching...")

        tmp_path = output_path + '.tmp'
        urllib.request.urlretrieve(url, tmp_path)

        # Check if it's a zip file
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path, 'r') as z:
                tif_files = [f for f in z.namelist() if f.endswith('.tif')]
                if tif_files:
                    z.extract(tif_files[0], os.path.dirname(output_path))
                    extracted = os.path.join(os.path.dirname(output_path), tif_files[0])
                    os.rename(extracted, output_path)
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, output_path)

        print(f"  [OK] {output_path}")

    except Exception as e:
        print(f"  Download failed: {e}")
        _download_thumbnail_fallback(image, output_path, region, bands)


def _download_thumbnail_fallback(image, output_path, region, bands=None):
    """Last resort: use getThumbURL for a pre-rendered PNG."""
    import urllib.request

    png_path = output_path.replace('.tif', '.png')
    print(f"  Using thumbnail fallback -> {png_path}")

    try:
        params = {
            'region': region.getInfo(),
            'dimensions': '2048x2048',
            'format': 'png',
        }
        if bands and len(bands) >= 3:
            params['bands'] = bands[:3]

        url = image.getThumbURL(params)
        urllib.request.urlretrieve(url, png_path)
        print(f"  [OK] {png_path} (thumbnail)")

    except Exception as e:
        print(f"  Thumbnail fallback also failed: {e}")
        print(f"  SKIPPING: {output_path}")


def _export_vector(fc, output_path):
    """Export a FeatureCollection to local GeoJSON."""
    try:
        import geemap
        print(f"  Exporting vector to {output_path}...")
        geemap.ee_export_vector(fc, filename=output_path)
        print(f"  [OK] {output_path}")

    except ImportError:
        print("  geemap not available, using getInfo fallback...")
        _export_vector_via_getinfo(fc, output_path)

    except Exception as e:
        print(f"  geemap export failed: {e}")
        _export_vector_via_getinfo(fc, output_path)


def _export_vector_via_getinfo(fc, output_path):
    """Fallback vector export using getInfo() -> GeoJSON."""
    try:
        geojson = fc.getInfo()
        # Convert to proper GeoJSON FeatureCollection
        features = []
        for feat in geojson.get('features', []):
            features.append({
                'type': 'Feature',
                'geometry': feat.get('geometry', {}),
                'properties': feat.get('properties', {})
            })
        output = {
            'type': 'FeatureCollection',
            'features': features
        }
        with open(output_path, 'w') as f:
            json.dump(output, f)
        print(f"  [OK] {output_path}")

    except Exception as e:
        print(f"  Vector export failed: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    from datetime import datetime
    print("=" * 60)
    print("PHASE 1: GEE DATA EXPORT PIPELINE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Scale: {EXPORT_SCALE}m | CRS: {EXPORT_CRS}")
    print("=" * 60)

    t0 = time.time()

    # 1. Sentinel-2 true-color composite
    export_sentinel2_composite(
        REGION,
        os.path.join(OUTPUT_DIR, 'sentinel2_truecolor_2024.tif'),
        year=2024,
        scale=EXPORT_SCALE
    )

    # 2. LULC classifications (4 periods)
    export_lulc_classifications(REGION, OUTPUT_DIR, scale=EXPORT_SCALE)

    # 3. Hansen GFC layers
    export_hansen_change(
        REGION,
        os.path.join(OUTPUT_DIR, 'hansen_gfc.tif'),
        scale=EXPORT_SCALE
    )

    # 4. SRTM hillshade
    export_srtm_hillshade(
        REGION,
        os.path.join(OUTPUT_DIR, 'srtm_hillshade.tif'),
        scale=EXPORT_SCALE
    )

    # 5. Administrative boundaries
    export_admin_boundaries(REGION, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"EXPORT COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
