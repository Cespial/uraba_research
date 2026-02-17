"""
run_analysis.py
===============
Script maestro Fase 3: Pipeline analitico completo para Uraba Antioqueno.
Optimizado para limites de memoria GEE.

Stages:
  1. Clasificacion LULC (4 periodos, 8 clases incl. manglares)
  2. Deteccion de cambio + transiciones
  3. Servicios ecosistemicos
  4. Analisis climatico
  5. Hotspot analysis
  6. GWR drivers
  7. CA-Markov proyecciones
"""

import ee
import os
import sys
import json
import time
import math
import numpy as np
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from gee_config import PERIODS, LULC_CLASSES, CARBON_POOLS, LANDTRENDR_PARAMS, N_CLASSES
from scripts.utils import get_study_area

import importlib
training_mod = importlib.import_module('scripts.02_training_samples')
classification_mod = importlib.import_module('scripts.03_classification')
change_mod = importlib.import_module('scripts.05_change_detection')
hotspot_mod = importlib.import_module('scripts.07_hotspot_analysis')
ecosystem_mod = importlib.import_module('scripts.08_ecosystem_services')
climate_mod = importlib.import_module('scripts.09_climate_analysis')
gwr_mod = importlib.import_module('scripts.10_gwr_drivers')
ca_markov_mod = importlib.import_module('scripts.11_ca_markov')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'phase3_stats')
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'phase3_analysis.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    log(f"  >> Guardado: {filename}")


def safe_getinfo(ee_obj, label=""):
    try:
        return ee_obj.getInfo()
    except Exception as e:
        log(f"  WARNING ({label}): {e}")
        return None


# ================================================================
# OPTIMIZED COMPOSITE - reduces memory footprint
# Wider temporal window for Choco bioregion (high cloud cover)
# ================================================================

def create_optimized_composite(start, end, region, year):
    """
    Composite optimizado: 10 bandas clave (6 spectral + 4 indices).
    Evita sobrecargar el computation graph.
    Uses wider temporal window for Choco bioregion cloud cover.
    """
    from gee_config import LANDSAT_BANDS
    from scripts.utils import mask_landsat_clouds, mask_sentinel2_clouds

    if year >= 2016:
        common = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterDate(start, end).filterBounds(region)
              .filter(ee.Filter.lt('CLOUD_COVER', 70))
              .map(mask_landsat_clouds))
        l8 = l8.select(list(LANDSAT_BANDS.values()), common)
        # Cast to generic Float to ensure homogeneous band types when merging
        l8 = l8.map(lambda img: img.toFloat())

        # Add Landsat 9 (available since Oct 2021)
        l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
              .filterDate(start, end).filterBounds(region)
              .filter(ee.Filter.lt('CLOUD_COVER', 70))
              .map(mask_landsat_clouds))
        l9 = l9.select(list(LANDSAT_BANDS.values()), common)
        l9 = l9.map(lambda img: img.toFloat())

        from gee_config import SENTINEL_BANDS
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(start, end).filterBounds(region)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
              .map(mask_sentinel2_clouds))
        s2 = s2.select(list(SENTINEL_BANDS.values()), common)
        # Cast to generic Float to match Landsat band types
        s2 = s2.map(lambda img: img.toFloat())

        merged = l8.merge(l9).merge(s2)
    else:
        common = list(LANDSAT_BANDS.values())
        merged = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterDate(start, end).filterBounds(region)
                  .filter(ee.Filter.lt('CLOUD_COVER', 70))
                  .map(mask_landsat_clouds))
        common_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        merged = merged.select(common, common_names)
        merged = merged.map(lambda img: img.toFloat())

    n_images = merged.size()

    # Median composite — clamp reflectance to [0, 1] to handle S2 saturated pixels
    composite = merged.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).median()
    composite = composite.clamp(0, 1)

    # Add key indices only (4 most important)
    nir = composite.select('nir')
    red = composite.select('red')
    green = composite.select('green')
    swir1 = composite.select('swir1')
    swir2 = composite.select('swir2')

    ndvi = nir.subtract(red).divide(nir.add(red).add(0.0001)).rename('NDVI')
    ndwi = green.subtract(nir).divide(green.add(nir).add(0.0001)).rename('NDWI')
    ndbi = swir1.subtract(nir).divide(swir1.add(nir).add(0.0001)).rename('NDBI')
    nbr = nir.subtract(swir2).divide(nir.add(swir2).add(0.0001)).rename('NBR')

    composite = composite.addBands([ndvi, ndwi, ndbi, nbr])

    # Terrain (2 bands)
    dem = ee.Image('USGS/SRTMGL1_003')
    composite = composite.addBands(dem.select('elevation').rename('elevation'))
    composite = composite.addBands(ee.Terrain.slope(dem).rename('slope'))

    composite = composite.clip(region)
    feature_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
                     'NDVI', 'NDWI', 'NDBI', 'NBR', 'elevation', 'slope']
    return composite, n_images, feature_bands


# ================================================================
# STAGE 1: CLASIFICACION (8 classes including Mangroves)
# ================================================================

def run_classification(region):
    log("=" * 60)
    log("STAGE 1: CLASIFICACION LULC (8 classes, Uraba Antioqueno)")
    log("=" * 60)

    n_classes = N_CLASSES  # 8 classes for Uraba

    results = {}
    all_metrics = {}
    all_importance = {}

    for period_key, period_info in PERIODS.items():
        year = period_info['map_year']
        t0 = time.time()
        log(f"\n  [{year}] {period_info['label']}")

        # Optimized composite (12 bands)
        log(f"    Composite...")
        composite, n_images, feature_bands = create_optimized_composite(
            period_info['start'], period_info['end'], region, year
        )
        n_img = safe_getinfo(n_images, f"n_img_{year}")
        log(f"    {n_img} images, {len(feature_bands)} features")

        # Training samples (reduced: 300/class)
        log(f"    Training samples (300/class, {n_classes} classes)...")
        reference = training_mod.get_reference_lulc(year, region)
        samples = training_mod.generate_stratified_samples(
            reference, region, n_per_class=300, seed=42 + year
        )
        training, validation = training_mod.split_train_validation(samples, 0.7, seed=42)

        # Extract spectral values
        log(f"    Extracting spectral values...")
        training_data = composite.select(feature_bands).sampleRegions(
            collection=training,
            properties=['lulc_reference'],
            scale=30,
            tileScale=4,
            geometries=False
        )
        validation_data = composite.select(feature_bands).sampleRegions(
            collection=validation,
            properties=['lulc_reference'],
            scale=30,
            tileScale=4,
            geometries=False
        )

        # RF with fewer trees (200)
        log(f"    Random Forest (200 trees)...")
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=200, minLeafPopulation=5,
            bagFraction=0.632, seed=42
        ).train(
            features=training_data,
            classProperty='lulc_reference',
            inputProperties=feature_bands
        )

        # Feature importance
        importance = ee.Dictionary(classifier.explain().get('importance'))
        importance_dict = safe_getinfo(importance, f"imp_{year}")
        if importance_dict:
            sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            log(f"    Top features: " + ", ".join(f"{k}:{v:.0f}" for k, v in sorted_imp[:5]))
            all_importance[period_key] = importance_dict

        # Classify
        log(f"    Classifying...")
        classified = composite.select(feature_bands).classify(classifier).rename('lulc')
        # Minimal smoothing to reduce salt-and-pepper noise (radius=1 preserves minority classes)
        classified = classified.focal_mode(1, 'square', 'pixels').rename('lulc')

        # Water mask enforcement
        jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(region)
        classified = classified.where(jrc.select('occurrence').gte(80), 5)

        # Mangrove mask enforcement using USGS Mangrove Forests (Giri et al. 2011)
        try:
            mf = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS')
            mf.size().getInfo()  # Eager check
            mangrove_mask = mf.mosaic().select('1').gte(1).clip(region)
            classified = classified.where(mangrove_mask, 8)
            log(f"    Mangrove mask applied from USGS Mangrove Forests")
        except Exception as e:
            log(f"    WARNING: Mangrove mask unavailable: {e}")

        # Accuracy
        log(f"    Accuracy assessment...")
        validated = validation_data.classify(classifier)
        error_matrix = validated.errorMatrix('lulc_reference', 'classification')
        oa_val = safe_getinfo(error_matrix.accuracy(), f"oa_{year}")
        kappa_val = safe_getinfo(error_matrix.kappa(), f"kappa_{year}")

        if oa_val:
            log(f"    OA: {oa_val:.4f} ({oa_val*100:.1f}%), Kappa: {kappa_val:.4f}")
        else:
            log(f"    OA/Kappa computation deferred")

        # Confusion matrix
        cm = safe_getinfo(error_matrix.array(), f"cm_{year}")
        if cm:
            log(f"    Confusion matrix retrieved ({len(cm)}x{len(cm[0])})")

        # Producer/User accuracies
        pa = safe_getinfo(error_matrix.producersAccuracy(), f"pa_{year}")
        ua = safe_getinfo(error_matrix.consumersAccuracy(), f"ua_{year}")

        # Area per class (at 100m scale to reduce memory)
        log(f"    Area per class...")
        class_areas = {}
        area_img = ee.Image.pixelArea().divide(10000)
        for cid in range(1, n_classes + 1):
            a = safe_getinfo(
                area_img.updateMask(classified.eq(cid)).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=region,
                    scale=100,
                    maxPixels=1e12,
                    tileScale=4,
                    bestEffort=True
                ),
                f"area_c{cid}_{year}"
            )
            ha = a.get('area', 0) if a else 0
            name = LULC_CLASSES.get(cid, {}).get('name', f'C{cid}')
            class_areas[cid] = {'name': name, 'area_ha': round(ha, 1)}
            log(f"      {name}: {ha:,.0f} ha")

        elapsed = time.time() - t0
        log(f"    Time: {elapsed:.0f}s")

        n_train = safe_getinfo(training_data.size(), f"n_train_{year}")
        n_val = safe_getinfo(validation_data.size(), f"n_val_{year}")

        results[period_key] = {
            'classified': classified,
            'composite': composite,
        }
        all_metrics[period_key] = {
            'year': year,
            'label': period_info['label'],
            'overall_accuracy': round(oa_val, 4) if oa_val else None,
            'kappa': round(kappa_val, 4) if kappa_val else None,
            'n_training': n_train,
            'n_validation': n_val,
            'n_images': n_img,
            'confusion_matrix': cm,
            'producers_accuracy': pa,
            'users_accuracy': ua,
            'class_areas_ha': {str(k): v for k, v in class_areas.items()},
        }

    # Summary
    log("\n  CLASSIFICATION SUMMARY:")
    log(f"  {'Period':<35} {'OA':>8} {'Kappa':>8}")
    log("  " + "-" * 55)
    for k, m in all_metrics.items():
        lab = m['label'][:33]
        oa = m['overall_accuracy']
        kp = m['kappa']
        if oa and kp:
            log(f"  {lab:<35} {oa:>7.1%} {kp:>8.4f}")
        else:
            log(f"  {lab:<35} {'N/A':>8} {'N/A':>8}")

    save_json(all_metrics, 'classification_metrics.json')
    save_json(all_importance, 'feature_importance.json')
    return results, all_metrics


# ================================================================
# STAGE 2: CHANGE DETECTION
# ================================================================

def run_change_detection(classified_maps, region):
    log("\n" + "=" * 60)
    log("STAGE 2: CHANGE DETECTION")
    log("=" * 60)

    n_classes = N_CLASSES
    periods_list = list(PERIODS.keys())
    transitions_cfg = [
        ('T1_T2', periods_list[0], periods_list[1]),
        ('T2_T3', periods_list[1], periods_list[2]),
        ('T3_T4', periods_list[2], periods_list[3]),
    ]

    all_trans = {}
    for trans_key, p_from, p_to in transitions_cfg:
        y_from = PERIODS[p_from]['map_year']
        y_to = PERIODS[p_to]['map_year']
        years_diff = y_to - y_from
        log(f"\n  {y_from} -> {y_to} ({years_diff} years)")

        lulc_from = classified_maps[p_from]['classified']
        lulc_to = classified_maps[p_to]['classified']

        transition_img, stats = change_mod.compute_transition_matrix(
            lulc_from, lulc_to, region, scale=100
        )
        stats_info = safe_getinfo(stats, f"trans_{trans_key}")

        matrix = {}
        if stats_info and 'groups' in stats_info:
            for g in stats_info['groups']:
                code = int(g['transition'])
                ha = round(g['sum'], 1)
                cf = code // 10
                ct = code % 10
                fn = LULC_CLASSES.get(cf, {}).get('name', f'C{cf}')
                tn = LULC_CLASSES.get(ct, {}).get('name', f'C{ct}')
                matrix[f"{cf}->{ct}"] = {
                    'from': fn, 'to': tn, 'area_ha': ha
                }

            # Top transitions (non-persistence)
            non_pers = [(k, v) for k, v in matrix.items()
                        if k.split('->')[0] != k.split('->')[1]]
            non_pers.sort(key=lambda x: abs(x[1]['area_ha']), reverse=True)
            log(f"    Major transitions:")
            for k, v in non_pers[:6]:
                log(f"      {v['from']} -> {v['to']}: {v['area_ha']:,.0f} ha")

        # Change rates (FAO Puyravaud)
        areas_from = {}
        areas_to = {}
        for k, v in matrix.items():
            cf, ct = k.split('->')
            cf, ct = int(cf), int(ct)
            areas_from[cf] = areas_from.get(cf, 0) + v['area_ha']
            areas_to[ct] = areas_to.get(ct, 0) + v['area_ha']

        rates = change_mod.compute_change_rates(areas_from, areas_to, years_diff)
        log(f"    Change rates:")
        for cid, r in rates.items():
            if abs(r['annual_rate_pct']) > 0.01:
                log(f"      {r['name']}: {r['annual_rate_pct']:+.3f}%/yr "
                    f"({r['net_change_ha']:+,.0f} ha)")

        all_trans[trans_key] = {
            'years': f"{y_from}-{y_to}",
            'years_between': years_diff,
            'transitions': matrix,
            'change_rates': {str(k): v for k, v in rates.items()},
        }

    # Hansen GFC
    log("\n  Hansen GFC v1.12...")
    hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12').clip(region)
    lossyear = hansen.select('lossyear')
    tc2000 = hansen.select('treecover2000')

    # Tree cover 2000 stats
    tc_stats = safe_getinfo(
        tc2000.reduceRegion(
            ee.Reducer.mean(), region, 100, maxPixels=1e12,
            tileScale=4, bestEffort=True
        ), "tc2000"
    )
    log(f"    Tree cover 2000: {tc_stats}")

    hansen_loss = {}
    for pk, pi in PERIODS.items():
        ys = int(pi['start'][:4]) - 2000
        ye = int(pi['end'][:4]) - 2000
        loss = lossyear.gte(ys).And(lossyear.lte(ye))
        loss_ha = loss.multiply(ee.Image.pixelArea()).divide(10000)
        total = safe_getinfo(
            loss_ha.reduceRegion(
                ee.Reducer.sum(), region, 100, maxPixels=1e12,
                tileScale=4, bestEffort=True
            ), f"hansen_{pk}"
        )
        ha = total.get('lossyear', 0) if total else 0
        hansen_loss[pk] = {
            'label': pi['label'],
            'loss_ha': round(ha, 1)
        }
        log(f"    {pi['label']}: {ha:,.0f} ha loss")

    all_trans['hansen_gfc'] = hansen_loss
    all_trans['hansen_treecover2000'] = tc_stats

    # LandTrendr
    log("\n  LandTrendr (2012-2024)...")
    lt_result = change_mod.run_landtrendr(region, 2012, 2024)
    lt_dist, rmse = change_mod.extract_disturbance_map(lt_result)
    log("  LandTrendr completed.")

    save_json(all_trans, 'change_detection_results.json')
    return all_trans


# ================================================================
# STAGE 3: ECOSYSTEM SERVICES
# ================================================================

def run_ecosystem_services(classified_maps, region):
    log("\n" + "=" * 60)
    log("STAGE 3: ECOSYSTEM SERVICES (Choco bioregion carbon values)")
    log("=" * 60)

    es = {}
    periods_list = list(PERIODS.keys())

    for pk, pi in PERIODS.items():
        year = pi['map_year']
        lulc = classified_maps[pk]['classified']
        log(f"\n  [{year}] {pi['label']}")

        # Carbon (IPCC Tier 2 - Choco values)
        carbon = ecosystem_mod.compute_carbon_storage(lulc, region)
        ct = safe_getinfo(
            carbon['c_total'].multiply(ee.Image.pixelArea().divide(10000)).reduceRegion(
                ee.Reducer.sum(), region, 100, maxPixels=1e12,
                tileScale=4, bestEffort=True
            ), f"carbon_{year}"
        )
        c_val = ct.get('c_total', 0) if ct else 0
        log(f"    Carbon: {c_val:,.0f} Mg C")

        # Water yield
        water = ecosystem_mod.compute_water_yield_proxy(lulc, region, year)
        wy = safe_getinfo(
            water['water_yield'].reduceRegion(
                ee.Reducer.mean(), region, 1000, maxPixels=1e12,
                tileScale=4, bestEffort=True
            ), f"water_{year}"
        )
        wy_val = wy.get('water_yield', 0) if wy else 0
        bf = safe_getinfo(
            water['baseflow'].reduceRegion(
                ee.Reducer.mean(), region, 1000, maxPixels=1e12,
                tileScale=4, bestEffort=True
            ), f"bf_{year}"
        )
        bf_val = bf.get('baseflow', 0) if bf else 0
        log(f"    Water yield: {wy_val:.1f} mm/yr, Baseflow: {bf_val:.1f} mm/yr")

        # Habitat quality
        hab = ecosystem_mod.compute_habitat_quality(lulc, region)
        hq = safe_getinfo(
            hab['habitat_quality'].reduceRegion(
                ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                region, 1000, maxPixels=1e12,
                tileScale=4, bestEffort=True
            ), f"hq_{year}"
        )
        hq_mean = hq.get('habitat_quality_mean', 0) if hq else 0
        hq_std = hq.get('habitat_quality_stdDev', 0) if hq else 0
        log(f"    Habitat quality: {hq_mean:.3f} +/- {hq_std:.3f}")

        es[pk] = {
            'year': year,
            'carbon_Mg_C': round(c_val, 0),
            'water_yield_mm': round(wy_val, 1),
            'baseflow_mm': round(bf_val, 1),
            'habitat_quality_mean': round(hq_mean, 4),
            'habitat_quality_std': round(hq_std, 4),
        }

    # Carbon change
    log("\n  Carbon change between periods:")
    for i in range(len(periods_list) - 1):
        pf, pt = periods_list[i], periods_list[i+1]
        yf = PERIODS[pf]['map_year']
        yt = PERIODS[pt]['map_year']
        c_change = ecosystem_mod.compute_carbon_change(
            classified_maps[pf]['classified'],
            classified_maps[pt]['classified'],
            region
        )
        net = safe_getinfo(
            c_change['net_change_Mg_C'], f"cdelta_{yf}_{yt}"
        )
        net_val = net.get('c_change', 0) if net else 0
        log(f"    {yf}->{yt}: {net_val:+,.0f} Mg C")
        es[f"carbon_change_{yf}_{yt}"] = {'net_Mg_C': round(net_val, 0)}

    save_json(es, 'ecosystem_services_results.json')
    return es


# ================================================================
# STAGE 4: CLIMATE ANALYSIS
# ================================================================

def run_climate_analysis(region):
    log("\n" + "=" * 60)
    log("STAGE 4: CLIMATE ANALYSIS (Choco: 2500-4000+ mm/yr)")
    log("=" * 60)

    clim = {}

    # Annual precipitation
    log("\n  Annual precipitation (CHIRPS)...")
    precip = {}
    for year in range(2012, 2025):
        p = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
             .filterDate(f'{year}-01-01', f'{year}-12-31')
             .filterBounds(region).sum().clip(region))
        s = safe_getinfo(
            p.reduceRegion(ee.Reducer.mean(), region, 5000,
                           maxPixels=1e12, tileScale=4, bestEffort=True),
            f"precip_{year}"
        )
        val = s.get('precipitation', 0) if s else 0
        precip[year] = round(val, 1)
        log(f"    {year}: {val:.0f} mm")
    clim['precipitation_annual_mm'] = precip

    # Annual LST
    log("\n  Land Surface Temperature (MODIS)...")
    lst = {}
    for year in range(2012, 2025):
        t = (ee.ImageCollection('MODIS/061/MOD11A2')
             .filterDate(f'{year}-01-01', f'{year}-12-31')
             .filterBounds(region).select('LST_Day_1km')
             .mean().multiply(0.02).subtract(273.15).clip(region))
        s = safe_getinfo(
            t.reduceRegion(ee.Reducer.mean(), region, 1000,
                           maxPixels=1e12, tileScale=4, bestEffort=True),
            f"lst_{year}"
        )
        val = s.get('LST_Day_1km', 0) if s else 0
        lst[year] = round(val, 2)
        log(f"    {year}: {val:.2f} C")
    clim['lst_annual_C'] = lst

    # Mann-Kendall trends
    log("\n  Mann-Kendall trends...")
    precip_vals = [precip[y] for y in range(2012, 2025)]
    lst_vals = [lst[y] for y in range(2012, 2025)]
    years_arr = list(range(2012, 2025))

    from scipy.stats import kendalltau
    tau_p, p_p = kendalltau(years_arr, precip_vals)
    tau_t, p_t = kendalltau(years_arr, lst_vals)
    log(f"    Precip trend: tau={tau_p:.3f}, p={p_p:.4f}")
    log(f"    LST trend: tau={tau_t:.3f}, p={p_t:.4f}")

    # Sen's slope
    sen_p = climate_mod.compute_sen_slope(np.array(precip_vals), np.array(years_arr, dtype=float))
    sen_t = climate_mod.compute_sen_slope(np.array(lst_vals), np.array(years_arr, dtype=float))
    log(f"    Precip Sen slope: {sen_p:.2f} mm/yr")
    log(f"    LST Sen slope: {sen_t:.4f} C/yr")

    clim['trends'] = {
        'precipitation': {
            'kendall_tau': round(tau_p, 4),
            'p_value': round(p_p, 6),
            'sen_slope_mm_yr': round(sen_p, 2),
            'significant': p_p < 0.05,
        },
        'lst': {
            'kendall_tau': round(tau_t, 4),
            'p_value': round(p_t, 6),
            'sen_slope_C_yr': round(sen_t, 4),
            'significant': p_t < 0.05,
        },
    }

    # SPI
    log("\n  SPI (key years)...")
    for year in [2013, 2016, 2020, 2024]:
        spi = climate_mod.compute_spi(region, year)
        s = safe_getinfo(
            spi.reduceRegion(
                ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                region, 5000, maxPixels=1e12, tileScale=4, bestEffort=True
            ), f"spi_{year}"
        )
        log(f"    SPI {year}: {s}")
        clim[f'spi_{year}'] = s

    # Drought frequency
    log("\n  Drought frequency (SPI < -1)...")
    drought = climate_mod.compute_drought_frequency(region, 2012, 2024, threshold=-1)
    df = safe_getinfo(
        drought.reduceRegion(
            ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            region, 5000, maxPixels=1e12, tileScale=4, bestEffort=True
        ), "drought_freq"
    )
    log(f"    Drought frequency: {df}")
    clim['drought_frequency'] = df

    save_json(clim, 'climate_analysis_results.json')
    return clim


# ================================================================
# STAGE 5: HOTSPOT ANALYSIS
# ================================================================

def run_hotspot_analysis(classified_maps, region):
    log("\n" + "=" * 60)
    log("STAGE 5: HOTSPOT ANALYSIS")
    log("=" * 60)

    hs = {}

    lulc_t1 = classified_maps['pre_acuerdo']['classified']
    lulc_t4 = classified_maps['post_acuerdo_2']['classified']
    # Forest includes dense (1), secondary (2), and mangroves (8)
    forest_t1 = lulc_t1.eq(1).Or(lulc_t1.eq(2)).Or(lulc_t1.eq(8))
    forest_t4 = lulc_t4.eq(1).Or(lulc_t4.eq(2)).Or(lulc_t4.eq(8))
    defor = forest_t1.And(forest_t4.Not()).rename('deforestation')

    combined = defor.addBands(ee.Image.pixelLonLat())

    log("  Sampling grid (1500 pts, 1km)...")
    sample_fc = combined.sample(
        region=region, scale=1000, numPixels=1500,
        seed=42, geometries=True
    )
    sample_info = safe_getinfo(sample_fc.limit(1500), "hs_sample")

    if sample_info and 'features' in sample_info:
        feats = sample_info['features']
        n = len(feats)
        log(f"  {n} points sampled")

        if n > 30:
            vals = np.array([f['properties'].get('deforestation', 0) for f in feats])
            coords = np.array([
                [f['properties'].get('longitude', 0),
                 f['properties'].get('latitude', 0)]
                for f in feats
            ])

            log("  Spatial weights (Queen)...")
            W = hotspot_mod.create_queen_weights(n, coords)

            log("  Moran's I...")
            morans = hotspot_mod.compute_morans_i(vals, W)
            log(f"    I={morans['I']:.4f}, z={morans['z_score']:.2f}, "
                f"p={morans['p_value']:.6f}")
            hs['morans_i'] = morans

            log("  Gi*...")
            gi = hotspot_mod.compute_getis_ord_gi_star(vals, W)
            cats = hotspot_mod.classify_hotspots(gi)

            counts = {}
            for name, val in [('hotspot_99', 3), ('hotspot_95', 2), ('hotspot_90', 1),
                              ('not_significant', 0), ('coldspot_90', -1),
                              ('coldspot_95', -2), ('coldspot_99', -3)]:
                counts[name] = int(np.sum(cats == val))

            log(f"    Hotspot 99%: {counts['hotspot_99']}")
            log(f"    Hotspot 95%: {counts['hotspot_95']}")
            log(f"    Not significant: {counts['not_significant']}")
            log(f"    Coldspot 99%: {counts['coldspot_99']}")
            hs['gi_star'] = counts
            hs['gi_z_stats'] = {
                'mean': round(float(np.mean(gi)), 4),
                'max': round(float(np.max(gi)), 4),
                'min': round(float(np.min(gi)), 4),
            }

            # Deforestation rate
            defor_rate = np.mean(vals)
            log(f"    Deforestation rate (mean): {defor_rate:.4f}")
            hs['deforestation_rate_mean'] = round(float(defor_rate), 4)

            # KDE
            defor_pts = coords[vals > 0]
            if len(defor_pts) > 10:
                density, xg, yg = hotspot_mod.compute_kernel_density(
                    defor_pts, grid_size=80, bandwidth=5000
                )
                hs['kde'] = {
                    'n_points': int(len(defor_pts)),
                    'max_density': round(float(np.max(density)), 6),
                }
                log(f"    KDE: {len(defor_pts)} defor points, max density: {np.max(density):.6f}")

    save_json(hs, 'hotspot_analysis_results.json')
    return hs


# ================================================================
# STAGE 6: GWR (11 driver variables for Uraba)
# ================================================================

def run_gwr_analysis(classified_maps, region):
    log("\n" + "=" * 60)
    log("STAGE 6: GWR DRIVERS (11 variables, Uraba)")
    log("=" * 60)

    gwr_results = {}

    lulc_2016 = classified_maps['transicion']['classified']
    lulc_2024 = classified_maps['post_acuerdo_2']['classified']
    # Forest includes dense (1), secondary (2), and mangroves (8)
    forest_16 = lulc_2016.eq(1).Or(lulc_2016.eq(2)).Or(lulc_2016.eq(8))
    forest_24 = lulc_2024.eq(1).Or(lulc_2024.eq(2)).Or(lulc_2024.eq(8))
    defor = forest_16.And(forest_24.Not()).rename('defor_rate')

    log("  Preparing driver variables...")
    drivers = gwr_mod.prepare_driver_variables_gee(region)

    log("  Sampling (1500 pts, 1km)...")
    sample = gwr_mod.extract_sample_data(
        drivers, defor, region, n_points=1500, scale=1000
    )
    sample_info = safe_getinfo(sample.limit(1500), "gwr_sample")

    var_names = [
        'elevation', 'slope', 'dist_rivers', 'dist_roads',
        'dist_urban', 'pop_density', 'precip_annual',
        'lst_mean', 'clay_content'
    ]

    if sample_info and 'features' in sample_info:
        feats = sample_info['features']
        log(f"  {len(feats)} features extracted")

        y_list, X_list, coords_list = [], [], []
        for f in feats:
            p = f['properties']
            y_val = p.get('defor_rate')
            if y_val is None:
                continue
            row = []
            ok = True
            for v in var_names:
                val = p.get(v)
                if val is None:
                    ok = False
                    break
                row.append(val)
            if ok and f.get('geometry'):
                y_list.append(y_val)
                X_list.append(row)
                coords_list.append(f['geometry']['coordinates'][:2])

        n_valid = len(y_list)
        log(f"  Valid data: {n_valid} points")

        if n_valid > 50:
            y = np.array(y_list)
            X = np.array(X_list)
            coords = np.array(coords_list)

            # Standardize
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1
            Xn = (X - X_mean) / X_std

            # VIF
            log("\n  VIF diagnostics...")
            vifs = gwr_mod.compute_vif(Xn)
            gwr_results['vif'] = {}
            for i, v in enumerate(var_names):
                log(f"    {v:<20} VIF={vifs[i]:.2f}")
                gwr_results['vif'][v] = vifs[i]

            # OLS
            log("\n  OLS global regression...")
            ols = gwr_mod.fit_ols(Xn, y)
            log(f"    R2={ols['r_squared']:.4f}, Adj R2={ols['adj_r_squared']:.4f}, "
                f"AIC={ols['aic']:.2f}")

            ols_names = ['intercept'] + var_names
            log(f"\n    {'Var':<20} {'Coef':>10} {'t':>8}")
            log("    " + "-" * 40)
            for i, nm in enumerate(ols_names):
                log(f"    {nm:<20} {ols['coefficients'][i]:>10.4f} "
                    f"{ols['t_statistics'][i]:>8.2f}")

            gwr_results['ols'] = {
                'r2': ols['r_squared'],
                'adj_r2': ols['adj_r_squared'],
                'aic': ols['aic'],
                'n': ols['n'],
                'coefficients': {ols_names[i]: round(ols['coefficients'][i], 6)
                                 for i in range(len(ols_names))},
                't_statistics': {ols_names[i]: round(ols['t_statistics'][i], 4)
                                 for i in range(len(ols_names))},
            }

            # GWR
            log("\n  Optimizing GWR bandwidth...")
            best_bw, best_aic = gwr_mod.optimize_bandwidth(
                Xn, y, coords, kernel='adaptive', n_steps=10
            )
            log(f"    Optimal: {best_bw} neighbors, AICc={best_aic:.2f}")

            log(f"  Fitting GWR (bw={best_bw})...")
            gwr = gwr_mod.compute_gwr(Xn, y, coords, bandwidth=best_bw)
            log(f"    Mean R2={gwr['mean_r2']:.4f}, Median R2={gwr['median_r2']:.4f}")
            log(f"    AIC={gwr['aic']:.2f}")

            summary = gwr_mod.summarize_gwr_results(gwr, var_names)
            log(f"\n    {'Var':<20} {'Mean':>8} {'Std':>8} {'%Pos':>6}")
            log("    " + "-" * 44)
            for nm, st in summary.items():
                log(f"    {nm:<20} {st['mean']:>8.4f} {st['std']:>8.4f} "
                    f"{st['pct_positive']:>5.1f}%")

            gwr_results['gwr'] = {
                'bandwidth': best_bw,
                'mean_r2': gwr['mean_r2'],
                'median_r2': gwr['median_r2'],
                'aic': gwr['aic'],
                'summary': summary,
            }

            comp = gwr_mod.compare_ols_gwr(ols, gwr)
            log(f"\n  OLS vs GWR: R2 improve={comp['r2_improvement']:.4f}, "
                f"AIC improve={comp['aic_improvement']:.2f}")
            gwr_results['comparison'] = comp

    save_json(gwr_results, 'gwr_drivers_results.json')
    return gwr_results


# ================================================================
# STAGE 7: CA-MARKOV (Uraba scenarios)
# ================================================================

def run_ca_markov(classified_maps, region):
    log("\n" + "=" * 60)
    log("STAGE 7: CA-MARKOV PROJECTIONS (Uraba scenarios)")
    log("=" * 60)

    ca = {}
    n_classes = N_CLASSES
    class_short = ['BDen', 'BSec', 'Past', 'Cult', 'Agua', 'Urb', 'Suel', 'Mang']

    # Extract LULC as arrays (reduced to 3000 pts at 1km to avoid memory limits)
    log("\n  Extracting LULC arrays (1km, 3000 pts)...")
    lulc_np = {}
    for pk in ['transicion', 'post_acuerdo_1', 'post_acuerdo_2']:
        year = PERIODS[pk]['map_year']
        lulc = classified_maps[pk]['classified']
        s = lulc.sample(region=region, scale=1000, numPixels=3000,
                        seed=42, geometries=False)
        info = safe_getinfo(s.limit(3000), f"lulc_arr_{year}")
        if info and 'features' in info:
            vals = np.array([f['properties'].get('lulc', 0) for f in info['features']])
            log(f"    {year}: {len(vals)} pts, classes: {np.unique(vals)}")
            lulc_np[pk] = vals

    if len(lulc_np) >= 2 and 'post_acuerdo_1' in lulc_np and 'post_acuerdo_2' in lulc_np:
        v2020 = lulc_np['post_acuerdo_1']
        v2024 = lulc_np['post_acuerdo_2']
        ml = min(len(v2020), len(v2024))
        v2020 = v2020[:ml]
        v2024 = v2024[:ml]

        # Transition probabilities 2020->2024
        log("\n  Transition matrix 2020->2024:")
        tp = ca_markov_mod.compute_transition_probabilities(v2020, v2024, n_classes=n_classes)
        header = "         " + "".join(f"{n:>7}" for n in class_short)
        log(f"  {header}")
        for i in range(n_classes):
            row = f"  {class_short[i]:<8}" + "".join(f"{tp[i,j]:>7.3f}" for j in range(n_classes))
            log(row)
        ca['transition_matrix'] = tp.tolist()

        # Current areas
        areas_2024 = np.array([np.sum(v2024 == c) for c in range(1, n_classes + 1)]).astype(float)
        total = areas_2024.sum()
        log(f"\n  Current areas (2024, {int(total)} pts):")
        for c in range(n_classes):
            pct = areas_2024[c] / total * 100
            log(f"    {class_short[c]}: {areas_2024[c]:.0f} ({pct:.1f}%)")
        ca['areas_2024'] = {class_short[c]: int(areas_2024[c]) for c in range(n_classes)}

        # Scenarios (adapted names for Uraba)
        scenarios = ca_markov_mod.create_scenario_matrices(tp)
        for sname, smat in scenarios.items():
            log(f"\n  === {sname} ===")
            for ty in [2030, 2040]:
                nsteps = max(1, (ty - 2024) // 4)
                proj = ca_markov_mod.project_markov(areas_2024, smat, nsteps)
                log(f"    {ty} (steps={nsteps}):")
                scenario_data = {}
                for c in range(n_classes):
                    pct = proj[c] / total * 100
                    chg = (proj[c] - areas_2024[c]) / max(areas_2024[c], 1) * 100
                    log(f"      {class_short[c]}: {pct:.1f}% ({chg:+.1f}%)")
                    scenario_data[class_short[c]] = {
                        'pct': round(pct, 2),
                        'change_pct': round(chg, 2),
                    }
                ca[f'{sname}_{ty}'] = scenario_data

        # Validation: 2016->2020 matrix to predict 2024
        if 'transicion' in lulc_np:
            log("\n  Validation: hindcast 2024 from 2020...")
            v2016 = lulc_np['transicion']
            ml2 = min(len(v2016), len(v2020))
            tp_val = ca_markov_mod.compute_transition_probabilities(
                v2016[:ml2], v2020[:ml2], n_classes=n_classes
            )
            areas_2020 = np.array([np.sum(v2020 == c) for c in range(1, n_classes + 1)]).astype(float)
            sim_2024 = ca_markov_mod.project_markov(areas_2020, tp_val, 1)

            log(f"    {'Class':<8} {'Obs':>8} {'Sim':>8} {'Diff':>8}")
            log("    " + "-" * 34)
            mae = 0
            validation = {}
            for c in range(n_classes):
                obs = areas_2024[c]
                sim = sim_2024[c]
                d = sim - obs
                mae += abs(d)
                log(f"    {class_short[c]:<8} {obs:>8.0f} {sim:>8.0f} {d:>+8.0f}")
                validation[class_short[c]] = {
                    'observed': int(obs), 'simulated': round(float(sim), 0)
                }

            rmae = mae / total * 100
            log(f"    Relative MAE: {rmae:.2f}%")
            ca['validation'] = {
                'method': 'Hindcast 2016->2020 matrix, predict 2024',
                'relative_MAE_pct': round(rmae, 2),
                'by_class': validation,
            }

    save_json(ca, 'ca_markov_results.json')
    return ca


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("FASE 3: ANALISIS ESTADISTICO - URABA ANTIOQUENO LULCC")
    log(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 70)

    region = get_study_area()
    area = safe_getinfo(region.area().divide(1e6), "area")
    log(f"Study area: {area:,.0f} km2")

    # Stage 1
    classified, metrics = run_classification(region)

    # Stage 2
    changes = run_change_detection(classified, region)

    # Stage 3
    ecosystem = run_ecosystem_services(classified, region)

    # Stage 4
    climate = run_climate_analysis(region)

    # Stage 5
    hotspots = run_hotspot_analysis(classified, region)

    # Stage 6
    gwr = run_gwr_analysis(classified, region)

    # Stage 7
    ca = run_ca_markov(classified, region)

    # Summary
    elapsed = time.time() - t0
    log("\n" + "=" * 70)
    log(f"COMPLETED in {elapsed/60:.1f} minutes")
    log("=" * 70)

    files = [f for f in sorted(os.listdir(OUTPUT_DIR)) if f.endswith('.json')]
    log(f"\nOutput files ({len(files)}):")
    for f in files:
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        log(f"  {f} ({sz:,} bytes)")

    summary = {
        'date': datetime.now().isoformat(),
        'time_min': round(elapsed / 60, 1),
        'area_km2': area,
        'stages': 7,
        'region': 'Uraba Antioqueno',
        'n_classes': N_CLASSES,
        'classification': {
            k: {'OA': v['overall_accuracy'], 'kappa': v['kappa']}
            for k, v in metrics.items()
        },
        'files': files,
    }
    save_json(summary, 'analysis_summary.json')
    log("\nFase 3 completada.")


if __name__ == '__main__':
    main()
