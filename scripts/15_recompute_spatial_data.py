#!/usr/bin/env python3
"""
Phase 2: Recompute Spatial Data for Cartographic Maps.

Re-runs Gi* hotspot analysis and GWR to save intermediate spatial data
(coordinates + values) needed for cartographic rendering.

Outputs (in data/map_exports/):
  - hotspot_coordinates.npy     (n, 2) [lon, lat]
  - hotspot_zscores.npy         (n,) Z-scores from Gi*
  - gwr_coordinates.npy         (1470, 2) [lon, lat]
  - gwr_local_betas.npy         (1470, 9) local coefficients
  - gwr_local_r2.npy            (1470,) local R-squared
  - gwr_variable_names.json     variable name list

Usage:
    python scripts/15_recompute_spatial_data.py
"""

import os
import sys
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ee
from gee_config import STUDY_AREA_BBOX, COLLECTIONS

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'map_exports')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BBOX = [-77.2, 7.0, -75.8, 8.9]
REGION = ee.Geometry.Rectangle(BBOX)


# ============================================================
# HOTSPOT SPATIAL DATA (Gi*)
# ============================================================

def regenerate_hotspot_spatial_data(output_dir):
    """
    Re-execute Getis-Ord Gi* analysis and save coordinates + Z-scores.
    Uses deforestation rate on a 1x1 km grid within the study area.
    """
    print("\n--- Regenerating Hotspot (Gi*) spatial data ---")

    # Step 1: Compute deforestation rate grid from Hansen GFC
    hansen = ee.Image(COLLECTIONS['hansen'])
    treecover = hansen.select('treecover2000')
    loss = hansen.select('loss')

    # Create 1km grid by sampling
    grid_points = ee.FeatureCollection.randomPoints(
        region=REGION, points=1500, seed=42
    )

    # Compute deforestation rate at each point (Hansen loss fraction in 5km radius)
    def compute_defor_rate(point):
        buffer = point.geometry().buffer(2500)  # 2.5 km radius ~ 1km grid cell
        mean_loss = loss.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=30,
            maxPixels=1e6,
            bestEffort=True
        ).get('loss')
        mean_tc = treecover.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=30,
            maxPixels=1e6,
            bestEffort=True
        ).get('treecover2000')
        return point.set({
            'defor_rate': ee.Algorithms.If(mean_loss, mean_loss, 0),
            'treecover': ee.Algorithms.If(mean_tc, mean_tc, 0)
        })

    print("  Computing deforestation rates at grid points...")
    grid_with_rates = grid_points.map(compute_defor_rate)

    # Extract data
    print("  Extracting grid data from GEE...")
    data = grid_with_rates.getInfo()

    coords = []
    defor_rates = []
    for feat in data['features']:
        geom = feat['geometry']['coordinates']
        props = feat['properties']
        coords.append([geom[0], geom[1]])
        defor_rates.append(props.get('defor_rate', 0) or 0)

    coords = np.array(coords)
    defor_rates = np.array(defor_rates)

    # Step 2: Compute Gi* Z-scores locally
    print("  Computing Getis-Ord Gi* Z-scores...")
    n = len(defor_rates)
    z_scores = np.zeros(n)

    # Distance-based weights (50 km threshold)
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(coords, coords, metric='euclidean')
    threshold_deg = 0.45  # ~50 km at these latitudes

    global_mean = np.mean(defor_rates)
    global_std = np.std(defor_rates)

    if global_std > 0:
        for i in range(n):
            # Neighbors within threshold
            neighbors = dist_matrix[i] < threshold_deg
            neighbors[i] = False  # exclude self
            n_neighbors = np.sum(neighbors)

            if n_neighbors > 0:
                w_sum = np.sum(defor_rates[neighbors])
                expected = global_mean * n_neighbors
                # Gi* statistic
                numerator = w_sum - expected
                s = np.sqrt(
                    (np.sum(defor_rates ** 2) / n - global_mean ** 2) *
                    (n * n_neighbors - n_neighbors ** 2) / (n - 1)
                )
                if s > 0:
                    z_scores[i] = numerator / s

    # Save outputs
    np.save(os.path.join(output_dir, 'hotspot_coordinates.npy'), coords)
    np.save(os.path.join(output_dir, 'hotspot_zscores.npy'), z_scores)

    print(f"  Saved {n} hotspot points")
    print(f"  Z-score range: [{z_scores.min():.2f}, {z_scores.max():.2f}]")
    print(f"  Hot (z>2.576): {np.sum(z_scores > 2.576)}")
    print(f"  Cold (z<-2.576): {np.sum(z_scores < -2.576)}")

    return coords, z_scores


# ============================================================
# GWR SPATIAL DATA
# ============================================================

def regenerate_gwr_spatial_data(output_dir):
    """
    Re-execute GWR analysis and save coordinates + local coefficients.
    Samples driver variables from GEE and fits GWR locally.
    """
    print("\n--- Regenerating GWR spatial data ---")

    # Step 1: Prepare driver variables in GEE
    print("  Preparing driver variables...")

    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.select('elevation')
    slope = ee.Terrain.slope(dem)

    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    water_occurrence = jrc.select('occurrence').gt(50)
    dist_rivers = water_occurrence.selfMask().fastDistanceTransform(256).sqrt() \
        .multiply(30).rename('dist_rivers')

    ghsl = ee.Image('JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020')
    built_up = ghsl.select('smod_code').gte(13)
    dist_roads = built_up.selfMask().fastDistanceTransform(256).sqrt() \
        .multiply(1000).rename('dist_roads')

    urban = ghsl.select('smod_code').gte(20)
    dist_urban = urban.selfMask().fastDistanceTransform(256).sqrt() \
        .multiply(1000).rename('dist_urban')

    pop = ee.ImageCollection('WorldPop/GP/100m/pop') \
        .filterDate('2020-01-01', '2020-12-31') \
        .filter(ee.Filter.eq('country', 'COL')) \
        .first().rename('pop_density')

    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate('2012-01-01', '2024-12-31') \
        .select('precipitation')
    precip = chirps.mean().multiply(365).rename('precip')

    modis_lst = ee.ImageCollection('MODIS/061/MOD11A2') \
        .filterDate('2012-01-01', '2024-12-31') \
        .select('LST_Day_1km')
    lst = modis_lst.mean().multiply(0.02).subtract(273.15).rename('lst')

    # Dependent variable: deforestation rate from Hansen
    hansen = ee.Image(COLLECTIONS['hansen'])
    defor = hansen.select('loss').rename('defor_rate')

    # Stack all variables
    stack = elevation.addBands([
        slope, dist_rivers, dist_roads, dist_urban,
        pop, precip, lst, defor
    ])

    # Step 2: Sample 1500 points
    print("  Sampling points from GEE...")
    samples = stack.sample(
        region=REGION,
        scale=1000,
        numPixels=1500,
        seed=42,
        geometries=True
    )

    data = samples.getInfo()
    print(f"  Sampled {len(data['features'])} points")

    # Parse into arrays
    variable_names = ['elevation', 'slope', 'dist_rivers', 'dist_roads',
                      'dist_urban', 'pop_density', 'precip', 'lst']

    coords_list = []
    X_list = []
    y_list = []

    for feat in data['features']:
        geom = feat['geometry']['coordinates']
        props = feat['properties']

        # Skip if any value is None
        vals = [props.get(v) for v in variable_names]
        defor_val = props.get('defor_rate')
        if any(v is None for v in vals) or defor_val is None:
            continue

        coords_list.append([geom[0], geom[1]])
        X_list.append(vals)
        y_list.append(defor_val)

    coords = np.array(coords_list)
    X = np.array(X_list)
    y = np.array(y_list)

    print(f"  Valid points: {len(y)}")

    # Standardize X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    # Step 3: Fit GWR locally
    print("  Fitting GWR (adaptive bisquare kernel, k=11)...")
    n = len(y)
    k = min(11, n - 1)  # adaptive bandwidth
    n_vars = X_norm.shape[1] + 1  # +1 for intercept

    local_betas = np.zeros((n, n_vars))
    local_r2 = np.zeros(n)

    from scipy.spatial.distance import cdist
    dist_matrix = cdist(coords, coords)

    for i in range(n):
        # Find k nearest neighbors
        dists = dist_matrix[i]
        sorted_idx = np.argsort(dists)
        neighbor_idx = sorted_idx[:k + 1]  # include self
        bandwidth = dists[sorted_idx[k]]

        if bandwidth == 0:
            bandwidth = 1e-6

        # Bisquare weights
        u = dists[neighbor_idx] / bandwidth
        weights = np.where(u < 1, (1 - u ** 2) ** 2, 0)

        # Weighted least squares
        X_local = np.column_stack([np.ones(len(neighbor_idx)), X_norm[neighbor_idx]])
        y_local = y[neighbor_idx]
        W = np.diag(weights)

        try:
            XtWX = X_local.T @ W @ X_local
            XtWy = X_local.T @ W @ y_local
            betas = np.linalg.solve(XtWX, XtWy)
            local_betas[i] = betas

            y_pred = X_local @ betas
            ss_res = np.sum(weights * (y_local - y_pred) ** 2)
            ss_tot = np.sum(weights * (y_local - np.average(y_local, weights=weights)) ** 2)
            if ss_tot > 0:
                local_r2[i] = 1 - ss_res / ss_tot
        except np.linalg.LinAlgError:
            pass

    # Save outputs
    np.save(os.path.join(output_dir, 'gwr_coordinates.npy'), coords)
    np.save(os.path.join(output_dir, 'gwr_local_betas.npy'), local_betas)
    np.save(os.path.join(output_dir, 'gwr_local_r2.npy'), local_r2)

    var_names_full = ['intercept'] + variable_names
    with open(os.path.join(output_dir, 'gwr_variable_names.json'), 'w') as f:
        json.dump(var_names_full, f)

    print(f"  Saved {n} GWR points")
    print(f"  Mean local R2: {local_r2.mean():.3f}")
    print(f"  Median local R2: {np.median(local_r2):.3f}")

    return coords, local_betas, local_r2


# ============================================================
# MAIN
# ============================================================

def main():
    from datetime import datetime
    print("=" * 60)
    print("PHASE 2: RECOMPUTE SPATIAL DATA (Gi* + GWR)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    t0 = time.time()

    # Hotspot analysis
    regenerate_hotspot_spatial_data(OUTPUT_DIR)

    # GWR analysis
    regenerate_gwr_spatial_data(OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"SPATIAL DATA RECOMPUTE COMPLETE in {elapsed/60:.1f} minutes")
    print("=" * 60)


if __name__ == '__main__':
    main()
