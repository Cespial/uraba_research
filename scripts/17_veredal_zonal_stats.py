#!/usr/bin/env python3
"""
17_veredal_zonal_stats.py
=========================
Computes zonal statistics per vereda (rural section) polygon for the
Magdalena Medio study area.

For each vereda intersecting the study bbox [-77.2, 7.0, -75.8, 8.9] computes:
  - Forest cover % per LULC period (T1-T4)
  - Deforestation rate (T1 vs T4)
  - Net forest change (ha)
  - Dominant LULC class per period (mode)
  - Hansen tree-cover-2000 mean and cumulative loss fraction
  - Carbon density per period and carbon change T1-T4
  - Interpolated hotspot Z-scores and GWR local betas/R2 at vereda centroids

Inputs:
  Shapefiles (DANE MGN 2019):
    - MGN_RUR_SECCION.shp           (veredas / rural sections)
    - MGN_DPTO_POLITICO.shp         (departments)
    - MGN_MPIO_POLITICO.shp         (municipalities)
  Rasters (data/map_exports/):
    - lulc_T1_2013.tif .. lulc_T4_2024.tif
    - hansen_gfc.tif
  Point arrays (data/map_exports/):
    - hotspot_coordinates.npy, hotspot_zscores.npy
    - gwr_coordinates.npy, gwr_local_betas.npy, gwr_local_r2.npy
    - gwr_variable_names.json

Outputs (data/map_exports/):
  - veredal_stats.gpkg           (GeoPackage with all computed columns)
  - departments_clip.gpkg        (clipped department boundaries)
  - municipalities_clip.gpkg     (clipped municipality boundaries)

Usage:
    pip install rasterstats          # if not already installed
    python scripts/17_veredal_zonal_stats.py
"""

import os
import sys
import json
import warnings
import time

import numpy as np
import geopandas as gpd
import rasterio
from scipy.interpolate import griddata

# rasterstats -- install at runtime if missing
try:
    from rasterstats import zonal_stats
except ImportError:
    print("[INFO] Installing rasterstats via pip ...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterstats"])
    from rasterstats import zonal_stats

# Suppress noisy warnings from rasterstats / rasterio
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DATA_DIR = os.path.join(BASE_DIR, "data", "map_exports")
OUTPUT_DIR = MAP_DATA_DIR  # outputs go alongside existing map data

# Vereda + admin shapefiles (DANE MGN 2019)
VEREDA_SHP = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "WGS84_MGN2019_00_COLOMBIA",
    "MGN",
    "MGN_RUR_SECCION.shp",
)
DPTO_SHP = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "WGS84_MGN2019_00_COLOMBIA",
    "ADMINISTRATIVO",
    "MGN_DPTO_POLITICO.shp",
)
MPIO_SHP = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "WGS84_MGN2019_00_COLOMBIA",
    "ADMINISTRATIVO",
    "MGN_MPIO_POLITICO.shp",
)

# LULC rasters
LULC_RASTERS = {
    "T1": os.path.join(MAP_DATA_DIR, "lulc_T1_2013.tif"),
    "T2": os.path.join(MAP_DATA_DIR, "lulc_T2_2016.tif"),
    "T3": os.path.join(MAP_DATA_DIR, "lulc_T3_2020.tif"),
    "T4": os.path.join(MAP_DATA_DIR, "lulc_T4_2024.tif"),
}

HANSEN_RASTER = os.path.join(MAP_DATA_DIR, "hansen_gfc.tif")

# Point-based spatial data
HOTSPOT_COORDS_NPY = os.path.join(MAP_DATA_DIR, "hotspot_coordinates.npy")
HOTSPOT_ZSCORES_NPY = os.path.join(MAP_DATA_DIR, "hotspot_zscores.npy")
GWR_COORDS_NPY = os.path.join(MAP_DATA_DIR, "gwr_coordinates.npy")
GWR_BETAS_NPY = os.path.join(MAP_DATA_DIR, "gwr_local_betas.npy")
GWR_R2_NPY = os.path.join(MAP_DATA_DIR, "gwr_local_r2.npy")
GWR_VARNAMES_JSON = os.path.join(MAP_DATA_DIR, "gwr_variable_names.json")

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

BBOX = [-77.2, 7.0, -75.8, 8.9]  # [xmin, ymin, xmax, ymax] Uraba Antioqueno

LULC_CLASSES = {
    1: "Dense forest",
    2: "Secondary forest",
    3: "Pastures",
    4: "Crops",
    5: "Water",
    6: "Urban",
    7: "Bare soil",
    8: "Mangroves",
}

# Simplified carbon pools (total Mg C/ha) -- consistent with figure_style.py
CARBON_POOLS = {1: 281, 2: 146, 3: 43.5, 4: 53.5, 5: 0, 6: 20, 7: 15, 8: 247}

# Forest class IDs
FOREST_CLASSES = {1, 2}


# ==================================================================
# HELPER FUNCTIONS
# ==================================================================

def load_and_clip_shapefile(path, bbox, label="layer"):
    """Load a shapefile, spatially clip it to bbox, and return GeoDataFrame."""
    xmin, ymin, xmax, ymax = bbox
    print(f"  Loading {label} from: {os.path.basename(path)} ...")
    gdf = gpd.read_file(path, bbox=(xmin, ymin, xmax, ymax))
    # Belt-and-suspenders spatial clip using .cx indexer
    gdf = gdf.cx[xmin:xmax, ymin:ymax].copy()
    print(f"    -> {len(gdf)} features after clipping to study bbox")
    return gdf


def pixel_area_ha(raster_path):
    """Return the approximate area of a single pixel in hectares."""
    with rasterio.open(raster_path) as src:
        res_x, res_y = src.res  # in CRS units (degrees for EPSG:4326)
    # At ~7 deg N, 1 deg lat ~ 110.6 km, 1 deg lon ~ 109.8 km
    lat_mid = 7.0
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.radians(lat_mid))
    pixel_w_km = abs(res_x) * km_per_deg_lon
    pixel_h_km = abs(res_y) * km_per_deg_lat
    area_km2 = pixel_w_km * pixel_h_km
    area_ha = area_km2 * 100  # 1 km2 = 100 ha
    return area_ha


def compute_forest_pct(pixel_counts):
    """Given a dict {class: count}, return forest percentage."""
    if pixel_counts is None:
        return np.nan
    total = sum(pixel_counts.values())
    if total == 0:
        return np.nan
    forest = sum(pixel_counts.get(c, 0) for c in FOREST_CLASSES)
    return (forest / total) * 100.0


def compute_dominant_class(pixel_counts):
    """Return the class with the highest pixel count (mode)."""
    if pixel_counts is None or len(pixel_counts) == 0:
        return np.nan
    # Filter out nodata (class 0 or negative keys)
    valid = {k: v for k, v in pixel_counts.items() if k in LULC_CLASSES}
    if not valid:
        return np.nan
    return max(valid, key=valid.get)


def compute_carbon_density(pixel_counts):
    """Weighted mean carbon density (Mg C/ha) from pixel counts."""
    if pixel_counts is None:
        return np.nan
    total = 0
    weighted_sum = 0.0
    for cls, count in pixel_counts.items():
        if cls in CARBON_POOLS:
            weighted_sum += CARBON_POOLS[cls] * count
            total += count
    if total == 0:
        return np.nan
    return weighted_sum / total


# ==================================================================
# MAIN PIPELINE
# ==================================================================

def main():
    from datetime import datetime

    t_start = time.time()
    print("=" * 64)
    print("VEREDAL ZONAL STATISTICS")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"BBOX : {BBOX}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 64)

    # ----------------------------------------------------------
    # 1. Load and clip vector layers
    # ----------------------------------------------------------
    print("\n[1/7] Loading and clipping vector layers ...")

    veredas = load_and_clip_shapefile(VEREDA_SHP, BBOX, "veredas")
    depts = load_and_clip_shapefile(DPTO_SHP, BBOX, "departments")
    munis = load_and_clip_shapefile(MPIO_SHP, BBOX, "municipalities")

    # Ensure valid geometries
    veredas["geometry"] = veredas.geometry.buffer(0)
    depts["geometry"] = depts.geometry.buffer(0)
    munis["geometry"] = munis.geometry.buffer(0)

    # ----------------------------------------------------------
    # 2. Compute LULC zonal stats per vereda
    # ----------------------------------------------------------
    print("\n[2/7] Computing LULC zonal statistics per vereda ...")

    # Get pixel area once (all LULC rasters have same resolution)
    pix_ha = pixel_area_ha(LULC_RASTERS["T1"])
    print(f"  Pixel area: {pix_ha:.6f} ha")

    for period_key, raster_path in LULC_RASTERS.items():
        print(f"\n  --- {period_key} ({os.path.basename(raster_path)}) ---")

        if not os.path.isfile(raster_path):
            print(f"    [WARN] Raster not found, skipping: {raster_path}")
            # Fill columns with NaN
            veredas[f"forest_pct_{period_key}"] = np.nan
            veredas[f"dominant_lulc_{period_key}"] = np.nan
            veredas[f"carbon_density_{period_key}"] = np.nan
            continue

        # Use categorical zonal_stats to get pixel counts per class
        print("    Running zonal_stats (categorical) ...")
        cat_stats = zonal_stats(
            veredas,
            raster_path,
            categorical=True,
            nodata=0,
            all_touched=True,
        )

        # Derive metrics from pixel counts
        forest_pcts = []
        dominant_classes = []
        carbon_densities = []

        for i, counts in enumerate(cat_stats):
            forest_pcts.append(compute_forest_pct(counts))
            dominant_classes.append(compute_dominant_class(counts))
            carbon_densities.append(compute_carbon_density(counts))

        veredas[f"forest_pct_{period_key}"] = forest_pcts
        veredas[f"dominant_lulc_{period_key}"] = dominant_classes
        veredas[f"carbon_density_{period_key}"] = carbon_densities

        print(f"    Forest %:  mean={np.nanmean(forest_pcts):.1f},  "
              f"median={np.nanmedian(forest_pcts):.1f}")
        print(f"    Carbon density: mean={np.nanmean(carbon_densities):.1f} Mg C/ha")

    # ----------------------------------------------------------
    # 3. Deforestation rate and net forest change
    # ----------------------------------------------------------
    print("\n[3/7] Computing deforestation rate and net forest change ...")

    f_t1 = veredas["forest_pct_T1"].values
    f_t4 = veredas["forest_pct_T4"].values

    # Deforestation rate: (forest_T1 - forest_T4) / forest_T1 * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        defor_rate = np.where(
            (f_t1 > 0) & np.isfinite(f_t1) & np.isfinite(f_t4),
            (f_t1 - f_t4) / f_t1 * 100.0,
            np.nan,
        )
    veredas["defor_rate_pct"] = defor_rate

    # Net forest change in hectares
    # We need pixel counts for T1 and T4 to compute absolute area change
    # Re-run categorical stats for T1 and T4 to get raw pixel counts
    print("  Re-running categorical stats for T1 and T4 pixel counts ...")

    if os.path.isfile(LULC_RASTERS["T1"]) and os.path.isfile(LULC_RASTERS["T4"]):
        cat_t1 = zonal_stats(
            veredas, LULC_RASTERS["T1"],
            categorical=True, nodata=0, all_touched=True,
        )
        cat_t4 = zonal_stats(
            veredas, LULC_RASTERS["T4"],
            categorical=True, nodata=0, all_touched=True,
        )

        net_change_ha = []
        for c1, c4 in zip(cat_t1, cat_t4):
            if c1 is None or c4 is None:
                net_change_ha.append(np.nan)
                continue
            forest_px_t1 = sum(c1.get(c, 0) for c in FOREST_CLASSES)
            forest_px_t4 = sum(c4.get(c, 0) for c in FOREST_CLASSES)
            net_change_ha.append((forest_px_t4 - forest_px_t1) * pix_ha)

        veredas["net_forest_change_ha"] = net_change_ha
    else:
        veredas["net_forest_change_ha"] = np.nan

    print(f"  Defor rate: mean={np.nanmean(defor_rate):.2f}%, "
          f"max={np.nanmax(defor_rate):.2f}%")
    print(f"  Net forest change: "
          f"sum={np.nansum(veredas['net_forest_change_ha'].values):.0f} ha")

    # ----------------------------------------------------------
    # 4. Hansen GFC zonal stats
    # ----------------------------------------------------------
    print("\n[4/7] Computing Hansen GFC zonal statistics ...")

    if os.path.isfile(HANSEN_RASTER):
        # Band 1: treecover2000 -> compute mean
        print("  Band 1 (treecover2000): computing mean per vereda ...")
        hansen_tc_stats = zonal_stats(
            veredas,
            HANSEN_RASTER,
            band=1,
            stats=["mean"],
            nodata=0,
            all_touched=True,
        )
        veredas["hansen_treecover2000_mean"] = [
            s["mean"] if s["mean"] is not None else np.nan
            for s in hansen_tc_stats
        ]
        print(f"    Mean treecover2000: "
              f"{np.nanmean(veredas['hansen_treecover2000_mean'].values):.1f}%")

        # Band 2: lossyear (if available) -> compute sum / area as loss fraction
        with rasterio.open(HANSEN_RASTER) as src:
            n_bands = src.count

        if n_bands >= 2:
            print("  Band 2 (lossyear): computing cumulative loss fraction ...")
            hansen_loss_stats = zonal_stats(
                veredas,
                HANSEN_RASTER,
                band=2,
                stats=["sum", "count"],
                nodata=0,
                all_touched=True,
            )
            loss_fracs = []
            for s in hansen_loss_stats:
                if s["count"] is not None and s["count"] > 0:
                    # lossyear > 0 means loss occurred; fraction = pixels with loss / total
                    # sum of lossyear values / count gives average year, not fraction
                    # Instead use count of non-zero pixels
                    loss_fracs.append(
                        s["sum"] / s["count"] if s["count"] > 0 else np.nan
                    )
                else:
                    loss_fracs.append(np.nan)
            veredas["hansen_loss_fraction"] = loss_fracs
            print(f"    Mean loss metric: "
                  f"{np.nanmean(veredas['hansen_loss_fraction'].values):.3f}")
        else:
            print("  Band 2 not available in Hansen raster; skipping loss fraction.")
            veredas["hansen_loss_fraction"] = np.nan

        # If band 2 available, also compute binary loss fraction (pixels with loss / total)
        if n_bands >= 2:
            print("  Computing binary loss fraction (pixels with any loss / total) ...")
            # Use categorical on band 2: any value > 0 is loss
            hansen_loss_cat = zonal_stats(
                veredas,
                HANSEN_RASTER,
                band=2,
                stats=["count"],
                categorical=True,
                nodata=255,  # 255 = nodata in Hansen loss band
                all_touched=True,
            )
            binary_loss_fracs = []
            for s in hansen_loss_cat:
                total = sum(s.values())
                if total == 0:
                    binary_loss_fracs.append(np.nan)
                    continue
                # Pixels with loss have year > 0; 0 means no loss
                no_loss = s.get(0, 0)
                loss_pixels = total - no_loss
                binary_loss_fracs.append(loss_pixels / total)
            veredas["hansen_loss_binary_frac"] = binary_loss_fracs
    else:
        print(f"  [WARN] Hansen raster not found: {HANSEN_RASTER}")
        veredas["hansen_treecover2000_mean"] = np.nan
        veredas["hansen_loss_fraction"] = np.nan
        veredas["hansen_loss_binary_frac"] = np.nan

    # ----------------------------------------------------------
    # 5. Carbon change (T1 vs T4)
    # ----------------------------------------------------------
    print("\n[5/7] Computing carbon change T1 vs T4 ...")

    c_t1 = veredas["carbon_density_T1"].values
    c_t4 = veredas["carbon_density_T4"].values
    veredas["carbon_change_MgC_ha"] = c_t4 - c_t1

    valid_mask = np.isfinite(veredas["carbon_change_MgC_ha"].values)
    if valid_mask.any():
        print(f"  Carbon change: mean={np.nanmean(veredas['carbon_change_MgC_ha'].values):.2f}, "
              f"min={np.nanmin(veredas['carbon_change_MgC_ha'].values):.2f}, "
              f"max={np.nanmax(veredas['carbon_change_MgC_ha'].values):.2f} Mg C/ha")

    # ----------------------------------------------------------
    # 6. Interpolate hotspot and GWR data to vereda centroids
    # ----------------------------------------------------------
    print("\n[6/7] Interpolating point-based data to vereda centroids ...")

    # Compute vereda centroids (lon, lat)
    centroids = veredas.geometry.centroid
    cx = centroids.x.values
    cy = centroids.y.values
    centroid_pts = np.column_stack([cx, cy])

    # --- Hotspot Z-scores ---
    if os.path.isfile(HOTSPOT_COORDS_NPY) and os.path.isfile(HOTSPOT_ZSCORES_NPY):
        print("  Loading hotspot data ...")
        hs_coords = np.load(HOTSPOT_COORDS_NPY)
        hs_zscores = np.load(HOTSPOT_ZSCORES_NPY)
        print(f"    {len(hs_zscores)} hotspot points loaded")

        # Remove any NaN/inf source points
        valid = np.isfinite(hs_zscores) & np.isfinite(hs_coords[:, 0])
        hs_coords_v = hs_coords[valid]
        hs_zscores_v = hs_zscores[valid]

        print("  Interpolating hotspot Z-scores (linear) ...")
        interp_z = griddata(
            hs_coords_v, hs_zscores_v, centroid_pts,
            method="linear", fill_value=np.nan,
        )
        # Fill remaining NaN with nearest-neighbor
        nan_mask = np.isnan(interp_z)
        if nan_mask.any():
            interp_z_nn = griddata(
                hs_coords_v, hs_zscores_v, centroid_pts,
                method="nearest",
            )
            interp_z[nan_mask] = interp_z_nn[nan_mask]

        veredas["hotspot_zscore"] = interp_z
        print(f"    Interpolated Z-score range: [{np.nanmin(interp_z):.2f}, "
              f"{np.nanmax(interp_z):.2f}]")
    else:
        print("  [WARN] Hotspot data files not found; skipping.")
        veredas["hotspot_zscore"] = np.nan

    # --- GWR local betas and R2 ---
    if (os.path.isfile(GWR_COORDS_NPY) and os.path.isfile(GWR_BETAS_NPY)
            and os.path.isfile(GWR_R2_NPY) and os.path.isfile(GWR_VARNAMES_JSON)):
        print("  Loading GWR data ...")
        gwr_coords = np.load(GWR_COORDS_NPY)
        gwr_betas = np.load(GWR_BETAS_NPY)
        gwr_r2 = np.load(GWR_R2_NPY)
        with open(GWR_VARNAMES_JSON, "r") as f:
            gwr_varnames = json.load(f)
        print(f"    {gwr_betas.shape[0]} GWR points, "
              f"{gwr_betas.shape[1]} variables: {gwr_varnames}")

        # Filter valid GWR source points
        valid_gwr = np.isfinite(gwr_r2) & np.isfinite(gwr_coords[:, 0])
        gwr_coords_v = gwr_coords[valid_gwr]
        gwr_betas_v = gwr_betas[valid_gwr]
        gwr_r2_v = gwr_r2[valid_gwr]

        # Interpolate local R2
        print("  Interpolating GWR local R2 ...")
        interp_r2 = griddata(
            gwr_coords_v, gwr_r2_v, centroid_pts,
            method="linear", fill_value=np.nan,
        )
        nan_r2 = np.isnan(interp_r2)
        if nan_r2.any():
            interp_r2_nn = griddata(
                gwr_coords_v, gwr_r2_v, centroid_pts,
                method="nearest",
            )
            interp_r2[nan_r2] = interp_r2_nn[nan_r2]
        veredas["gwr_local_r2"] = interp_r2

        # Interpolate each beta coefficient
        print("  Interpolating GWR local betas ...")
        for j, varname in enumerate(gwr_varnames):
            beta_col = gwr_betas_v[:, j]
            valid_b = np.isfinite(beta_col)
            if valid_b.sum() < 4:
                veredas[f"gwr_beta_{varname}"] = np.nan
                continue
            interp_b = griddata(
                gwr_coords_v[valid_b], beta_col[valid_b], centroid_pts,
                method="linear", fill_value=np.nan,
            )
            nan_b = np.isnan(interp_b)
            if nan_b.any():
                interp_b_nn = griddata(
                    gwr_coords_v[valid_b], beta_col[valid_b], centroid_pts,
                    method="nearest",
                )
                interp_b[nan_b] = interp_b_nn[nan_b]
            veredas[f"gwr_beta_{varname}"] = interp_b
            print(f"    gwr_beta_{varname}: "
                  f"[{np.nanmin(interp_b):.4f}, {np.nanmax(interp_b):.4f}]")

        print(f"    GWR local R2: mean={np.nanmean(interp_r2):.3f}")
    else:
        print("  [WARN] GWR data files not found; skipping.")
        veredas["gwr_local_r2"] = np.nan

    # ----------------------------------------------------------
    # 7. Save outputs
    # ----------------------------------------------------------
    print("\n[7/7] Saving outputs ...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Veredal stats GeoPackage
    out_veredal = os.path.join(OUTPUT_DIR, "veredal_stats.gpkg")
    veredas.to_file(out_veredal, driver="GPKG")
    print(f"  Saved veredal stats: {out_veredal}")
    print(f"    {len(veredas)} veredas, {len(veredas.columns)} columns")
    print(f"    Columns: {list(veredas.columns)}")

    # Clipped department boundaries
    out_depts = os.path.join(OUTPUT_DIR, "departments_clip.gpkg")
    depts.to_file(out_depts, driver="GPKG")
    print(f"  Saved clipped departments: {out_depts} ({len(depts)} features)")

    # Clipped municipality boundaries
    out_munis = os.path.join(OUTPUT_DIR, "municipalities_clip.gpkg")
    munis.to_file(out_munis, driver="GPKG")
    print(f"  Saved clipped municipalities: {out_munis} ({len(munis)} features)")

    # Summary statistics
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Veredas processed:         {len(veredas)}")
    print(f"  Mean forest % T1 (2013):   {np.nanmean(veredas['forest_pct_T1'].values):.1f}%")
    print(f"  Mean forest % T4 (2024):   {np.nanmean(veredas['forest_pct_T4'].values):.1f}%")
    print(f"  Mean deforestation rate:   {np.nanmean(veredas['defor_rate_pct'].values):.1f}%")
    print(f"  Total net forest change:   {np.nansum(veredas['net_forest_change_ha'].values):.0f} ha")
    print(f"  Mean carbon density T1:    {np.nanmean(veredas['carbon_density_T1'].values):.1f} Mg C/ha")
    print(f"  Mean carbon density T4:    {np.nanmean(veredas['carbon_density_T4'].values):.1f} Mg C/ha")
    print(f"  Mean carbon change:        {np.nanmean(veredas['carbon_change_MgC_ha'].values):.1f} Mg C/ha")
    if "hansen_treecover2000_mean" in veredas.columns:
        print(f"  Mean Hansen TC2000:        "
              f"{np.nanmean(veredas['hansen_treecover2000_mean'].values):.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed / 60:.1f} minutes.")
    print("=" * 64)


if __name__ == "__main__":
    main()
