import ee
import os
from dotenv import load_dotenv

load_dotenv()

# Inicializar Earth Engine
try:
    ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
    print(f"GEE inicializado: {os.getenv('GEE_PROJECT_ID')}")
except Exception as e:
    print(f"Error GEE: {e}")
    raise

# ============================================================
# AREA DE ESTUDIO: Uraba Antioqueno (11-14 municipios)
# ============================================================

# Bounding box para la region de Uraba Antioqueno + Darien Chocoano
# West: -77.2 (Darien/Choco border)
# East: -75.8 (Dabeiba/Mutata eastern limit)
# South: 7.0 (southern Mutata)
# North: 8.9 (Arboletes/San Juan de Uraba coast)
STUDY_AREA_BBOX = ee.Geometry.Rectangle([-77.2, 7.0, -75.8, 8.9])

# Municipios de Uraba Antioqueno por subregion
MUNICIPIOS = {
    'antioquia_eje_bananero': [
        'Apartado', 'Carepa', 'Chigorodo', 'Turbo',
    ],
    'antioquia_norte': [
        'Necoclí', 'San Juan de Uraba', 'Arboletes',
        'San Pedro de Uraba',
    ],
    'antioquia_sur': [
        'Mutata', 'Dabeiba',
    ],
    'antioquia_atrato': [
        'Murindo', 'Vigia del Fuerte',
    ],
    'choco_darien': [
        'Acandi', 'Unguia', 'Riosucio',
    ],
}

# ============================================================
# PERIODOS DE ANALISIS (same 4 periods for cross-regional comparability)
# ============================================================

PERIODS = {
    'pre_acuerdo': {
        'label': 'T1: Pre-Agreement (active conflict)',
        'map_year': 2013,
        'start': '2012-01-01',
        'end': '2014-12-31',
        'context': 'Havana negotiations; AUC post-demobilization; FARC presence; banana/palm expansion'
    },
    'transicion': {
        'label': 'T2: Transition (post-ceasefire)',
        'map_year': 2016,
        'start': '2015-01-01',
        'end': '2017-06-30',
        'context': 'Peace agreement signed; FARC demobilization; governance vacuums; Clan del Golfo expansion'
    },
    'post_acuerdo_1': {
        'label': 'T3: Early post-agreement',
        'map_year': 2020,
        'start': '2019-01-01',
        'end': '2021-06-30',
        'context': 'PDET Uraba implementation; COVID-19; migration crisis; deforestation surge'
    },
    'post_acuerdo_2': {
        'label': 'T4: Recent post-agreement',
        'map_year': 2024,
        'start': '2023-01-01',
        'end': '2024-12-31',
        'context': 'Petro government; Paz Total; deforestation reduction efforts; port development'
    }
}

# ============================================================
# COLECCIONES SATELITALES GEE
# ============================================================

COLLECTIONS = {
    'landsat8': 'LANDSAT/LC08/C02/T1_L2',
    'landsat9': 'LANDSAT/LC09/C02/T1_L2',
    'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
    'modis_ndvi': 'MODIS/061/MOD13Q1',
    'modis_lst': 'MODIS/061/MOD11A2',
    'chirps': 'UCSB-CHG/CHIRPS/DAILY',
    'era5': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    'srtm': 'USGS/SRTMGL1_003',
    'hansen': 'UMD/hansen/global_forest_change_2024_v1_12',
    'jrc_water': 'JRC/GSW1_4/GlobalSurfaceWater',
    'worldpop': 'WorldPop/GP/100m/pop',
    'ghsl': 'JRC/GHSL/P2023A/GHS_SMOD_V2-0',
    'soilgrids': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
    'mangrove_forests': 'LANDSAT/MANGROVE_FORESTS',
}

# ============================================================
# CLASES LULC (8 clases — includes Mangroves for coastal Uraba)
# ============================================================

LULC_CLASSES = {
    1: {'name': 'Dense forest', 'color': '#006400', 'description': 'Canopy cover >60%, humid tropical (Choco)'},
    2: {'name': 'Secondary forest', 'color': '#32CD32', 'description': 'Canopy 30-60%, succession'},
    3: {'name': 'Pastures', 'color': '#FFD700', 'description': 'Cattle ranching, natural/improved grasses'},
    4: {'name': 'Crops', 'color': '#FF8C00', 'description': 'Banana, plantain, oil palm, coca'},
    5: {'name': 'Water', 'color': '#0000FF', 'description': 'Rivers, Gulf of Uraba, wetlands'},
    6: {'name': 'Urban', 'color': '#FF0000', 'description': 'Towns, infrastructure, ports'},
    7: {'name': 'Bare soil', 'color': '#8B4513', 'description': 'Exposed areas, mining, roads'},
    8: {'name': 'Mangroves', 'color': '#2E8B57', 'description': 'Coastal mangrove forests, estuarine'},
}

N_CLASSES = len(LULC_CLASSES)

# ============================================================
# PARAMETROS RANDOM FOREST
# ============================================================

RF_PARAMS = {
    'numberOfTrees': 500,
    'minLeafPopulation': 5,
    'bagFraction': 0.632,
    'seed': 42,
}

# ============================================================
# BANDAS E INDICES ESPECTRALES
# ============================================================

# Mapeo de bandas Landsat 8/9
LANDSAT_BANDS = {
    'blue': 'SR_B2',
    'green': 'SR_B3',
    'red': 'SR_B4',
    'nir': 'SR_B5',
    'swir1': 'SR_B6',
    'swir2': 'SR_B7',
}

# Mapeo de bandas Sentinel-2
SENTINEL_BANDS = {
    'blue': 'B2',
    'green': 'B3',
    'red': 'B4',
    'nir': 'B8',
    'swir1': 'B11',
    'swir2': 'B12',
}

# Indices espectrales a calcular
SPECTRAL_INDICES = ['NDVI', 'EVI', 'NDWI', 'NDBI', 'BSI', 'NBR', 'SAVI', 'MNDWI']

# ============================================================
# PARAMETROS LANDTRENDR
# ============================================================

LANDTRENDR_PARAMS = {
    'maxSegments': 6,
    'spikeThreshold': 0.9,
    'vertexCountOvershoot': 3,
    'preventOneYearRecovery': True,
    'recoveryThreshold': 0.25,
    'pvalThreshold': 0.05,
    'bestModelProportion': 0.75,
    'minObservationsNeeded': 6,
}

# ============================================================
# POOLS DE CARBONO (Mg C/ha) - Tier 2 Choco Bioregion
# Sources: Phillips et al. 2011, Alvarez et al. 2012, IFN Colombia,
#          Blanco-Libreros 2016, Twilley et al. 2018
# ============================================================

CARBON_POOLS = {
    1: {'c_above': 155, 'c_above_se': 20, 'c_below': 39, 'c_below_se': 10,
        'c_soil': 65, 'c_soil_se': 15, 'c_dead': 22, 'c_dead_se': 6},  # Dense tropical humid (Choco)
    2: {'c_above': 65, 'c_above_se': 15, 'c_below': 16, 'c_below_se': 5,
        'c_soil': 55, 'c_soil_se': 12, 'c_dead': 10, 'c_dead_se': 4},  # Secondary forest
    3: {'c_above': 5, 'c_above_se': 2, 'c_below': 3, 'c_below_se': 1,
        'c_soil': 35, 'c_soil_se': 8, 'c_dead': 0.5, 'c_dead_se': 0.3},  # Pastures
    4: {'c_above': 12, 'c_above_se': 4, 'c_below': 3, 'c_below_se': 1,
        'c_soil': 38, 'c_soil_se': 9, 'c_dead': 0.5, 'c_dead_se': 0.3},  # Crops (banana/palm higher)
    5: {'c_above': 0, 'c_above_se': 0, 'c_below': 0, 'c_below_se': 0,
        'c_soil': 0, 'c_soil_se': 0, 'c_dead': 0, 'c_dead_se': 0},  # Water
    6: {'c_above': 2, 'c_above_se': 1, 'c_below': 0, 'c_below_se': 0,
        'c_soil': 18, 'c_soil_se': 5, 'c_dead': 0, 'c_dead_se': 0},  # Urban
    7: {'c_above': 0, 'c_above_se': 0, 'c_below': 0, 'c_below_se': 0,
        'c_soil': 15, 'c_soil_se': 4, 'c_dead': 0, 'c_dead_se': 0},  # Bare soil
    8: {'c_above': 90, 'c_above_se': 25, 'c_below': 25, 'c_below_se': 8,
        'c_soil': 120, 'c_soil_se': 30, 'c_dead': 12, 'c_dead_se': 4},  # Mangroves (blue carbon)
}

print("Configuracion completa cargada")
print(f"  Area de estudio: Uraba Antioqueno ({len(sum(MUNICIPIOS.values(), []))} municipios)")
print(f"  Periodos: {len(PERIODS)}")
print(f"  Clases LULC: {len(LULC_CLASSES)} (including Mangroves)")
print(f"  Colecciones GEE: {len(COLLECTIONS)}")
