# Transiciones LULC Posconflicto — Uraba Antioqueno

Paper academico + pipeline Google Earth Engine para el estudio de transiciones de uso del suelo posconflicto en el Uraba Antioqueno, Colombia (2013-2024). Universidad EAFIT.

Cristian Espinal Maya · Santiago Jimenez Londono — School of Applied Sciences and Engineering, Universidad EAFIT

## Metodologia

Pipeline de 7 fases con Google Earth Engine:

1. **Preprocesamiento** — Compositos Sentinel-2/Landsat, indices espectrales
2. **Clasificacion LULC** — Random Forest, 8 clases (incluye manglares), 4 periodos (2013, 2016, 2020, 2024)
3. **Deteccion de cambio** — Matrices de transicion, tasas de cambio por clase
4. **Servicios ecosistemicos** — Valoracion de stocks de carbono
5. **Analisis climatico** — Correlacion con variables climaticas
6. **Analisis espacial** — Hotspots (Getis-Ord Gi*), GWR (Geographically Weighted Regression)
7. **Proyecciones** — CA-Markov para escenarios futuros

## Estructura

```
├── scripts/            # 17+ modulos de analisis
├── overleaf/           # Manuscrito LaTeX (template MDPI) + supplementary + cover letter
├── data/               # Exports geoespaciales, metadata de variables GWR
├── outputs/            # Figuras (14 PNG), estadisticas JSON, tablas CSV
├── run_analysis.py     # Orquestador principal
├── gee_config.py       # Configuracion GEE para area de estudio de Uraba
└── requirements.txt    # Dependencias Python
```

## Stack

Python (earthengine-api, geemap, geopandas, scikit-learn, rasterio, scipy) + LaTeX (MDPI) + Google Earth Engine
