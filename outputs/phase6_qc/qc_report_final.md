# FASE 6: REPORTE DE VALIDACION DE CALIDAD FINAL — URABA ANTIOQUENO
**Fecha:** 2026-02-15 16:14
**Archivos validados:** 8 JSONs, 6 CSVs, 10 figuras, 1 manuscrito

---

## Resumen

| Resultado | Cantidad |
|-----------|----------|
| PASSED | 199 |
| WARNINGS | 15 |
| ERRORS | 10 |

## Checks Passed

- [x] JSON exists: classification_metrics.json
- [x] JSON non-empty: classification_metrics.json
- [x] JSON exists: change_detection_results.json
- [x] JSON non-empty: change_detection_results.json
- [x] JSON exists: ecosystem_services_results.json
- [x] JSON non-empty: ecosystem_services_results.json
- [x] JSON exists: climate_analysis_results.json
- [x] JSON non-empty: climate_analysis_results.json
- [x] JSON exists: hotspot_analysis_results.json
- [x] JSON non-empty: hotspot_analysis_results.json
- [x] JSON exists: gwr_drivers_results.json
- [x] JSON non-empty: gwr_drivers_results.json
- [x] JSON exists: feature_importance.json
- [x] JSON non-empty: feature_importance.json
- [x] JSON exists: ca_markov_results.json
- [x] JSON non-empty: ca_markov_results.json
- [x] pre_acuerdo: year=2013
- [x] pre_acuerdo: OA in valid range (0.665)
- [x] pre_acuerdo: Kappa in valid range (0.598)
- [x] pre_acuerdo: n_training > 0 (1277)
- [x] pre_acuerdo: n_validation > 0 (513)
- [x] pre_acuerdo: class 1 area >= 0
- [x] pre_acuerdo: class 2 area >= 0
- [x] pre_acuerdo: class 3 area >= 0
- [x] pre_acuerdo: class 4 area >= 0
- [x] pre_acuerdo: class 5 area >= 0
- [x] pre_acuerdo: class 6 area >= 0
- [x] pre_acuerdo: class 7 area >= 0
- [x] pre_acuerdo: class 8 area >= 0
- [x] pre_acuerdo: has 8 LULC classes (8 found)
- [x] pre_acuerdo: dense forest > 0 ha (740,642 ha)
- [x] transicion: year=2016
- [x] transicion: OA in valid range (0.710)
- [x] transicion: Kappa in valid range (0.652)
- [x] transicion: n_training > 0 (1243)
- [x] transicion: n_validation > 0 (556)
- [x] transicion: class 1 area >= 0
- [x] transicion: class 2 area >= 0
- [x] transicion: class 3 area >= 0
- [x] transicion: class 4 area >= 0
- [x] transicion: class 5 area >= 0
- [x] transicion: class 6 area >= 0
- [x] transicion: class 7 area >= 0
- [x] transicion: class 8 area >= 0
- [x] transicion: has 8 LULC classes (8 found)
- [x] transicion: dense forest > 0 ha (720,067 ha)
- [x] post_acuerdo_1: year=2020
- [x] post_acuerdo_1: OA in valid range (0.665)
- [x] post_acuerdo_1: Kappa in valid range (0.598)
- [x] post_acuerdo_1: n_training > 0 (1292)
- [x] post_acuerdo_1: n_validation > 0 (508)
- [x] post_acuerdo_1: class 1 area >= 0
- [x] post_acuerdo_1: class 2 area >= 0
- [x] post_acuerdo_1: class 3 area >= 0
- [x] post_acuerdo_1: class 4 area >= 0
- [x] post_acuerdo_1: class 5 area >= 0
- [x] post_acuerdo_1: class 6 area >= 0
- [x] post_acuerdo_1: class 7 area >= 0
- [x] post_acuerdo_1: class 8 area >= 0
- [x] post_acuerdo_1: has 8 LULC classes (8 found)
- [x] post_acuerdo_1: dense forest > 0 ha (612,924 ha)
- [x] post_acuerdo_2: year=2024
- [x] post_acuerdo_2: OA in valid range (0.719)
- [x] post_acuerdo_2: Kappa in valid range (0.662)
- [x] post_acuerdo_2: n_training > 0 (1309)
- [x] post_acuerdo_2: n_validation > 0 (491)
- [x] post_acuerdo_2: class 1 area >= 0
- [x] post_acuerdo_2: class 2 area >= 0
- [x] post_acuerdo_2: class 3 area >= 0
- [x] post_acuerdo_2: class 4 area >= 0
- [x] post_acuerdo_2: class 5 area >= 0
- [x] post_acuerdo_2: class 6 area >= 0
- [x] post_acuerdo_2: class 7 area >= 0
- [x] post_acuerdo_2: class 8 area >= 0
- [x] post_acuerdo_2: has 8 LULC classes (8 found)
- [x] post_acuerdo_2: dense forest > 0 ha (449,476 ha)
- [x] Cross-period area consistency: max diff 1.1% (threshold 20%)
- [x] T1_T2: has transitions
- [x] T1_T2: has change_rates
- [x] T1_T2: total transition area (3,205,487 ha)
- [x] T1_T2: class 1 annual rate reasonable (-1.1%/yr)
- [x] T1_T2: class 2 annual rate reasonable (8.5%/yr)
- [x] T1_T2: class 3 annual rate reasonable (-3.6%/yr)
- [x] T1_T2: class 5 annual rate reasonable (-0.3%/yr)
- [x] T1_T2: class 6 annual rate reasonable (-1.9%/yr)
- [x] T1_T2: class 8 annual rate reasonable (2.9%/yr)
- [x] T2_T3: has transitions
- [x] T2_T3: has change_rates
- [x] T2_T3: total transition area (3,239,518 ha)
- [x] T2_T3: class 1 annual rate reasonable (-4.0%/yr)
- [x] T2_T3: class 2 annual rate reasonable (-2.8%/yr)
- [x] T2_T3: class 3 annual rate reasonable (3.5%/yr)
- [x] T2_T3: class 5 annual rate reasonable (0.1%/yr)
- [x] T2_T3: class 6 annual rate reasonable (-1.5%/yr)
- [x] T2_T3: class 8 annual rate reasonable (12.5%/yr)
- [x] T3_T4: has transitions
- [x] T3_T4: has change_rates
- [x] T3_T4: total transition area (3,243,545 ha)
- [x] T3_T4: class 1 annual rate reasonable (-7.8%/yr)
- [x] T3_T4: class 2 annual rate reasonable (2.6%/yr)
- [x] T3_T4: class 3 annual rate reasonable (4.4%/yr)
- [x] T3_T4: class 5 annual rate reasonable (-0.0%/yr)
- [x] T3_T4: class 6 annual rate reasonable (-7.9%/yr)
- [x] T3_T4: class 8 annual rate reasonable (-2.6%/yr)
- [x] Hansen GFC data present
- [x] Hansen pre_acuerdo: loss > 0 (26,279 ha)
- [x] Hansen transicion: loss > 0 (37,434 ha)
- [x] Hansen post_acuerdo_1: loss > 0 (29,211 ha)
- [x] Hansen post_acuerdo_2: loss > 0 (18,008 ha)
- [x] pre_acuerdo: carbon > 0 (354 Mt)
- [x] pre_acuerdo: water yield reasonable (2042 mm)
- [x] pre_acuerdo: habitat quality 0-1 (0.195)
- [x] transicion: carbon > 0 (369 Mt)
- [x] transicion: water yield reasonable (1937 mm)
- [x] transicion: habitat quality 0-1 (0.212)
- [x] post_acuerdo_1: carbon > 0 (348 Mt)
- [x] post_acuerdo_1: water yield reasonable (1905 mm)
- [x] post_acuerdo_1: habitat quality 0-1 (0.200)
- [x] post_acuerdo_2: carbon > 0 (315 Mt)
- [x] post_acuerdo_2: water yield reasonable (2194 mm)
- [x] post_acuerdo_2: habitat quality 0-1 (0.247)
- [x] Carbon declining trend: 354 -> 315 Mt
- [x] Carbon change 2013-2016 matches stocks (diff: 0.0 Mt)
- [x] Carbon change 2016-2020 matches stocks (diff: 0.0 Mt)
- [x] Carbon change 2020-2024 matches stocks (diff: 0.0 Mt)
- [x] VIF elevation: 7.30 < 10
- [x] VIF slope: 2.80 < 10
- [x] VIF dist_rivers: 1.83 < 10
- [x] VIF dist_roads: 2.48 < 10
- [x] VIF dist_urban: 1.73 < 10
- [x] VIF pop_density: 1.16 < 10
- [x] VIF precip_annual: 2.43 < 10
- [x] VIF lst_mean: 8.28 < 10
- [x] VIF clay_content: 1.34 < 10
- [x] OLS R2 valid: 0.1437
- [x] GWR R2 (0.4264) > OLS R2 (0.1437)
- [x] GWR AIC (-7143) < OLS AIC (-2923)
- [x] Comparison OLS R2 matches OLS section
- [x] Comparison GWR R2 matches GWR section
- [x] TM row 0 sums to 1.0000 (~1.0)
- [x] TM row 1 sums to 1.0000 (~1.0)
- [x] TM row 2 sums to 1.0000 (~1.0)
- [x] TM row 4 sums to 1.0000 (~1.0)
- [x] TM row 5 sums to 1.0000 (~1.0)
- [x] TM row 7 sums to 1.0000 (~1.0)
- [x] Scenario exists: BAU_2030
- [x] BAU_2030: class pcts sum to 100.0% (~100%)
- [x] Scenario exists: BAU_2040
- [x] BAU_2040: class pcts sum to 100.0% (~100%)
- [x] Scenario exists: Conservation_2030
- [x] Conservation_2030: class pcts sum to 100.0% (~100%)
- [x] Scenario exists: Conservation_2040
- [x] Conservation_2040: class pcts sum to 100.0% (~100%)
- [x] Scenario exists: PDET_2030
- [x] PDET_2030: class pcts sum to 100.0% (~100%)
- [x] Scenario exists: PDET_2040
- [x] PDET_2040: class pcts sum to 100.0% (~100%)
- [x] Conservation forest (36.52%) >= BAU forest (29.4%)
- [x] Table exists: table01_accuracy.csv
- [x] table01_accuracy.csv: has data rows (4 rows)
- [x] Table exists: table02_class_areas.csv
- [x] table02_class_areas.csv: has data rows (8 rows)
- [x] Table exists: table03_change_rates.csv
- [x] table03_change_rates.csv: has data rows (18 rows)
- [x] Table exists: table04_ecosystem_services.csv
- [x] table04_ecosystem_services.csv: has data rows (4 rows)
- [x] Table exists: table05_gwr_results.csv
- [x] table05_gwr_results.csv: has data rows (9 rows)
- [x] Table exists: table06_camarkov_projections.csv
- [x] table06_camarkov_projections.csv: has data rows (7 rows)
- [x] Table01 OA 2013 (66.5%) matches JSON (66.5%)
- [x] Table01 OA 2016 (71.0%) matches JSON (71.0%)
- [x] Table01 OA 2020 (66.5%) matches JSON (66.5%)
- [x] Table01 OA 2024 (71.9%) matches JSON (71.9%)
- [x] Table04 Carbon 2013 (354.5 Mt) matches JSON (354.5 Mt)
- [x] Table04 Carbon 2016 (368.5 Mt) matches JSON (368.5 Mt)
- [x] Table04 Carbon 2020 (347.8 Mt) matches JSON (347.8 Mt)
- [x] Table04 Carbon 2024 (315.1 Mt) matches JSON (315.1 Mt)
- [x] No [XX] placeholders remaining (0 found)
- [x] No 'to be populated' remaining (0 found)
- [x] Manuscript mentions Uraba
- [x] Manuscript cites study area ~11,000 km2
- [x] Manuscript addresses H1
- [x] Manuscript addresses H2
- [x] Manuscript addresses H3
- [x] Manuscript addresses H4
- [x] Manuscript cites Olofsson et al.
- [x] Manuscript cites Pontius (QD/AD)
- [x] Manuscript cites Hansen GFC
- [x] Total area T1 (3,208,788) vs T4 (3,243,545): 1.1% diff
- [x] T1 class 1: change det (740,503) vs clf (740,642) = 0.0% diff
- [x] T1 class 2: change det (510,560) vs clf (510,575) = 0.0% diff
- [x] T1 class 3: change det (1,023,347) vs clf (1,026,478) = 0.3% diff
- [x] T1 carbon: pixel-based (354 Tg) vs Olofsson-based (354 Tg) = 0.0% diff (expected: different areas)
- [x] pre_acuerdo: OA (0.665) > chance level (0.125 for 8 classes)
- [x] transicion: OA (0.710) > chance level (0.125 for 8 classes)
- [x] post_acuerdo_1: OA (0.665) > chance level (0.125 for 8 classes)
- [x] post_acuerdo_2: OA (0.719) > chance level (0.125 for 8 classes)
- [x] Elevation negatively associated with deforestation (-0.144)

## Warnings

- [!] pre_acuerdo: total area reasonable (3,208,788 ha)
- [!] transicion: total area reasonable (3,239,518 ha)
- [!] post_acuerdo_1: total area reasonable (3,243,545 ha)
- [!] post_acuerdo_2: total area reasonable (3,243,545 ha)
- [!] Manuscript references Fig. 1
- [!] Manuscript references Fig. 2
- [!] Manuscript references Fig. 3
- [!] Manuscript references Fig. 4
- [!] Manuscript references Fig. 5
- [!] Manuscript references Fig. 6
- [!] Manuscript references Fig. 7
- [!] Manuscript references Fig. 8
- [!] Manuscript references Fig. 9
- [!] Manuscript references Fig. 10
- [!] T1 total area (32,088 km2) in expected ~11,000 km2 range

## Errors

- [X] Figure exists: fig01_study_area.png
- [X] Figure exists: fig02_lulc_maps.png
- [X] Figure exists: fig03_area_trends.png
- [X] Figure exists: fig04_transition_matrices.png
- [X] Figure exists: fig05_deforestation_rates.png
- [X] Figure exists: fig06_hotspots.png
- [X] Figure exists: fig07_ecosystem_services.png
- [X] Figure exists: fig08_gwr_coefficients.png
- [X] Figure exists: fig09_future_scenarios.png
- [X] Figure exists: fig10_climate_deforestation.png

---

## Datos Validados

### Phase 3 JSONs
- classification_metrics.json: 4 periods, 8 classes, OA/Kappa/confusion matrices
- change_detection_results.json: 3 transition matrices + Hansen GFC
- ecosystem_services_results.json: Carbon, water yield, habitat quality
- climate_analysis_results.json: Precipitation, LST, SPI, trends
- hotspot_analysis_results.json: Moran's I, Gi* counts
- gwr_drivers_results.json: OLS, GWR, VIF, 8 drivers
- feature_importance.json: RF importance per period
- ca_markov_results.json: Transition matrix, 6 scenarios

### Phase 4 Figures (10)
- fig04a_lulc_composition.png (175 KB)
- fig04b_forest_trend.png (228 KB)
- fig05_transition_matrices.png (309 KB)
- fig06_deforestation_rates.png (264 KB)
- fig08_ecosystem_services.png (460 KB)
- fig11_camarkov_scenarios.png (252 KB)
- fig_s1_feature_importance.png (210 KB)
- fig_s2_climate.png (265 KB)

### Phase 4 Tables (6)
- table01_accuracy.csv
- table02_class_areas.csv
- table03_change_rates.csv
- table04_ecosystem_services.csv
- table05_gwr_results.csv
- table06_camarkov_projections.csv

### Phase 5 Manuscript
- Manuscript validated for structural completeness and key references
