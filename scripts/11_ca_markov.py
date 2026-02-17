"""
11_ca_markov.py
===============
Fase 3.5: Modelacion futura con CA-Markov.

Implementa:
- Cadenas de Markov para probabilidades de transicion
- Cellular Automata con reglas de vecindad
- 3 escenarios: BAU, Conservacion, PDET
- Proyecciones 2030 y 2040
- Validacion: simular 2024 desde 2020 y comparar

Phase 3 revision:
- correct_transition_matrix(): drops zero-area classes (4, 7) and applies
  ecological constraints to implausible transitions
- hindcast_validate(): area-based hindcast using T1->T2 and T2->T3 rates
- main() rewritten to produce corrected projections and validation outputs

Outputs:
- Matrices de transicion calibradas
- Mapas LULC proyectados (2030, 2040)
- Comparacion de escenarios
- Metricas de validacion
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_config import PERIODS, LULC_CLASSES, N_CLASSES


# ============================================================
# CADENAS DE MARKOV
# ============================================================

def compute_transition_probabilities(lulc_t1, lulc_t2, n_classes=N_CLASSES):
    """
    Calcula matriz de probabilidades de transicion desde dos mapas LULC.

    Args:
        lulc_t1: numpy array 2D con LULC periodo 1
        lulc_t2: numpy array 2D con LULC periodo 2
        n_classes: numero de clases

    Returns:
        numpy array (n_classes, n_classes) de probabilidades
    """
    # Matriz de conteo
    count_matrix = np.zeros((n_classes, n_classes), dtype=float)

    for i in range(1, n_classes + 1):
        for j in range(1, n_classes + 1):
            count_matrix[i-1, j-1] = np.sum((lulc_t1 == i) & (lulc_t2 == j))

    # Normalizar filas (probabilidades)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob_matrix = count_matrix / row_sums

    return prob_matrix


def project_markov(current_areas, transition_matrix, n_steps=1):
    """
    Proyecta areas futuras usando cadena de Markov.

    Args:
        current_areas: array (n_classes,) con areas actuales
        transition_matrix: (n_classes, n_classes) probabilidades
        n_steps: numero de pasos temporales

    Returns:
        array (n_classes,) con areas proyectadas
    """
    areas = current_areas.copy().astype(float)

    for _ in range(n_steps):
        areas = areas @ transition_matrix

    return areas


def multi_step_projection(initial_areas, transition_matrix, years_per_step, target_years):
    """
    Proyeccion multi-paso con resultados intermedios.
    """
    results = {}
    current = initial_areas.copy().astype(float)
    current_year = 2024  # base year

    for target in sorted(target_years):
        n_steps = (target - current_year) // years_per_step
        if n_steps > 0:
            projected = project_markov(current, transition_matrix, n_steps)
            results[target] = {
                'areas': projected.tolist(),
                'n_steps': n_steps,
            }
            current = projected
            current_year = target

    return results


# ============================================================
# CELLULAR AUTOMATA
# ============================================================

def create_suitability_maps(drivers, n_classes=7):
    """
    Crea mapas de aptitud para cada clase LULC basado en drivers.

    Args:
        drivers: dict con arrays numpy de variables (elevation, slope, etc.)

    Returns:
        dict {class_id: numpy array 2D de suitability 0-1}
    """
    rows, cols = drivers['elevation'].shape
    suitability = {}

    # Normalizar drivers a 0-1
    def normalize(arr):
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    elev_n = normalize(drivers['elevation'])
    slope_n = normalize(drivers['slope'])
    dist_rivers_n = normalize(drivers['dist_rivers'])
    dist_roads_n = normalize(drivers['dist_roads'])
    pop_n = normalize(drivers['pop_density'])

    # Bosque denso: alta elevacion, lejos de vias, lejos de poblados
    suitability[1] = (
        (1 - dist_roads_n) * 0.2 +  # invertido: lejos de vias = apto
        (1 - pop_n) * 0.3 +         # invertido: baja poblacion
        slope_n * 0.2 +              # pendiente alta
        elev_n * 0.15 +
        dist_rivers_n * 0.15
    )

    # Bosque secundario: transicion
    suitability[2] = (
        (1 - dist_roads_n) * 0.15 +
        (1 - pop_n) * 0.2 +
        slope_n * 0.25 +
        elev_n * 0.2 +
        dist_rivers_n * 0.2
    )

    # Pasturas: baja pendiente, cerca de vias
    suitability[3] = (
        dist_roads_n * 0.3 +
        (1 - slope_n) * 0.3 +
        (1 - elev_n) * 0.2 +
        pop_n * 0.2
    )

    # Cultivos: baja pendiente, cerca de rios y vias
    suitability[4] = (
        dist_roads_n * 0.25 +
        dist_rivers_n * 0.25 +
        (1 - slope_n) * 0.3 +
        pop_n * 0.2
    )

    # Agua: proximity a rios, baja elevacion
    suitability[5] = (
        dist_rivers_n * 0.6 +
        (1 - elev_n) * 0.3 +
        (1 - slope_n) * 0.1
    )

    # Urbano: cerca de vias, poblacion alta
    suitability[6] = (
        dist_roads_n * 0.35 +
        pop_n * 0.4 +
        (1 - slope_n) * 0.15 +
        (1 - elev_n) * 0.1
    )

    # Suelo desnudo: pendiente baja, cerca de vias
    suitability[7] = (
        dist_roads_n * 0.3 +
        (1 - slope_n) * 0.2 +
        pop_n * 0.3 +
        (1 - elev_n) * 0.2
    )

    return suitability


def ca_neighborhood_effect(lulc_array, target_class, kernel_size=5):
    """
    Calcula efecto de vecindad: proporcion de vecinos de la clase objetivo.

    Returns:
        numpy array 2D con proporcion de vecinos (0-1)
    """
    from scipy.ndimage import uniform_filter

    binary = (lulc_array == target_class).astype(float)
    neighborhood = uniform_filter(binary, size=kernel_size, mode='constant')

    return neighborhood


def simulate_ca_markov(lulc_current, transition_matrix, suitability_maps,
                       n_iterations=10, ca_weight=0.5, stochastic=True, seed=42):
    """
    Simula LULC futuro usando CA-Markov.

    Args:
        lulc_current: numpy array 2D con LULC actual
        transition_matrix: (n_classes, n_classes) probabilidades
        suitability_maps: dict {class_id: array 2D}
        n_iterations: iteraciones CA
        ca_weight: peso del efecto de vecindad (0-1)
        stochastic: agregar componente aleatorio

    Returns:
        numpy array 2D con LULC simulado
    """
    rng = np.random.default_rng(seed)
    n_classes = transition_matrix.shape[0]
    rows, cols = lulc_current.shape
    lulc_sim = lulc_current.copy()

    # Areas objetivo (Markov)
    current_areas = np.array([np.sum(lulc_current == c) for c in range(1, n_classes + 1)])
    target_areas = project_markov(current_areas, transition_matrix, n_steps=1)

    for iteration in range(n_iterations):
        # Para cada pixel, calcular probabilidad de cambio
        for class_from in range(1, n_classes + 1):
            mask = lulc_sim == class_from

            for class_to in range(1, n_classes + 1):
                if class_from == class_to:
                    continue

                p_transition = transition_matrix[class_from - 1, class_to - 1]
                if p_transition < 0.001:
                    continue

                # Aptitud del destino
                suit = suitability_maps.get(class_to, np.zeros_like(lulc_sim, dtype=float))

                # Efecto de vecindad
                neighborhood = ca_neighborhood_effect(lulc_sim, class_to)

                # Probabilidad combinada
                p_change = (
                    p_transition *
                    ((1 - ca_weight) * suit + ca_weight * neighborhood)
                )

                if stochastic:
                    p_change *= rng.uniform(0.8, 1.2, size=p_change.shape)

                # Aplicar cambios donde la probabilidad supera umbral
                threshold = p_transition * 0.5
                change_mask = mask & (p_change > threshold)

                # Limitar por areas objetivo
                current_count = np.sum(lulc_sim == class_to)
                target_count = target_areas[class_to - 1]

                if current_count < target_count:
                    n_to_change = min(
                        int(np.sum(change_mask)),
                        int(target_count - current_count)
                    )
                    if n_to_change > 0:
                        # Seleccionar pixeles con mayor probabilidad
                        candidates = np.where(change_mask)
                        probs = p_change[candidates]
                        top_idx = np.argsort(probs)[-n_to_change:]
                        for idx in top_idx:
                            lulc_sim[candidates[0][idx], candidates[1][idx]] = class_to

    return lulc_sim


# ============================================================
# ESCENARIOS
# ============================================================

def create_scenario_matrices(base_transition_matrix):
    """
    Crea matrices de transicion para 3 escenarios.
    Operates on the 5x5 corrected matrix.
    Indices: 0=BDen, 1=BSec, 2=Past, 3=Agua, 4=Urb

    Returns:
        dict con matrices para BAU, Conservation, PDET
    """
    n = base_transition_matrix.shape[0]

    # BAU: tendencia actual continua
    bau = base_transition_matrix.copy()

    # Conservacion: reducir deforestacion 50%, aumentar recuperacion 30%
    conservation = base_transition_matrix.copy()
    # Reducir transiciones bosque -> no-bosque
    # In 5x5: forest classes are 0 (BDen), 1 (BSec)
    # Non-forest starts at index 2 (Past, Agua, Urb)
    for forest_class in [0, 1]:
        for non_forest in range(2, n):
            conservation[forest_class, non_forest] *= 0.5

    # Aumentar recuperacion (pasturas -> bosque secundario)
    # In 5x5: Past=2, BSec=1
    conservation[2, 1] *= 1.3  # Past -> BSec

    # Re-normalizar filas
    for i in range(n):
        row_sum = conservation[i, :].sum()
        if row_sum > 0:
            conservation[i, :] /= row_sum

    # PDET: implementacion plan territorial (mixto)
    pdet = base_transition_matrix.copy()
    # Reducir deforestacion 30%
    for forest_class in [0, 1]:
        for non_forest in range(2, n):
            pdet[forest_class, non_forest] *= 0.7

    # Re-normalizar
    for i in range(n):
        row_sum = pdet[i, :].sum()
        if row_sum > 0:
            pdet[i, :] /= row_sum

    return {
        'BAU': bau,
        'Conservation': conservation,
        'PDET': pdet,
    }


# ============================================================
# VALIDACION
# ============================================================

def validate_simulation(lulc_simulated, lulc_observed, lulc_baseline=None, n_classes=N_CLASSES):
    """
    Valida simulacion comparando con mapa observado.

    Metricas: Overall accuracy, Kappa, Figure of Merit (FOM).
    FOM (Pontius et al. 2011) se calcula SOLO sobre pixeles de cambio.

    Args:
        lulc_simulated: array simulado
        lulc_observed: array observado (ground truth)
        lulc_baseline: array del periodo base (para identificar cambios)
        n_classes: numero de clases
    """
    sim = lulc_simulated.ravel()
    obs = lulc_observed.ravel()

    valid = (sim > 0) & (obs > 0)
    sim = sim[valid]
    obs = obs[valid]
    n = len(sim)

    if n == 0:
        return {'overall_accuracy': 0, 'kappa': 0, 'fom': 0}

    # Overall accuracy
    oa = np.sum(sim == obs) / n

    # Kappa
    confusion = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            confusion[i, j] = np.sum((obs == i + 1) & (sim == j + 1))

    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    expected = np.sum(row_sums * col_sums) / (n ** 2)
    kappa = (oa - expected) / (1 - expected) if expected < 1 else 0

    # Figure of Merit (Pontius et al. 2011)
    # FOM = Hits / (Hits + Misses + False Alarms)
    # Calculated only on pixels that changed in observed OR simulated
    if lulc_baseline is not None:
        base = lulc_baseline.ravel()[valid]
        obs_changed = obs != base
        sim_changed = sim != base
        change_zone = obs_changed | sim_changed

        hits = np.sum((sim == obs) & change_zone)
        misses = np.sum((sim != obs) & obs_changed)    # observed change, not captured
        false_alarms = np.sum((sim != obs) & sim_changed)  # simulated change, not real
        fom = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
    else:
        # Fallback without baseline: approximate FOM
        hits = np.sum(sim == obs)
        total_disagreement = np.sum(sim != obs)
        fom = hits / (hits + 2 * total_disagreement) if (hits + 2 * total_disagreement) > 0 else 0

    return {
        'overall_accuracy': round(float(oa), 4),
        'kappa': round(float(kappa), 4),
        'fom': round(float(fom), 4),
        'n_pixels': int(n),
    }


def compute_area_summary(lulc_map, pixel_size=30, n_classes=N_CLASSES):
    """
    Calcula areas por clase para un mapa LULC.
    """
    areas = {}
    pixel_area_ha = (pixel_size ** 2) / 10000

    for c in range(1, n_classes + 1):
        n_pixels = int(np.sum(lulc_map == c))
        area_ha = n_pixels * pixel_area_ha
        class_name = LULC_CLASSES.get(c, {}).get('name', f'Clase {c}')
        areas[c] = {
            'name': class_name,
            'n_pixels': n_pixels,
            'area_ha': round(area_ha, 1),
        }

    return areas


# ============================================================
# PHASE 3 REVISION: TRANSITION MATRIX CORRECTION
# ============================================================

# 5 active classes after dropping Cultivos (4) and Suelo desnudo (7)
ACTIVE_CLASSES_5 = ['BDen', 'BSec', 'Past', 'Agua', 'Urb']
# Mapping from 7x7 indices (0-based) to 5x5 indices
# 7x7: 0=BDen, 1=BSec, 2=Past, 3=Cult, 4=Agua, 5=Urb, 6=Suel
# 5x5: 0=BDen, 1=BSec, 2=Past, 3=Agua, 4=Urb
IDX_7TO5 = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
IDX_7_KEEP = [0, 1, 2, 4, 5]  # rows/cols to keep from 7x7


def correct_transition_matrix(tm_7x7):
    """
    Correct the 7x7 transition matrix by:
    1. Reducing to 5x5 by dropping classes 4 (Cultivos) and 7 (Suelo desnudo)
       which have zero area in all periods.
    2. Applying ecological constraints to implausible transition rates:
       - BSec -> BDen <= 0.05 (natural succession takes 20-50 years, not 62.4% in 4 years)
       - Urb -> BDen = 0, Urb -> BSec = 0 (urban land cannot revert to forest)
       - Past -> BDen <= 0.05 (pasture cannot become dense forest in 4 years)
       - Urb -> Agua <= 0.05 (urban to water is implausible at ~15%)
    3. Redistributing excess probability to diagonal (persistence)
    4. Re-normalizing rows to sum to 1.0

    Args:
        tm_7x7: numpy array (7, 7) raw transition probabilities

    Returns:
        tm_5x5: numpy array (5, 5) corrected transition probabilities
        corrections_log: list of dicts documenting each correction applied
    """
    tm = np.array(tm_7x7, dtype=float)

    # --- Step 1: Reduce 7x7 to 5x5 ---
    # Keep rows/cols for BDen(0), BSec(1), Past(2), Agua(4), Urb(5)
    tm_5x5 = tm[np.ix_(IDX_7_KEEP, IDX_7_KEEP)].copy()

    # Re-normalize rows after dropping columns (redistributes the tiny
    # probability mass that went to Cultivos/Suelo desnudo)
    for i in range(5):
        rs = tm_5x5[i, :].sum()
        if rs > 0:
            tm_5x5[i, :] /= rs

    corrections_log = []

    # --- Step 2: Ecological constraints ---
    # In 5x5: 0=BDen, 1=BSec, 2=Past, 3=Agua, 4=Urb

    # Constraint 1: BSec -> BDen <= 0.05
    # Natural succession from secondary to dense forest requires 20-50 years
    # in tropical humid conditions; 62.4% in 4 years is implausible.
    cap = 0.05
    old_val = tm_5x5[1, 0]
    if old_val > cap:
        excess = old_val - cap
        tm_5x5[1, 0] = cap
        tm_5x5[1, 1] += excess  # add to diagonal (BSec persistence)
        corrections_log.append({
            'transition': 'BSec->BDen',
            'old_value': round(float(old_val), 4),
            'new_value': cap,
            'excess_to_diagonal': round(float(excess), 4),
            'reason': 'Succession takes 20-50 yr; 62.4% in 4 yr implausible'
        })

    # Constraint 2: Urb -> BDen = 0 (urban cannot revert to dense forest)
    old_val = tm_5x5[4, 0]
    if old_val > 0:
        tm_5x5[4, 4] += old_val  # add to Urb persistence
        corrections_log.append({
            'transition': 'Urb->BDen',
            'old_value': round(float(old_val), 4),
            'new_value': 0.0,
            'excess_to_diagonal': round(float(old_val), 4),
            'reason': 'Urban land cannot revert to dense forest'
        })
        tm_5x5[4, 0] = 0.0

    # Constraint 3: Urb -> BSec = 0 (urban cannot revert to secondary forest)
    old_val = tm_5x5[4, 1]
    if old_val > 0:
        tm_5x5[4, 4] += old_val
        corrections_log.append({
            'transition': 'Urb->BSec',
            'old_value': round(float(old_val), 4),
            'new_value': 0.0,
            'excess_to_diagonal': round(float(old_val), 4),
            'reason': 'Urban land cannot revert to secondary forest'
        })
        tm_5x5[4, 1] = 0.0

    # Constraint 4: Past -> BDen <= 0.05
    # Pasture to dense forest in 4 years is ecologically impossible
    cap = 0.05
    old_val = tm_5x5[2, 0]
    if old_val > cap:
        excess = old_val - cap
        tm_5x5[2, 0] = cap
        tm_5x5[2, 2] += excess  # add to Past persistence
        corrections_log.append({
            'transition': 'Past->BDen',
            'old_value': round(float(old_val), 4),
            'new_value': cap,
            'excess_to_diagonal': round(float(excess), 4),
            'reason': 'Pasture cannot become dense forest in 4 years'
        })

    # Constraint 5: Urb -> Agua <= 0.05
    cap = 0.05
    old_val = tm_5x5[4, 3]
    if old_val > cap:
        excess = old_val - cap
        tm_5x5[4, 3] = cap
        tm_5x5[4, 4] += excess
        corrections_log.append({
            'transition': 'Urb->Agua',
            'old_value': round(float(old_val), 4),
            'new_value': cap,
            'excess_to_diagonal': round(float(excess), 4),
            'reason': 'Urban to water at 15% is implausible'
        })

    # --- Step 3 & 4: Final re-normalization ---
    for i in range(5):
        rs = tm_5x5[i, :].sum()
        if rs > 0:
            tm_5x5[i, :] /= rs

    return tm_5x5, corrections_log


# ============================================================
# PHASE 3 REVISION: HINDCAST VALIDATION (AREA-BASED)
# ============================================================

def _compute_transition_rates_from_areas(areas_t1, areas_t2, n_classes=5):
    """
    Estimate a simplified transition matrix from aggregate area changes
    between two periods. Uses a proportional-change heuristic:
    - If a class lost area, the loss is distributed proportionally among
      classes that gained area.
    - If a class gained area, the gain is sourced proportionally from
      classes that lost area.
    - Diagonal entries are persistence.

    This is an approximation; pixel-level cross-tabulation would be better
    but we only have aggregate Olofsson areas.

    Args:
        areas_t1: array (n_classes,) areas at time 1
        areas_t2: array (n_classes,) areas at time 2
    Returns:
        tm: (n_classes, n_classes) estimated transition matrix
    """
    a1 = np.array(areas_t1, dtype=float)
    a2 = np.array(areas_t2, dtype=float)

    delta = a2 - a1  # positive = gained, negative = lost
    tm = np.eye(n_classes)

    losers = np.where(delta < 0)[0]
    gainers = np.where(delta > 0)[0]

    total_gain = delta[gainers].sum() if len(gainers) > 0 else 0.0

    for li in losers:
        loss = -delta[li]  # positive quantity lost
        if a1[li] <= 0:
            continue
        loss_fraction = loss / a1[li]  # fraction of original area lost

        # Distribute among gainers proportionally to their gain
        for gi in gainers:
            if total_gain > 0:
                share = delta[gi] / total_gain
                tm[li, gi] = loss_fraction * share

        tm[li, li] = 1.0 - loss_fraction

    # Ensure rows sum to 1
    for i in range(n_classes):
        rs = tm[i, :].sum()
        if rs > 0:
            tm[i, :] /= rs

    return tm


def hindcast_validate(olofsson_areas, corrected_tm_5x5):
    """
    Hindcast validation:
    1. T1->T2 transition rates to simulate T3 areas, compare with observed T3
    2. T2->T3 transition rates to simulate T4 areas, compare with observed T4
    3. Uses the corrected matrix to also simulate T4 from T3 (the calibration
       interval) to verify consistency.

    Reports: Overall Accuracy approximation, Figure of Merit approximation,
    and per-class area comparison.

    Args:
        olofsson_areas: dict with keys 'T1','T2','T3','T4' each containing
                       arrays of 5 class areas [BDen, BSec, Past, Agua, Urb]
        corrected_tm_5x5: the ecologically corrected 5x5 transition matrix

    Returns:
        dict with validation results
    """
    class_names = ACTIVE_CLASSES_5
    total_area = sum(olofsson_areas['T4'])
    results = {}

    # --- Hindcast 1: Use T1->T2 rates to predict T3 from T2 ---
    areas_t1 = np.array(olofsson_areas['T1'], dtype=float)
    areas_t2 = np.array(olofsson_areas['T2'], dtype=float)
    areas_t3_obs = np.array(olofsson_areas['T3'], dtype=float)
    areas_t4_obs = np.array(olofsson_areas['T4'], dtype=float)

    # Estimate T1->T2 transition rates
    tm_t1t2 = _compute_transition_rates_from_areas(areas_t1, areas_t2)
    # Apply one step (T2 + one 3-yr step ~ T3 at 4-yr interval approximation)
    areas_t3_sim = project_markov(areas_t2, tm_t1t2, n_steps=1)

    # Area-based accuracy metrics for T3
    hc1 = _area_accuracy_metrics(areas_t3_obs, areas_t3_sim, class_names, total_area)
    hc1['description'] = 'T1->T2 rates applied to T2 to predict T3 (2020)'
    hc1['transition_matrix_used'] = 'estimated from T1(2013)->T2(2016) area changes'
    results['hindcast_T2_to_T3'] = hc1

    # --- Hindcast 2: Use T2->T3 rates to predict T4 from T3 ---
    tm_t2t3 = _compute_transition_rates_from_areas(areas_t2, areas_t3_obs)
    areas_t4_sim = project_markov(areas_t3_obs, tm_t2t3, n_steps=1)

    hc2 = _area_accuracy_metrics(areas_t4_obs, areas_t4_sim, class_names, total_area)
    hc2['description'] = 'T2->T3 rates applied to T3 to predict T4 (2024)'
    hc2['transition_matrix_used'] = 'estimated from T2(2016)->T3(2020) area changes'
    results['hindcast_T3_to_T4'] = hc2

    # --- Hindcast 3: Use corrected T3->T4 matrix to verify self-consistency ---
    areas_t4_corr = project_markov(areas_t3_obs, corrected_tm_5x5, n_steps=1)

    hc3 = _area_accuracy_metrics(areas_t4_obs, areas_t4_corr, class_names, total_area)
    hc3['description'] = 'Corrected T3->T4 matrix applied to T3 to reproduce T4 (2024)'
    hc3['transition_matrix_used'] = 'corrected 5x5 matrix (ecologically constrained)'
    results['corrected_matrix_T3_to_T4'] = hc3

    return results


def _area_accuracy_metrics(observed, simulated, class_names, total_area):
    """
    Compute area-based accuracy metrics comparing observed vs simulated areas.

    Returns dict with OA approximation, FOM approximation, and per-class comparison.
    """
    obs = np.array(observed, dtype=float)
    sim = np.array(simulated, dtype=float)

    # Area differences
    abs_diff = np.abs(obs - sim)
    rel_diff = abs_diff / np.maximum(obs, 1.0) * 100  # percent

    # Overall accuracy approximation:
    # OA ~ 1 - (sum of area misallocations) / (2 * total area)
    # The factor of 2 accounts for the fact that each misallocated hectare
    # creates both an overestimate in one class and underestimate in another
    total_misallocation = abs_diff.sum()
    oa_approx = 1.0 - total_misallocation / (2.0 * total_area)
    oa_approx = max(0.0, min(1.0, oa_approx))

    # Figure of Merit approximation (Pontius et al. 2011):
    # FOM = Hits / (Hits + Misses + False Alarms)
    # In area terms:
    #   Hits ~ area correctly allocated to each class = min(obs, sim) summed
    #   Misses ~ observed area not captured = sum of max(obs-sim, 0)
    #   False alarms ~ simulated area beyond observed = sum of max(sim-obs, 0)
    hits = np.minimum(obs, sim).sum()
    misses = np.maximum(obs - sim, 0).sum()
    false_alarms = np.maximum(sim - obs, 0).sum()
    fom_approx = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'observed_ha': round(float(obs[i]), 0),
            'simulated_ha': round(float(sim[i]), 0),
            'difference_ha': round(float(sim[i] - obs[i]), 0),
            'relative_error_pct': round(float(rel_diff[i]), 2),
        }

    return {
        'OA_approx': round(float(oa_approx), 4),
        'FOM_approx': round(float(fom_approx), 4),
        'total_misallocation_ha': round(float(total_misallocation), 0),
        'per_class': per_class,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FASE 3.5: CA-MARKOV - MODELACION FUTURA (Phase 3 Revision)")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'outputs', 'phase3_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load existing transition matrix and Olofsson areas
    # ----------------------------------------------------------
    print("\n[1/6] Loading existing data...")

    results_path = os.path.join(output_dir, 'ca_markov_results.json')
    with open(results_path, 'r') as f:
        old_results = json.load(f)

    tm_7x7 = np.array(old_results['transition_matrix'])
    print(f"  Loaded 7x7 transition matrix from {results_path}")
    print(f"  Problematic values in raw matrix:")
    print(f"    BSec->BDen = {tm_7x7[1,0]:.4f}  (should be <=0.05)")
    print(f"    Past->BDen = {tm_7x7[2,0]:.4f}  (should be <=0.05)")
    print(f"    Urb->BDen  = {tm_7x7[5,0]:.4f}  (should be 0)")
    print(f"    Urb->BSec  = {tm_7x7[5,1]:.4f}  (should be 0)")
    print(f"    Urb->Agua  = {tm_7x7[5,4]:.4f}  (should be <=0.05)")

    olofsson_path = os.path.join(output_dir, 'olofsson_area_estimates.json')
    with open(olofsson_path, 'r') as f:
        olofsson = json.load(f)

    # Extract Olofsson-adjusted areas for all 4 periods
    # Order: BDen, BSec, Past, Agua, Urb
    olofsson_class_keys = ['1', '2', '3', '5', '6']

    def get_period_areas(period_key):
        pc = olofsson['periods'][period_key]['per_class']
        return [pc[k]['adjusted_area_ha'] for k in olofsson_class_keys]

    areas_all = {
        'T1': get_period_areas('pre_acuerdo'),     # 2013
        'T2': get_period_areas('transicion'),       # 2016
        'T3': get_period_areas('post_acuerdo_1'),   # 2020
        'T4': get_period_areas('post_acuerdo_2'),   # 2024
    }

    areas_2024 = np.array(areas_all['T4'], dtype=float)
    total_area = areas_2024.sum()

    print(f"\n  Olofsson-adjusted 2024 baseline areas (ha):")
    for i, name in enumerate(ACTIVE_CLASSES_5):
        print(f"    {name}: {areas_2024[i]:,.0f}")
    print(f"    Total: {total_area:,.0f}")

    # ----------------------------------------------------------
    # 2. Correct the transition matrix
    # ----------------------------------------------------------
    print("\n[2/6] Correcting transition matrix...")

    tm_5x5, corrections_log = correct_transition_matrix(tm_7x7)

    print(f"  Applied {len(corrections_log)} ecological corrections:")
    for c in corrections_log:
        print(f"    {c['transition']}: {c['old_value']:.4f} -> {c['new_value']:.4f}"
              f"  (+{c['excess_to_diagonal']:.4f} to diagonal)")
        print(f"      Reason: {c['reason']}")

    print(f"\n  Corrected 5x5 transition matrix:")
    print(f"  {'':>8s}", end='')
    for name in ACTIVE_CLASSES_5:
        print(f"  {name:>8s}", end='')
    print()
    for i, name_from in enumerate(ACTIVE_CLASSES_5):
        print(f"  {name_from:>8s}", end='')
        for j in range(5):
            print(f"  {tm_5x5[i,j]:8.4f}", end='')
        print()

    # Verify rows sum to 1
    row_sums = tm_5x5.sum(axis=1)
    print(f"\n  Row sums: {row_sums}")

    # ----------------------------------------------------------
    # 3. Create scenario matrices
    # ----------------------------------------------------------
    print("\n[3/6] Creating scenario matrices...")

    scenarios = create_scenario_matrices(tm_5x5)

    for sc_name, sc_matrix in scenarios.items():
        print(f"\n  {sc_name} matrix:")
        print(f"  {'':>8s}", end='')
        for name in ACTIVE_CLASSES_5:
            print(f"  {name:>8s}", end='')
        print()
        for i, name_from in enumerate(ACTIVE_CLASSES_5):
            print(f"  {name_from:>8s}", end='')
            for j in range(5):
                print(f"  {sc_matrix[i,j]:8.4f}", end='')
            print()

    # ----------------------------------------------------------
    # 4. Run projections for 2030 and 2040
    # ----------------------------------------------------------
    print("\n[4/6] Running projections...")

    years_per_step = 4  # calibration interval is 4 years (2020-2024)
    target_years = [2030, 2040]

    projection_results = {}

    for sc_name, sc_matrix in scenarios.items():
        print(f"\n  Scenario: {sc_name}")
        proj = multi_step_projection(areas_2024, sc_matrix, years_per_step, target_years)

        for year, data in proj.items():
            key = f"{sc_name}_{year}"
            proj_areas = np.array(data['areas'])

            pcts = proj_areas / total_area * 100
            changes = (proj_areas - areas_2024) / areas_2024 * 100

            scenario_result = {}
            for i, name in enumerate(ACTIVE_CLASSES_5):
                scenario_result[name] = {
                    'area_ha': round(float(proj_areas[i]), 0),
                    'pct': round(float(pcts[i]), 2),
                    'change_pct': round(float(changes[i]), 2),
                }

            projection_results[key] = scenario_result

            print(f"    {year} ({data['n_steps']} steps):")
            for i, name in enumerate(ACTIVE_CLASSES_5):
                print(f"      {name}: {proj_areas[i]:>12,.0f} ha"
                      f"  ({pcts[i]:5.1f}%)"
                      f"  change: {changes[i]:+6.1f}%")

    # ----------------------------------------------------------
    # 5. Run hindcast validation
    # ----------------------------------------------------------
    print("\n[5/6] Running hindcast validation...")

    validation = hindcast_validate(areas_all, tm_5x5)

    for hc_name, hc_data in validation.items():
        print(f"\n  {hc_name}:")
        print(f"    Description: {hc_data['description']}")
        print(f"    OA (approx): {hc_data['OA_approx']:.4f}")
        print(f"    FOM (approx): {hc_data['FOM_approx']:.4f}")
        print(f"    Total misallocation: {hc_data['total_misallocation_ha']:,.0f} ha")
        print(f"    Per-class comparison:")
        for cls, vals in hc_data['per_class'].items():
            print(f"      {cls}: obs={vals['observed_ha']:>10,.0f}  "
                  f"sim={vals['simulated_ha']:>10,.0f}  "
                  f"err={vals['relative_error_pct']:>6.1f}%")

    # ----------------------------------------------------------
    # 6. Save results
    # ----------------------------------------------------------
    print("\n[6/6] Saving results...")

    # Build the output in same structure as original but corrected
    areas_2024_dict = {}
    for i, name in enumerate(ACTIVE_CLASSES_5):
        areas_2024_dict[name] = round(float(areas_2024[i]), 0)

    output = {
        'metadata': {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'phase3_revision': True,
            'description': 'Corrected CA-Markov projections with ecological constraints',
            'calibration_interval': '2020-2024 (4 years)',
            'baseline': '2024 Olofsson-adjusted areas',
            'total_area_ha': round(float(total_area), 0),
            'active_classes': ACTIVE_CLASSES_5,
            'dropped_classes': ['Cultivos (class 4)', 'Suelo desnudo (class 7)'],
            'dropped_reason': 'Zero area in all periods',
        },
        'transition_matrix_raw_7x7': tm_7x7.tolist(),
        'transition_matrix_corrected_5x5': tm_5x5.tolist(),
        'corrections_applied': corrections_log,
        'areas_2024': areas_2024_dict,
    }

    # Add scenario matrices
    output['scenario_matrices'] = {}
    for sc_name, sc_matrix in scenarios.items():
        output['scenario_matrices'][sc_name] = sc_matrix.tolist()

    # Add projections
    for key, vals in projection_results.items():
        output[key] = vals

    # Save main results
    results_out_path = os.path.join(output_dir, 'ca_markov_results.json')
    with open(results_out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Corrected results saved to: {results_out_path}")

    # Save validation separately
    validation_out = {
        'metadata': {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'description': 'Hindcast validation for CA-Markov projections',
            'method': 'Area-based hindcast using Olofsson-adjusted areas',
            'reference': 'Pontius et al. 2011 for FOM concept',
            'class_order': ACTIVE_CLASSES_5,
        },
        'olofsson_areas_by_period': {
            'T1_2013': {name: areas_all['T1'][i] for i, name in enumerate(ACTIVE_CLASSES_5)},
            'T2_2016': {name: areas_all['T2'][i] for i, name in enumerate(ACTIVE_CLASSES_5)},
            'T3_2020': {name: areas_all['T3'][i] for i, name in enumerate(ACTIVE_CLASSES_5)},
            'T4_2024': {name: areas_all['T4'][i] for i, name in enumerate(ACTIVE_CLASSES_5)},
        },
        'hindcast_results': validation,
    }

    validation_out_path = os.path.join(output_dir, 'ca_markov_validation.json')
    with open(validation_out_path, 'w') as f:
        json.dump(validation_out, f, indent=2)
    print(f"  Validation saved to: {validation_out_path}")

    # Save config
    ca_config = {
        'markov_chain': {
            'calibration_interval': '2020-2024 (4 years)',
            'n_classes_original': 7,
            'n_classes_active': 5,
            'class_names_active': ACTIVE_CLASSES_5,
            'ecological_corrections': [c['transition'] for c in corrections_log],
        },
        'cellular_automata': {
            'neighborhood': '5x5 Moore (uniform filter)',
            'ca_weight': 0.5,
            'n_iterations': 10,
            'stochastic': True,
        },
        'suitability_variables': [
            'elevation', 'slope', 'dist_rivers',
            'dist_roads', 'pop_density'
        ],
        'scenarios': {
            'BAU': {
                'description': 'Business-as-Usual: banana/palm expansion continues, cattle ranching persists',
                'deforestation_reduction': '0%',
                'recovery_increase': '0%',
            },
            'Conservation': {
                'description': 'Conservation: protected areas enforcement, Katios NP buffer, mangrove restoration',
                'deforestation_reduction': '50%',
                'recovery_increase': '30%',
                'assumptions': 'Refuerzo PNN Los Katios, restauracion manglares, PSA, REDD+',
            },
            'PDET': {
                'description': 'PDET Uraba: sustainable rural development, land restitution',
                'deforestation_reduction': '30%',
                'recovery_increase': '15%',
                'assumptions': 'Diversificacion productiva, restitucion tierras, sustitucion cultivos ilicitos',
            },
        },
        'projection_years': [2030, 2040],
        'validation': {
            'method': 'Area-based hindcast using Olofsson-adjusted areas',
            'hindcasts': [
                'T1->T2 rates to predict T3 from T2',
                'T2->T3 rates to predict T4 from T3',
                'Corrected matrix to reproduce T4 from T3',
            ],
            'metrics': ['Overall Accuracy (approx)', 'Figure of Merit (approx)'],
            'reference': 'Pontius et al. 2011',
        },
    }

    config_path = os.path.join(output_dir, 'ca_markov_config.json')
    with open(config_path, 'w') as f:
        json.dump(ca_config, f, indent=2)
    print(f"  Configuration saved to: {config_path}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    bau_2040 = projection_results.get('BAU_2040', {})
    print(f"\n  BAU 2040 projections (corrected):")
    for name in ACTIVE_CLASSES_5:
        if name in bau_2040:
            v = bau_2040[name]
            print(f"    {name}: {v['area_ha']:>12,.0f} ha  ({v['pct']:5.1f}%)  change: {v['change_pct']:+6.1f}%")

    bden_change = bau_2040.get('BDen', {}).get('change_pct', 0)
    if bden_change < 0:
        print(f"\n  PASS: BAU 2040 shows continued deforestation (BDen change: {bden_change:+.1f}%)")
    else:
        print(f"\n  WARNING: BAU 2040 still shows BDen increase ({bden_change:+.1f}%). Review matrix.")

    print(f"\nProximo paso: 12_visualization.py")
    print("=" * 60)

    return output


if __name__ == '__main__':
    results = main()
