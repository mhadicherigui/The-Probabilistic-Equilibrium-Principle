#!/usr/bin/env python3
"""
ghz_presentable.py

Script présentable et reproductible pour :
 - CHSH (Bell)
 - Double-slit S2 (Bloch) et S1 (Poincaré)
 - GHZ (même règles d'origine) : runs sans post-sélection + post-sélection minimale pour comparaison

Principes :
 - Paramètres par défaut choisis pour être physiquement défendables.
 - Pas de modification des règles probabilistes du modèle GHZ (option B).
 - Résultats imprimés clairement et sauvegardés (CSV + JSON).
 - Dépendance : numpy seulement.

Usage : python ghz_presentable.py
"""

from __future__ import annotations
import numpy as np
import os
import json
import csv
from datetime import datetime
import time

# ---------------------------
# Réglages généraux
# ---------------------------
SEED = 42
np.random.seed(SEED)

RESULTS_DIR = "results_presentable"
os.makedirs(RESULTS_DIR, exist_ok=True)

NOW_TAG = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


# ---------------------------
# Utilitaires
# ---------------------------
def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv_rows(header: list[str], rows: list[list], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------
# CHSH (Bell) - identique aux versions testées
# ---------------------------
def compute_S(pa: float, pb: float, N: int = 100_000):
    """
    Retourne (S_abs, pa_emp, pb_emp, SE_S_approx)
    """
    def get_probs(delta_deg):
        delta = np.deg2rad(delta_deg)
        c2 = np.cos(delta) ** 2
        s2 = np.sin(delta) ** 2
        return [0.5 * c2, 0.5 * c2, 0.5 * s2, 0.5 * s2]

    def flip_outcome(out, f_up, f_down):
        if out == 1 and np.random.random() < f_down:
            return -1
        if out == -1 and np.random.random() < f_up:
            return 1
        return out

    def compute_E(delta_deg, Nlocal, pa_local, pb_local):
        probs = get_probs(delta_deg)
        choices = ['pp', 'mm', 'pm', 'mp']
        sum_ab = 0.0
        count_ap = 0
        count_bp = 0
        for _ in range(Nlocal):
            o = np.random.choice(choices, p=probs)
            if o == 'pp':
                A, B = 1, 1
            elif o == 'mm':
                A, B = -1, -1
            elif o == 'pm':
                A, B = 1, -1
            else:
                A, B = -1, 1

            f_a_up = f_a_down = 0.0
            if pa_local < 0.5:
                f_a_down = 1 - 2 * pa_local
            elif pa_local > 0.5:
                f_a_up = 2 * pa_local - 1
            A = flip_outcome(A, f_a_up, f_a_down)

            f_b_up = f_b_down = 0.0
            if pb_local < 0.5:
                f_b_down = 1 - 2 * pb_local
            elif pb_local > 0.5:
                f_b_up = 2 * pb_local - 1
            B = flip_outcome(B, f_b_up, f_b_down)

            sum_ab += A * B
            if A == 1:
                count_ap += 1
            if B == 1:
                count_bp += 1

        return sum_ab / Nlocal, count_ap / Nlocal, count_bp / Nlocal

    deltas = {'ab': 22.5, 'abp': 67.5, 'apb': -22.5, 'apbp': 22.5}
    E1, pa1, pb1 = compute_E(deltas['ab'], N, pa, pb)
    E2, pa2, pb2 = compute_E(deltas['abp'], N, pa, pb)
    E3, pa3, pb3 = compute_E(deltas['apb'], N, pa, pb)
    E4, pa4, pb4 = compute_E(deltas['apbp'], N, pa, pb)

    S_abs = abs(E1 + E3 + E4 - E2)
    avg_pa = (pa1 + pa2 + pa3 + pa4) / 4.0
    avg_pb = (pb1 + pb2 + pb3 + pb4) / 4.0

    # Estimation approchée de l'erreur standard sur S (ordre de grandeur)
    var_terms = [(1 - E1 ** 2), (1 - E2 ** 2), (1 - E3 ** 2), (1 - E4 ** 2)]
    se_S = np.sqrt(sum(var_terms)) / np.sqrt(N)

    return S_abs, avg_pa, avg_pb, se_S


# ---------------------------
# Double-slit S2 (Bloch)
# ---------------------------
def simulate_double_slit_s2(distribution='uniform', N: int = 100_000,
                            gamma: float = 3 * np.pi, k: float = 0.8, g: float = 0.4):
    """
    Paramètres par défaut choisis pour être physiquement défendables
    gamma = 3π (franges raisonnables), k,g ajustés pour bonne visibilité.
    """
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros_like(alphas)
    delta = np.pi / 2

    if distribution == 'uniform':
        u = np.random.uniform(0, 1, N)
        v = np.random.uniform(0, 1, N)
        theta = np.arccos(1 - 2 * u)
        phi = 2 * np.pi * v
        lambd_x = np.sin(theta) * np.cos(phi)
        lambd_y = np.sin(theta) * np.sin(phi)
        lambd_z = np.cos(theta)
    else:
        z = np.random.beta(2, 5, N)  # polar clustering moderate
        cos_theta = 2 * z - 1
        theta = np.arccos(cos_theta)
        phi = 2 * np.pi * np.random.uniform(0, 1, N)
        lambd_x = np.sin(theta) * np.cos(phi)
        lambd_y = np.sin(theta) * np.sin(phi)
        lambd_z = cos_theta

    for i, alpha in enumerate(alphas):
        p1 = np.array([np.cos(alpha), np.sin(alpha), 0.0])
        p2 = np.array([np.cos(alpha + delta), np.sin(alpha + delta), 0.0])
        dot1 = lambd_x * p1[0] + lambd_y * p1[1] + lambd_z * p1[2]
        dot2 = lambd_x * p2[0] + lambd_y * p2[1] + lambd_z * p2[2]
        delta_phi = g * k * (dot1 - dot2) + gamma * np.sin(alpha)
        I[i] = 1 + np.mean(np.cos(delta_phi))

    Imax, Imin = float(np.max(I)), float(np.min(I))
    V = (Imax - Imin) / (Imax + Imin)
    fringes = int(round(float(gamma) / np.pi))
    return Imax, Imin, V, fringes


# ---------------------------
# Double-slit S1 (Poincaré)
# ---------------------------
def simulate_double_slit_s1(distribution='uniform', N: int = 100_000,
                            gamma: float = 5 * np.pi, k: float = 0.3, noise_sigma: float = 0.5):
    """
    Par défaut noise_sigma = 0.5 rad (≈29°) : raisonnable physiquement tout en laissant une visibilité élevée.
    """
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros_like(alphas)
    delta = np.pi / 2

    if distribution == 'uniform':
        lambd = np.random.uniform(0, 2 * np.pi, N)
        lambd2 = lambd.copy()
    else:
        z = np.random.beta(2, 5, N)
        lambd = 2 * np.pi * z
        lambd2 = (lambd + np.random.normal(0, noise_sigma, N)) % (2 * np.pi)

    for i, alpha in enumerate(alphas):
        delta_phi = 2 * gamma * np.sin(alpha) + k * (np.cos(alpha - lambd) - np.cos(alpha + delta - lambd2))
        I[i] = 1 + np.mean(np.cos(delta_phi))

    Imax, Imin = float(np.max(I)), float(np.min(I))
    V = (Imax - Imin) / (Imax + Imin)
    fringes = int(round(float(gamma) / np.pi))
    return Imax, Imin, V, fringes


# ---------------------------
# GHZ variant (Option B) - règles inchangées, on modifie seulement l'échantillonnage
# ---------------------------
def simulate_ghz_variant(model: str = 'S2', alpha: float = 10.0, beta: float = 10.0,
                         sigma_common: float = 0.002, sigma_delta: float = 0.0005,
                         N: int = 100_000):
    """
    Conserve la règle p_i = clip(R/4 + 0.5).
    alpha,beta contrôlent la concentration Beta pour la variable lambda de base.
    Retourne : M, p_avg, efficiency_avg, raw_data, counts_summary
    raw_data : dict pour chaque réglage ('xxx','xyy','yxy','yyx') -> tuple (A1,A2,A3,p1,p2,p3)
    counts_summary : p_ppp et p_mmm (analyse seulement)
    """

    # S2: vecteurs sur la sphère
    if model == 'S2':
        z = np.random.beta(alpha, beta, N)
        cos_t = 2 * z - 1
        theta = np.arccos(cos_t)
        phi = 2 * np.pi * np.random.uniform(0, 1, N)
        sin_t = np.sin(theta)
        base = np.column_stack((sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t))

        eta_common = np.random.normal(0, sigma_common, (N, 3))
        delta1 = np.random.normal(0, sigma_delta, (N, 3))
        delta2 = np.random.normal(0, sigma_delta, (N, 3))
        delta3 = np.random.normal(0, sigma_delta, (N, 3))

        def normalize_rows(v):
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return v / n

        l1 = normalize_rows(base + eta_common + delta1)
        l2 = normalize_rows(base + eta_common + delta2)
        l3 = normalize_rows(base + eta_common + delta3)

        def sample_s2():
            z2 = np.random.beta(alpha, beta, N)
            cos_t2 = 2 * z2 - 1
            theta2 = np.arccos(cos_t2)
            phi2 = 2 * np.pi * np.random.uniform(0, 1, N)
            sin_t2 = np.sin(theta2)
            return np.column_stack((sin_t2 * np.cos(phi2), sin_t2 * np.sin(phi2), cos_t2))

        l1pp = sample_s2()
        l2pp = sample_s2()
        l3pp = sample_s2()

        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])

        def sign_dot(vecs, n):
            s = np.sign(np.dot(vecs, n))
            s[s == 0] = 1
            return s

        def get_outcomes(n1, n2, n3):
            s1_1 = sign_dot(base, n1)
            s2_1 = sign_dot(l1, n1)
            s3_1 = sign_dot(l1pp, n1)
            R1 = s1_1 * s2_1 + s3_1
            p1 = np.clip(R1 / 4.0 + 0.5, 0, 1)
            A1 = (np.random.random(N) < p1).astype(int) * 2 - 1

            s1_2 = sign_dot(base, n2)
            s2_2 = sign_dot(l2, n2)
            s3_2 = sign_dot(l2pp, n2)
            R2 = s1_2 * s2_2 + s3_2
            p2 = np.clip(R2 / 4.0 + 0.5, 0, 1)
            A2 = (np.random.random(N) < p2).astype(int) * 2 - 1

            s1_3 = sign_dot(base, n3)
            s2_3 = sign_dot(l3, n3)
            s3_3 = sign_dot(l3pp, n3)
            R3 = s1_3 * s2_3 + s3_3
            p3 = np.clip(R3 / 4.0 + 0.5, 0, 1)
            A3 = (np.random.random(N) < p3).astype(int) * 2 - 1

            product = A1 * A2 * A3
            E = float(np.mean(product))
            eff = float(np.mean((A1 == 1) & (A2 == 1) & (A3 == 1)))
            p_avg = float((np.mean(p1) + np.mean(p2) + np.mean(p3)) / 3.0)
            return E, p_avg, eff, (A1, A2, A3, p1, p2, p3)

        E_xxx, p_xxx, eff_xxx, data_xxx = get_outcomes(x, x, x)
        E_xyy, p_xyy, eff_xyy, data_xyy = get_outcomes(x, y, y)
        E_yxy, p_yxy, eff_yxy, data_yxy = get_outcomes(y, x, y)
        E_yyx, p_yyx, eff_yyx, data_yyx = get_outcomes(y, y, x)

    elif model == 'S1':
        # angle-based sampling
        z = np.random.beta(alpha, beta, N)
        lambda_angle = 2 * np.pi * z
        eta_common = np.random.normal(0, sigma_common, N)
        delta1 = np.random.normal(0, sigma_delta, N)
        delta2 = np.random.normal(0, sigma_delta, N)
        delta3 = np.random.normal(0, sigma_delta, N)

        l1 = (lambda_angle + eta_common + delta1) % (2 * np.pi)
        l2 = (lambda_angle + eta_common + delta2) % (2 * np.pi)
        l3 = (lambda_angle + eta_common + delta3) % (2 * np.pi)

        l1pp = (2 * np.pi * np.random.beta(alpha, beta, N)) % (2 * np.pi)
        l2pp = (2 * np.pi * np.random.beta(alpha, beta, N)) % (2 * np.pi)
        l3pp = (2 * np.pi * np.random.beta(alpha, beta, N)) % (2 * np.pi)

        x, y = 0.0, np.pi / 2.0

        def sign_cos(n, lv):
            s = np.sign(np.cos(n - lv))
            s[s == 0] = 1
            return s

        def get_outcomes(n1, n2, n3):
            s1_1 = sign_cos(n1, lambda_angle)
            s2_1 = sign_cos(n1, l1)
            s3_1 = sign_cos(n1, l1pp)
            R1 = s1_1 * s2_1 + s3_1
            p1 = np.clip(R1 / 4.0 + 0.5, 0, 1)
            A1 = (np.random.random(N) < p1).astype(int) * 2 - 1

            s1_2 = sign_cos(n2, lambda_angle)
            s2_2 = sign_cos(n2, l2)
            s3_2 = sign_cos(n2, l2pp)
            R2 = s1_2 * s2_2 + s3_2
            p2 = np.clip(R2 / 4.0 + 0.5, 0, 1)
            A2 = (np.random.random(N) < p2).astype(int) * 2 - 1

            s1_3 = sign_cos(n3, lambda_angle)
            s2_3 = sign_cos(n3, l3)
            s3_3 = sign_cos(n3, l3pp)
            R3 = s1_3 * s2_3 + s3_3
            p3 = np.clip(R3 / 4.0 + 0.5, 0, 1)
            A3 = (np.random.random(N) < p3).astype(int) * 2 - 1

            product = A1 * A2 * A3
            E = float(np.mean(product))
            eff = float(np.mean((A1 == 1) & (A2 == 1) & (A3 == 1)))
            p_avg = float((np.mean(p1) + np.mean(p2) + np.mean(p3)) / 3.0)
            return E, p_avg, eff, (A1, A2, A3, p1, p2, p3)

        E_xxx, p_xxx, eff_xxx, data_xxx = get_outcomes(x, x, x)
        E_xyy, p_xyy, eff_xyy, data_xyy = get_outcomes(x, y, y)
        E_yxy, p_yxy, eff_yxy, data_yxy = get_outcomes(y, x, y)
        E_yyx, p_yyx, eff_yyx, data_yyx = get_outcomes(y, y, x)

    # M and averages
    M = abs(E_xyy + E_yxy + E_yyx - E_xxx)
    p_avg = (p_xxx + p_xyy + p_yxy + p_yyx) / 4.0
    eff_avg = (eff_xxx + eff_xyy + eff_yxy + eff_yyx) / 4.0

    raw_data = {
        'xxx': data_xxx,
        'xyy': data_xyy,
        'yxy': data_yxy,
        'yyx': data_yyx
    }

    # counts for +++ and ---
    counts_ppp = 0
    counts_mmm = 0
    total_trials = N * 4
    for key in ['xxx', 'xyy', 'yxy', 'yyx']:
        A1, A2, A3, p1, p2, p3 = raw_data[key]
        counts_ppp += int(np.sum((A1 == 1) & (A2 == 1) & (A3 == 1)))
        counts_mmm += int(np.sum((A1 == -1) & (A2 == -1) & (A3 == -1)))

    p_ppp = counts_ppp / total_trials
    p_mmm = counts_mmm / total_trials
    counts_summary = {'counts_ppp': counts_ppp, 'counts_mmm': counts_mmm, 'p_ppp': p_ppp, 'p_mmm': p_mmm}

    return float(M), float(p_avg), float(eff_avg), raw_data, counts_summary


# ---------------------------
# Post-selection minimale automatique (analyse seulement)
# ---------------------------
def ghz_minimal_postselection_from_raw(raw_data, eps_target: float = 0.98, max_fraction: float = 0.33):
    """
    Retour d'un dictionnaire :
      - fractions (par réglage)
      - E_cond (E conditionné par réglage)
      - M_post (valeur M après post-selection)
      - overall_efficiency (moyenne des fractions sélectionnées)
    """
    expected_signs = {'xxx': 1, 'xyy': -1, 'yxy': -1, 'yyx': -1}
    N = raw_data['xxx'][0].shape[0]
    fractions = {}
    E_cond = {}
    selected_indices = {}

    def product_and_score(arr_tuple, expected):
        A1, A2, A3, p1, p2, p3 = arr_tuple
        prod = A1 * A2 * A3
        if expected == 1:
            score = p1 * p2 * p3
        else:
            a = (1 - p1) * (1 - p2) * (1 - p3)
            b = (1 - p1) * p2 * p3
            c = p1 * (1 - p2) * p3
            d = p1 * p2 * (1 - p3)
            score = a + b + c + d
        return prod, score

    for key in ['xxx', 'xyy', 'yxy', 'yyx']:
        prod_arr, score = product_and_score(raw_data[key], expected=expected_signs[key])
        idx_sorted = np.argsort(-score)
        prod_sorted = prod_arr[idx_sorted]

        achieved = False
        frac_candidates = np.concatenate((
            np.linspace(0.001, 0.01, 10),
            np.linspace(0.01, min(0.1, max_fraction), 10),
            np.linspace(min(0.1, max_fraction), max_fraction, 10)
        ))
        for frac in frac_candidates:
            k = max(1, int(np.ceil(frac * N)))
            sel_mean = float(np.mean(prod_sorted[:k])) if k > 0 else float(np.mean(prod_sorted))
            if expected_signs[key] == 1 and sel_mean >= eps_target:
                fractions[key] = k / N
                selected_indices[key] = idx_sorted[:k]
                E_cond[key] = sel_mean
                achieved = True
                break
            if expected_signs[key] == -1 and sel_mean <= -eps_target:
                fractions[key] = k / N
                selected_indices[key] = idx_sorted[:k]
                E_cond[key] = sel_mean
                achieved = True
                break

        if not achieved:
            A1, A2, A3, _, _, _ = raw_data[key]
            prod_arr2 = A1 * A2 * A3
            mask = (prod_arr2 == expected_signs[key])
            k = int(np.sum(mask))
            if k == 0:
                fractions[key] = 0.0
                selected_indices[key] = np.array([], dtype=int)
                E_cond[key] = float(np.mean(prod_arr2))
            else:
                fractions[key] = k / N
                selected_indices[key] = np.where(mask)[0]
                E_cond[key] = float(np.mean(prod_arr2[mask]))

    # Final conditional E and M_post
    E_values = {}
    for key in ['xxx', 'xyy', 'yxy', 'yyx']:
        idx = selected_indices.get(key, np.array([], dtype=int))
        A1, A2, A3, _, _, _ = raw_data[key]
        prod_arr = A1 * A2 * A3
        if idx.size == 0:
            E_values[key] = float(np.mean(prod_arr))
        else:
            E_values[key] = float(np.mean(prod_arr[idx]))

    M_post = abs(E_values['xyy'] + E_values['yxy'] + E_values['yyx'] - E_values['xxx'])
    overall_eff = float(np.mean([fractions.get(k, 0.0) for k in ['xxx', 'xyy', 'yxy', 'yyx']]))

    return {'fractions': fractions, 'E_cond': E_values, 'M_post': float(M_post), 'overall_efficiency': overall_eff}


# ---------------------------
# Routine principale (exécution et sauvegarde)
# ---------------------------
def main():
    t0 = time.time()
    print_header("Simulation — paramètres défendables et présentables")
    print(f"Seed: {SEED}   Tag: {NOW_TAG}")

    # 1) CHSH
    print_header("CHSH (Bell) — tests rapides")
    chsh_results = []
    for pa, pb in [(0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.35, 0.35)]:
        S, pa_emp, pb_emp, se_S = compute_S(pa, pb, N=100_000)
        chsh_results.append({'pa': pa, 'pb': pb, 'S': S, 'pa_emp': pa_emp, 'pb_emp': pb_emp, 'se_S': se_S})
        print(f"pa={pa:.2f}, pb={pb:.2f} → |S|={S:.3f}, P(A+)= {pa_emp:.3f}, P(B+)= {pb_emp:.3f}, SE(S)~{se_S:.4f}")

    # Save CHSH
    chsh_csv_rows = [[r['pa'], r['pb'], r['S'], r['pa_emp'], r['pb_emp'], r['se_S']] for r in chsh_results]
    save_csv_rows(['pa', 'pb', 'S', 'pa_emp', 'pb_emp', 'se_S'], chsh_csv_rows,
                  os.path.join(RESULTS_DIR, f"chsh_{NOW_TAG}.csv"))

    # 2) Double-slit S2 & S1 (défendables)
    print_header("Double-slit S2 (Bloch) — paramètres défendables")
    s2_params = {'distribution': 'uniform', 'N': 100_000, 'gamma': 3 * np.pi, 'k': 0.8, 'g': 0.4}
    Imax_s2, Imin_s2, V_s2, fr_s2 = simulate_double_slit_s2(**s2_params)
    print(f"S2 (gamma=3π, k=0.8, g=0.4): Imax={Imax_s2:.3f}, Imin={Imin_s2:.3f}, V={V_s2:.3f}, Fringes≈{fr_s2}")

    print_header("Double-slit S1 (Poincaré) — paramètres défendables")
    s1_params = {'distribution': 'uniform', 'N': 100_000, 'gamma': 5 * np.pi, 'k': 0.3, 'noise_sigma': 0.5}
    Imax_s1, Imin_s1, V_s1, fr_s1 = simulate_double_slit_s1(**s1_params)
    print(f"S1 (gamma=5π, k=0.3, noise_sigma=0.5): Imax={Imax_s1:.3f}, Imin={Imin_s1:.3f}, V={V_s1:.3f}, Fringes≈{fr_s1}")

    # Save double-slit summary
    ds_rows = [
        ['S2', s2_params['gamma'] / np.pi, s2_params['k'], s2_params['g'], Imax_s2, Imin_s2, V_s2, fr_s2],
        ['S1', s1_params['gamma'] / np.pi, s1_params['k'], s1_params['noise_sigma'], Imax_s1, Imin_s1, V_s1, fr_s1]
    ]
    save_csv_rows(['model', 'gamma_pi', 'k_or_noise', 'param2', 'Imax', 'Imin', 'V', 'fringes'],
                  ds_rows, os.path.join(RESULTS_DIR, f"double_slit_summary_{NOW_TAG}.csv"))

    # 3) GHZ variant (sampling changes only)
    print_header("GHZ — exécution (sampling contrôlé, règles inchangées)")
    ghz_params = {
        'model': 'S2',
        'alpha': 10.0,       # concentration Beta raisonnable
        'beta': 10.0,
        'sigma_common': 0.002,
        'sigma_delta': 0.0005,
        'N': 120_000
    }
    M_no_post, pavg_no_post, eff_no_post, raw_no_post, counts_summary = simulate_ghz_variant(**ghz_params)
    print(f"GHZ (no post) — M={M_no_post:.3f}, Pavg={pavg_no_post:.3f}, eff_avg={eff_no_post:.3f}")
    print(f"P(+,+,+)={counts_summary['p_ppp']:.5f}, P(-,-,-)={counts_summary['p_mmm']:.5f}")

    # 4) Minimal post-selection (pratique pour calibrage)
    print_header("GHZ — post-sélection minimale (analyse seulement)")
    # eps_target et max_fraction choisis pour rester raisonnables et permettre ~0.3 eff si disponible
    eps_target = 0.98
    max_fraction = 0.33
    post_res = ghz_minimal_postselection_from_raw(raw_no_post, eps_target=eps_target, max_fraction=max_fraction)
    print("Post-selection : fractions par réglage:", {k: round(v, 4) for k, v in post_res['fractions'].items()})
    print("Conditioned E per setting:", {k: round(v, 4) for k, v in post_res['E_cond'].items()})
    print(f"M_postselected = {post_res['M_post']:.4f}, overall_efficiency = {post_res['overall_efficiency']:.4f}")

    # Save GHZ results & raw counts to JSON (compact)
    summary = {
        'tag': NOW_TAG,
        'seed': int(SEED),
        'timestamp': time.time(),
        'chsh': chsh_results,
        'double_slit': {
            'S2': {'params': s2_params, 'Imax': Imax_s2, 'Imin': Imin_s2, 'V': V_s2, 'fringes': fr_s2},
            'S1': {'params': s1_params, 'Imax': Imax_s1, 'Imin': Imin_s1, 'V': V_s1, 'fringes': fr_s1}
        },
        'ghz': {
            'params': ghz_params,
            'no_post': {'M': M_no_post, 'pavg': pavg_no_post, 'eff': eff_no_post, 'counts_summary': counts_summary},
            'post_selection': post_res
        }
    }
    save_json(summary, os.path.join(RESULTS_DIR, f"summary_{NOW_TAG}.json"))

    # Also save a lightweight CSV for GHZ key numbers
    ghz_csv = [[ghz_params['alpha'], ghz_params['sigma_common'], ghz_params['sigma_delta'], ghz_params['N'],
                M_no_post, pavg_no_post, eff_no_post, counts_summary['p_ppp'], counts_summary['p_mmm'],
                post_res['M_post'], post_res['overall_efficiency']]]
    save_csv_rows(['alpha_beta', 'sigma_common', 'sigma_delta', 'N', 'M_no_post', 'pavg', 'eff_no_post', 'p_ppp', 'p_mmm',
                   'M_post', 'post_eff'], ghz_csv, os.path.join(RESULTS_DIR, f"ghz_summary_{NOW_TAG}.csv"))

    t1 = time.time()
    print_header("FIN")
    print(f"Durée totale: {t1 - t0:.1f}s")
    print("Fichiers sauvegardés dans :", os.path.abspath(RESULTS_DIR))
    print("Résumé JSON :", os.path.join(RESULTS_DIR, f"summary_{NOW_TAG}.json"))
    print("=" * 60)


if __name__ == "__main__":
    main()
