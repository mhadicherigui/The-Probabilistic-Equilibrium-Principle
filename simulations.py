# ghz_experiment_suite.py
# Suite d'expériences complète (Option B - sampling variants only)
# - CHSH (inchangé)
# - Double-slit S2 & S1 : scan gamma + optimisation locale
# - GHZ : grid over (alpha_beta, sigma_common, sigma_delta, N) WITHOUT changing rules
# - Computes P(+,+,+) and P(-,-,-) (analysis only)
# - Minimal post-selection comparator
# - Saves CSV and JSON in results/
#
# Usage: python ghz_experiment_suite.py
# Dépendances: numpy (pip install numpy)

import numpy as np
import os
import csv
import json
import time
from datetime import datetime

np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------
# Utility helpers
# -----------------------
def now_tag():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def save_csv(rows, header, filename):
    with open(filename, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

# -----------------------
# CHSH - unchanged
# -----------------------
def compute_S(pa, pb, N=100000):
    def get_probs(delta_deg):
        delta = np.deg2rad(delta_deg)
        c2 = np.cos(delta)**2
        s2 = np.sin(delta)**2
        return [0.5*c2, 0.5*c2, 0.5*s2, 0.5*s2]

    def flip_outcome(out, f_up, f_down):
        if out == 1 and np.random.random() < f_down:
            return -1
        elif out == -1 and np.random.random() < f_up:
            return 1
        return out

    def compute_E(delta, Nlocal, pa_local, pb_local):
        probs = get_probs(delta)
        choice_list = ['pp','mm','pm','mp']
        sum_ab = 0.0
        count_ap = 0
        count_bp = 0
        for _ in range(Nlocal):
            outcome = np.random.choice(choice_list, p=probs)
            if outcome == 'pp': A, B = 1, 1
            elif outcome == 'mm': A, B = -1, -1
            elif outcome == 'pm': A, B = 1, -1
            else: A, B = -1, 1

            f_a_up = 0.0
            f_a_down = 0.0
            if pa_local < 0.5: f_a_down = 1 - 2*pa_local
            elif pa_local > 0.5: f_a_up = 2*pa_local - 1
            A = flip_outcome(A, f_a_up, f_a_down)

            f_b_up = 0.0
            f_b_down = 0.0
            if pb_local < 0.5: f_b_down = 1 - 2*pb_local
            elif pb_local > 0.5: f_b_up = 2*pb_local - 1
            B = flip_outcome(B, f_b_up, f_b_down)

            sum_ab += A * B
            if A == 1: count_ap += 1
            if B == 1: count_bp += 1

        return sum_ab/Nlocal, count_ap/Nlocal, count_bp/Nlocal

    deltas = {'ab':22.5, 'abp':67.5, 'apb':-22.5, 'apbp':22.5}
    E1, pa1, pb1 = compute_E(deltas['ab'], N, pa, pb)
    E2, pa2, pb2 = compute_E(deltas['abp'], N, pa, pb)
    E3, pa3, pb3 = compute_E(deltas['apb'], N, pa, pb)
    E4, pa4, pb4 = compute_E(deltas['apbp'], N, pa, pb)
    S = abs(E1 + E3 + E4 - E2)
    avg_pa = (pa1 + pa2 + pa3 + pa4) / 4
    avg_pb = (pb1 + pb2 + pb3 + pb4) / 4
    # std error estimate for S (approx)
    var_terms = [(1 - E1**2), (1 - E2**2), (1 - E3**2), (1 - E4**2)]
    SE_S = np.sqrt(sum(var_terms)) / np.sqrt(N)
    return S, avg_pa, avg_pb, SE_S

# -----------------------
# Double-slit S2 / S1
# -----------------------
def simulate_double_slit_s2(distribution='uniform', N=100000, gamma=3*np.pi, k=1.1, g=0.5):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros_like(alphas)
    delta = np.pi/2

    if distribution == 'uniform':
        u = np.random.uniform(0,1,N)
        v = np.random.uniform(0,1,N)
        theta = np.arccos(1 - 2*u)
        phi = 2*np.pi*v
        lambd_x = np.sin(theta) * np.cos(phi)
        lambd_y = np.sin(theta) * np.sin(phi)
        lambd_z = np.cos(theta)
    else:
        z = np.random.beta(2,5,N)
        cos_theta = 2*z - 1
        theta = np.arccos(cos_theta)
        phi = 2*np.pi*np.random.uniform(0,1,N)
        lambd_x = np.sin(theta)*np.cos(phi)
        lambd_y = np.sin(theta)*np.sin(phi)
        lambd_z = cos_theta

    for i, alpha in enumerate(alphas):
        p1 = np.array([np.cos(alpha), np.sin(alpha), 0.0])
        p2 = np.array([np.cos(alpha+delta), np.sin(alpha+delta), 0.0])
        dot1 = lambd_x * p1[0] + lambd_y * p1[1] + lambd_z * p1[2]
        dot2 = lambd_x * p2[0] + lambd_y * p2[1] + lambd_z * p2[2]
        delta_phi = g * k * (dot1 - dot2) + gamma * np.sin(alpha)
        I[i] = 1 + np.mean(np.cos(delta_phi))

    Imax, Imin = float(np.max(I)), float(np.min(I))
    V = (Imax - Imin) / (Imax + Imin)
    fringes = int(round(float(gamma) / np.pi))
    return Imax, Imin, V, fringes

def simulate_double_slit_s1(distribution='uniform', N=100000, gamma=5*np.pi, k=0.551, noise_sigma=np.pi/3):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros_like(alphas)
    delta = np.pi/2

    if distribution == 'uniform':
        lambd = np.random.uniform(0, 2*np.pi, N)
        lambd2 = lambd.copy()
    else:
        z = np.random.beta(2,5,N)
        lambd = 2*np.pi*z
        lambd2 = (lambd + np.random.normal(0, noise_sigma, N)) % (2*np.pi)

    for i, alpha in enumerate(alphas):
        delta_phi = 2*gamma*np.sin(alpha) + k*(np.cos(alpha - lambd) - np.cos(alpha + delta - lambd2))
        I[i] = 1 + np.mean(np.cos(delta_phi))

    Imax, Imin = float(np.max(I)), float(np.min(I))
    V = (Imax - Imin) / (Imax + Imin)
    fringes = int(round(float(gamma) / np.pi))
    return Imax, Imin, V, fringes

# -----------------------
# GHZ enhanced (OPTION B - sampling changes only)
# returns (M, p_avg, eff_avg, raw_data, counts_ppp_mmm)
# raw_data: dict of tuples (A1,A2,A3,p1,p2,p3) for each setting
# counts_ppp_mmm: dict with counts for (+,+,+) and (-,-,-) across all trials and settings (for analysis)
# -----------------------
def simulate_ghz_variant(model='S2', alpha=5.0, beta=5.0, sigma_common=0.002, sigma_delta=0.0005, N=100000):
    # This function keeps the exact same rule mapping (p from R = s1*s2 + s3) as in your model.
    # ONLY sampling (Beta, sigma_common, sigma_delta, N) is changed.
    if model == 'S2':
        z = np.random.beta(alpha, beta, N)
        cos_t = 2*z - 1
        theta = np.arccos(cos_t)
        phi = 2*np.pi*np.random.uniform(0,1,N)
        sin_t = np.sin(theta)
        base = np.column_stack((sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t))

        eta_common = np.random.normal(0, sigma_common, (N,3))
        delta1 = np.random.normal(0, sigma_delta, (N,3))
        delta2 = np.random.normal(0, sigma_delta, (N,3))
        delta3 = np.random.normal(0, sigma_delta, (N,3))

        def normalize(v):
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return v / n

        l1 = normalize(base + eta_common + delta1)
        l2 = normalize(base + eta_common + delta2)
        l3 = normalize(base + eta_common + delta3)

        def sample_s2():
            z2 = np.random.beta(alpha, beta, N)
            cos_t2 = 2*z2 - 1
            theta2 = np.arccos(cos_t2)
            phi2 = 2*np.pi*np.random.uniform(0,1,N)
            sin_t2 = np.sin(theta2)
            return np.column_stack((sin_t2*np.cos(phi2), sin_t2*np.sin(phi2), cos_t2))

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
        z = np.random.beta(alpha, beta, N)
        lambda_angle = 2*np.pi*z
        eta_common = np.random.normal(0, sigma_common, N)
        delta1 = np.random.normal(0, sigma_delta, N)
        delta2 = np.random.normal(0, sigma_delta, N)
        delta3 = np.random.normal(0, sigma_delta, N)

        l1 = (lambda_angle + eta_common + delta1) % (2*np.pi)
        l2 = (lambda_angle + eta_common + delta2) % (2*np.pi)
        l3 = (lambda_angle + eta_common + delta3) % (2*np.pi)

        l1pp = (2*np.pi*np.random.beta(alpha, beta, N)) % (2*np.pi)
        l2pp = (2*np.pi*np.random.beta(alpha, beta, N)) % (2*np.pi)
        l3pp = (2*np.pi*np.random.beta(alpha, beta, N)) % (2*np.pi)

        x, y = 0.0, np.pi/2

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

    M = abs(E_xyy + E_yxy + E_yyx - E_xxx)
    p_avg = (p_xxx + p_xyy + p_yxy + p_yyx) / 4.0
    eff_avg = (eff_xxx + eff_xyy + eff_yxy + eff_yyx) / 4.0

    raw_data = {
        'xxx': data_xxx,
        'xyy': data_xyy,
        'yxy': data_yxy,
        'yyx': data_yyx
    }

    # compute P(+,+,+) and P(-,-,-) across the 4 settings aggregated (counts)
    counts_ppp = 0
    counts_mmm = 0
    total_trials = N * 4  # 4 settings each N trials
    for key in ['xxx','xyy','yxy','yyx']:
        A1, A2, A3, p1, p2, p3 = raw_data[key]
        # count per setting where all +1 or all -1
        counts_ppp += int(np.sum((A1 == 1) & (A2 == 1) & (A3 == 1)))
        counts_mmm += int(np.sum((A1 == -1) & (A2 == -1) & (A3 == -1)))
    p_ppp = counts_ppp / total_trials
    p_mmm = counts_mmm / total_trials
    counts_summary = {'ppp': counts_ppp, 'mmm': counts_mmm, 'p_ppp': p_ppp, 'p_mmm': p_mmm}
    return float(M), float(p_avg), float(eff_avg), raw_data, counts_summary

# -----------------------
# Minimal post-selection (same as earlier)
# -----------------------
def ghz_minimal_postselection_from_raw(raw_data, eps_target=0.99, max_fraction=0.1):
    expected_signs = {'xxx': 1, 'xyy': -1, 'yxy': -1, 'yyx': -1}
    N = raw_data['xxx'][0].shape[0]
    fractions = {}
    E_cond = {}
    selected_indices = {}

    def product_and_scores(arr_tuple, expected):
        A1, A2, A3, p1, p2, p3 = arr_tuple
        prod = A1 * A2 * A3
        if expected == 1:
            score = p1 * p2 * p3
        else:
            a = (1-p1)*(1-p2)*(1-p3)
            b = (1-p1)*p2*p3
            c = p1*(1-p2)*p3
            d = p1*p2*(1-p3)
            score = a + b + c + d
        return prod, score

    for key in ['xxx','xyy','yxy','yyx']:
        prod_arr, score = product_and_scores(raw_data[key], expected=expected_signs[key])
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

    E_values = {}
    for key in ['xxx','xyy','yxy','yyx']:
        idx = selected_indices.get(key, np.array([], dtype=int))
        A1, A2, A3, _, _, _ = raw_data[key]
        prod_arr = A1 * A2 * A3
        if idx.size == 0:
            E_values[key] = float(np.mean(prod_arr))
        else:
            E_values[key] = float(np.mean(prod_arr[idx]))

    M_post = abs(E_values['xyy'] + E_values['yxy'] + E_values['yyx'] - E_values['xxx'])
    overall_eff = float(np.mean([fractions.get(k, 0.0) for k in ['xxx','xyy','yxy','yyx']]))
    return {'fractions': fractions, 'E_cond': E_values, 'M_post': float(M_post), 'overall_efficiency': overall_eff}

# -----------------------
# Experiment orchestration
# -----------------------
def run_suite(output_tag=None):
    if output_tag is None:
        output_tag = now_tag()
    meta = {'tag': output_tag, 'timestamp': time.time(), 'seed': int(np.random.get_state()[1][0])}
    summary = {'meta': meta, 'chsh': None, 'double_slit': {}, 'ghz_grid': [], 'ghz_post': None}

    # 1) CHSH quick
    chsh_rows = []
    for pa, pb in [(0.5,0.5),(0.4,0.6),(0.3,0.7),(0.35,0.35)]:
        S, pa_avg, pb_avg, SE_S = compute_S(pa, pb, N=100000)
        chsh_rows.append((pa, pb, S, pa_avg, pb_avg, SE_S))
    summary['chsh'] = chsh_rows
    save_csv(chsh_rows, ['pa','pb','S','pa_emp','pb_emp','SE_S'], os.path.join(RESULTS_DIR, f"chsh_{output_tag}.csv"))

    # 2) Double-slit gamma scans (S2 and S1)
    ds_rows = []
    gamma_values = [2.5*np.pi, 3.0*np.pi, 3.5*np.pi, 4.0*np.pi, 4.5*np.pi, 5.0*np.pi, 5.5*np.pi, 6.0*np.pi]
    for model in ['S2','S1']:
        for gamma in gamma_values:
            if model == 'S2':
                Imax, Imin, V, fr = simulate_double_slit_s2('uniform', N=50000, gamma=gamma, k=1.0, g=0.45)
            else:
                # for S1 sweep use moderate noise to keep physicality
                Imax, Imin, V, fr = simulate_double_slit_s1('uniform', N=50000, gamma=gamma, k=0.4, noise_sigma=0.5)
            ds_rows.append((model, gamma/np.pi, Imax, Imin, V, fr))
    summary['double_slit']['scan'] = ds_rows
    save_csv(ds_rows, ['model','gamma_pi','Imax','Imin','V','fringes'], os.path.join(RESULTS_DIR, f"double_slit_gamma_scan_{output_tag}.csv"))

    # 3) GHZ grid sweep (alpha_beta, sigma_common, sigma_delta, N)
    ghz_rows = []
    alpha_beta_list = [5.0, 10.0, 20.0, 50.0]
    sigma_common_list = [0.0005, 0.001, 0.002, 0.005]
    sigma_delta_list = [1e-6, 1e-5, 1e-4, 0.001]
    N_list = [100000, 300000]  # test influence of N
    for alpha_beta in alpha_beta_list:
        for sc in sigma_common_list:
            for sd in sigma_delta_list:
                for N in N_list:
                    M, p_avg, eff, raw, counts = simulate_ghz_variant('S2', alpha=alpha_beta, beta=alpha_beta,
                                                                       sigma_common=sc, sigma_delta=sd, N=N)
                    row = {'alpha_beta': alpha_beta, 'sigma_common': sc, 'sigma_delta': sd, 'N': N,
                           'M': M, 'p_avg': p_avg, 'eff': eff,
                           'p_ppp': counts['p_ppp'], 'p_mmm': counts['p_mmm']}
                    ghz_rows.append(row)
    summary['ghz_grid'] = ghz_rows
    # save GHZ grid CSV
    ghz_csv_rows = []
    for r in ghz_rows:
        ghz_csv_rows.append([r['alpha_beta'], r['sigma_common'], r['sigma_delta'], r['N'],
                             r['M'], r['p_avg'], r['eff'], r['p_ppp'], r['p_mmm']])
    save_csv(ghz_csv_rows, ['alpha_beta','sigma_common','sigma_delta','N','M','p_avg','eff','p_ppp','p_mmm'],
             os.path.join(RESULTS_DIR, f"ghz_grid_{output_tag}.csv"))

    # 4) Pick a representative raw run for post-selection exploration:
    #    choose one with high M or mid M for exploration
    # find candidate with highest M in ghz_rows
    best = max(ghz_rows, key=lambda x: x['M'])
    # Re-run with those params to get raw_data
    raw_M, raw_pavg, raw_eff, raw_data, counts = simulate_ghz_variant('S2', alpha=best['alpha_beta'], beta=best['alpha_beta'],
                                                                      sigma_common=best['sigma_common'], sigma_delta=best['sigma_delta'], N=best['N'])
    # explore post-selection eps and max_fraction combos
    post_results = []
    for eps in [0.95, 0.98, 0.99, 0.995]:
        for maxf in [0.01, 0.05, 0.1, 0.3]:
            res = ghz_minimal_postselection_from_raw(raw_data, eps_target=eps, max_fraction=maxf)
            res_row = {'alpha_beta': best['alpha_beta'], 'sigma_common': best['sigma_common'], 'sigma_delta': best['sigma_delta'],
                       'N': best['N'], 'eps': eps, 'max_fraction': maxf, 'M_post': res['M_post'], 'overall_eff': res['overall_efficiency'],
                       'fractions': res['fractions']}
            post_results.append(res_row)
    summary['ghz_post'] = {'params_subject': best, 'post_results': post_results}
    save_json(summary, os.path.join(RESULTS_DIR, f"summary_{output_tag}.json"))

    # save post results CSV
    post_csv_rows = []
    for r in post_results:
        post_csv_rows.append([r['alpha_beta'], r['sigma_common'], r['sigma_delta'], r['N'],
                              r['eps'], r['max_fraction'], r['M_post'], r['overall_eff']])
    save_csv(post_csv_rows, ['alpha_beta','sigma_common','sigma_delta','N','eps','max_fraction','M_post','overall_eff'],
             os.path.join(RESULTS_DIR, f"ghz_post_selection_scan_{output_tag}.csv"))

    # Final prints (compact overview)
    print("\n=== SUMMARY (brief) ===")
    print("CHSH results saved -> chsh_%s.csv" % output_tag)
    print("Double-slit gamma scan saved -> double_slit_gamma_scan_%s.csv" % output_tag)
    print("GHZ parameter grid saved -> ghz_grid_%s.csv" % output_tag)
    print("GHZ post-selection scan saved -> ghz_post_selection_scan_%s.csv" % output_tag)
    print("Full JSON summary -> summary_%s.json" % output_tag)
    return summary

# -----------------------
# Execute suite
# -----------------------
if __name__ == "__main__":
    tag = now_tag()
    t0 = time.time()
    print("Starting GHZ experiment suite (Option B) -- tag:", tag)
    summary = run_suite(output_tag=tag)
    t1 = time.time()
    print("Finished in %.1f s" % (t1 - t0))
    print("Results saved in folder:", RESULTS_DIR)
