# ghz_full_improved.py
# Script complet : CHSH, double-slit S2/S1 optimisées,
# GHZ enhanced (no post-selection), GHZ minimal post-selection comparator,
# and reproduction grid search.
#
# Dépendances : numpy
# Usage : python ghz_full_improved.py

import numpy as np
import time

# reproducibility
np.random.seed(42)

# ========================================
# CHSH (Bell) Simulation - unchanged logic
# ========================================
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
    return S, avg_pa, avg_pb

# ========================================
# Double-Slit S2 (Bloch) - simulate + optimizer
# ========================================
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
    else:  # moderate_bias Beta(2,5)
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

def optimize_double_slit_s2(distribution='uniform', N=50000):
    # coarse grid search (fast) - adjust ranges if you want fine tuning
    gamma_list = [2.5*np.pi, 3.0*np.pi, 3.5*np.pi]
    k_list = [0.7, 0.9, 1.0, 1.1]
    g_list = [0.35, 0.45, 0.55]
    best = None
    for gamma in gamma_list:
        for k in k_list:
            for g in g_list:
                Imax, Imin, V, fr = simulate_double_slit_s2(distribution=distribution, N=N, gamma=gamma, k=k, g=g)
                if best is None or V > best[0]:
                    best = (V, gamma, k, g, Imax, Imin, fr)
    return best

# ========================================
# Double-Slit S1 (Poincaré) - simulate + optimizer
# ========================================
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

def optimize_double_slit_s1(distribution='uniform', N=50000):
    gamma_list = [4.0*np.pi, 5.0*np.pi, 6.0*np.pi]
    k_list = [0.3, 0.4, 0.55, 0.7]
    noise_list = [np.pi/4, np.pi/3, np.pi/2]
    best = None
    for gamma in gamma_list:
        for k in k_list:
            for noise in noise_list:
                Imax, Imin, V, fr = simulate_double_slit_s1(distribution=distribution, N=N, gamma=gamma, k=k, noise_sigma=noise)
                if best is None or V > best[0]:
                    best = (V, gamma, k, noise, Imax, Imin, fr)
    return best

# ========================================
# GHZ enhanced (NO post-selection) - improved control
# - Allows Beta shapes to concentrate the base distribution
# - Stronger common component vs tiny individual noise
# ========================================
def simulate_ghz_enhanced(model='S2', alpha=5.0, beta=5.0,
                          sigma_common=0.002, sigma_delta=0.0005, N=100000):
    """
    model: 'S2' (Bloch vector) or 'S1' (angles)
    alpha,beta: Beta distribution parameters for base (default concentrated Beta(5,5))
    sigma_common: std of common perturbation (eta)
    sigma_delta: std of per-particle noise
    returns: (M, p_avg, eff_avg, raw_data)
    raw_data holds per-setting arrays for possible post-selection
    """
    if model == 'S2':
        # base distribution concentrated via Beta(alpha,beta)
        z = np.random.beta(alpha, beta, N)
        cos_t = 2*z - 1
        theta = np.arccos(cos_t)
        phi = 2*np.pi*np.random.uniform(0,1,N)
        sin_t = np.sin(theta)
        base = np.column_stack((sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t))

        # common fluctuation and small independent deltas
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

        # secondary randoms
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
        # S1 angle-based
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
    return float(M), float(p_avg), float(eff_avg), raw_data

# ========================================
# GHZ minimal post-selection routine (automatic/light)
# ========================================
def ghz_minimal_postselection_from_raw(raw_data, eps_target=0.99, max_fraction=0.1):
    """
    raw_data: dict with keys 'xxx','xyy','yxy','yyx' containing tuples
              (A1,A2,A3,p1,p2,p3) arrays for N trials.
    eps_target: desired absolute mean for conditioned product (e.g. 0.99)
    max_fraction: maximum allowed fraction to select per setting (safety)
    Returns dict with fractions, conditioned E, M_post, overall_efficiency.
    """
    expected_signs = {'xxx': 1, 'xyy': -1, 'yxy': -1, 'yyx': -1}
    N = raw_data['xxx'][0].shape[0]
    fractions = {}
    E_cond = {}
    selected_indices = {}

    def product_and_scores(arr_tuple, expected):
        A1, A2, A3, p1, p2, p3 = arr_tuple
        prod = A1 * A2 * A3
        # Score: probability that product == expected under independent Bernoulli approx
        if expected == 1:
            score = p1 * p2 * p3
        else:
            # probability product == -1 (odd number of -1s)
            a = (1-p1)*(1-p2)*(1-p3)
            b = (1-p1)*p2*p3
            c = p1*(1-p2)*p3
            d = p1*p2*(1-p3)
            score = a + b + c + d
        return prod, score

    for key in ['xxx','xyy','yxy','yyx']:
        prod_arr, score = product_and_scores(raw_data[key], expected=expected_signs[key])
        idx_sorted = np.argsort(-score)  # descending by score
        prod_sorted = prod_arr[idx_sorted]

        # Try increasing fractions until conditioned mean reaches eps_target (or reach max_fraction)
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
            # fallback: select trials where product equals expected_sign (strict)
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

    # compute final conditioned E and M_post
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

# ========================================
# Small grid to try to reproduce post-selected behavior WITHOUT post-selection
# ========================================
def reproduce_without_postselection_grid(model='S2', alpha=5.0, beta=5.0, N=80000):
    sigma_common_list = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
    sigma_delta_list = [0.0, 0.0001, 0.0005, 0.001, 0.002]
    best = None
    for sc in sigma_common_list:
        for sd in sigma_delta_list:
            M, p_avg, eff_avg, _ = simulate_ghz_enhanced(model=model, alpha=alpha, beta=beta,
                                                         sigma_common=sc, sigma_delta=sd, N=N)
            if best is None or M > best[0]:
                best = (M, sc, sd, p_avg, eff_avg)
    return best

# ========================================
# Main : run the battery of tests and print results
# ========================================
def format_time(t): return time.strftime("%H:%M:%S", time.localtime(t))

if __name__ == "__main__":
    start = time.time()
    print("=== RUN START:", format_time(start))

    # CHSH
    print("\nCHSH (Bell) RESULTS:")
    for pa, pb in [(0.5,0.5), (0.4,0.6), (0.3,0.7), (0.35,0.35)]:
        S, pa_avg, pb_avg = compute_S(pa, pb, N=100000)
        print(f"pa={pa}, pb={pb}: |S|={S:.3f}, P(A+)={pa_avg:.3f}, P(B+)={pb_avg:.3f}")

    # Double-slit S2
    print("\nDOUBLE-SLIT S2 baseline & optimization:")
    Imax_u, Imin_u, V_u, fr_u = simulate_double_slit_s2('uniform', N=100000, gamma=3*np.pi, k=1.1, g=0.5)
    print(f"S2 uniform baseline: Imax={Imax_u:.3f}, Imin={Imin_u:.3f}, V={V_u:.3f}, Fringes={fr_u}")
    best_s2 = optimize_double_slit_s2(distribution='uniform', N=40000)
    print("S2 best found (grid): V={:.3f}, gamma={:.3f}π, k={}, g={}".format(best_s2[0], best_s2[1]/np.pi, best_s2[2], best_s2[3]))
    print(" --> Imax, Imin, fringes =", best_s2[4], best_s2[5], best_s2[6])

    # Double-slit S1
    print("\nDOUBLE-SLIT S1 baseline & optimization:")
    Imax1_u, Imin1_u, V1_u, fr1_u = simulate_double_slit_s1('uniform', N=100000, gamma=5*np.pi, k=0.551, noise_sigma=np.pi/3)
    print(f"S1 uniform baseline: Imax={Imax1_u:.3f}, Imin={Imin1_u:.3f}, V={V1_u:.3f}, Fringes={fr1_u}")
    best_s1 = optimize_double_slit_s1(distribution='uniform', N=40000)
    print("S1 best found (grid): V={:.3f}, gamma={:.3f}π, k={}, noise_sigma={:.3f}".format(best_s1[0], best_s1[1]/np.pi, best_s1[2], best_s1[3]))
    print(" --> Imax, Imin, fringes =", best_s1[4], best_s1[5], best_s1[6])

    # GHZ enhanced no post-selection: some recommended test points
    print("\nGHZ enhanced (no post-selection) - recommended parameter tests:")
    test_params = [
        # more concentrated base, tiny delta => stronger correlations
        {'alpha':5.0, 'beta':5.0, 'sigma_common':0.0005, 'sigma_delta':0.0001},
        {'alpha':5.0, 'beta':5.0, 'sigma_common':0.001,  'sigma_delta':0.0001},
        {'alpha':5.0, 'beta':5.0, 'sigma_common':0.002,  'sigma_delta':0.0005},
        {'alpha':10.0,'beta':10.0,'sigma_common':0.002,  'sigma_delta':0.0002},
        {'alpha':10.0,'beta':10.0,'sigma_common':0.0005, 'sigma_delta':0.00005}
    ]
    for p in test_params:
        M, pavg, eff, raw = simulate_ghz_enhanced('S2', alpha=p['alpha'], beta=p['beta'],
                                                 sigma_common=p['sigma_common'], sigma_delta=p['sigma_delta'], N=100000)
        print("alpha={},beta={},sc={},sd={} -> M={:.3f}, Pavg={:.3f}, eff={:.3f}".format(
            p['alpha'], p['beta'], p['sigma_common'], p['sigma_delta'], M, pavg, eff
        ))

    # Use one run to test minimal post-selection
    print("\nGHZ: minimal post-selection tests (try eps_target and max_fraction variations):")
    # perform a tuned run (concentrated base but not extreme) to get raw data
    M_run, p_run, eff_run, raw_run = simulate_ghz_enhanced('S2', alpha=5.0, beta=5.0, sigma_common=0.002, sigma_delta=0.0005, N=120000)
    print("Raw run (no post): M={:.3f}, Pavg={:.3f}, eff={:.3f}".format(M_run, p_run, eff_run))
    for eps in [0.95, 0.98, 0.99, 0.995]:
        for maxf in [0.01, 0.05, 0.1]:
            post_res = ghz_minimal_postselection_from_raw(raw_run, eps_target=eps, max_fraction=maxf)
            print(f"eps={eps}, maxf={maxf} -> M_post={post_res['M_post']:.3f}, overall_eff={post_res['overall_efficiency']:.4f}")

    # Try to reproduce WITHOUT post-selection (small grid)
    print("\nAttempt to reproduce post-selected M without post-selection (grid search):")
    best_repro = reproduce_without_postselection_grid(model='S2', alpha=5.0, beta=5.0, N=80000)
    print("Best no-post found: M={:.3f}, sigma_common={}, sigma_delta={}, Pavg={:.3f}, effavg={:.3f}".format(
        best_repro[0], best_repro[1], best_repro[2], best_repro[3], best_repro[4]
    ))

    end = time.time()
    print("\n=== RUN END:", format_time(end), " elapsed: {:.1f}s".format(end - start))
    print("\nNotes / recommandations rapides:")
    print("- Si tu veux pousser M sans post-sélection, concentre encore plus la distribution de base (Beta large)")
    print("- Réduis sigma_delta à presque zéro et garde sigma_common non nul, ou augmente la taille N pour meilleure précision")
    print("- Pour une post-sélection 'légère' vise overall_eff >= 0.01 (1%) ou 0.05 (5%) en choisissant eps_target un peu plus faible (0.98) et max_fraction plus grand")
    print("- Dis-moi si tu veux que je rende ce script plus agressif (grilles plus fines, N plus grand) ou que je génère CSV / plots / CI yaml pour GitHub Actions.")
