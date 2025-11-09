# ghz_comparison.py
# Script complet : CHSH, double-slit S2/S1, GHZ (sans post-selection enhanced),
# GHZ avec post-selection minimale (auto-calculée), et tentative de reproduction sans post-selection.
#
# Usage : python ghz_comparison.py
#
import numpy as np

# Seed global pour reproductibilité exacte
np.random.seed(42)

# ========================================
# CHSH (Bell) Simulation - identique
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

    def compute_E(delta, N, pa, pb):
        probs = get_probs(delta)
        choice_list = ['pp','mm','pm','mp']
        sum_ab = 0.0
        count_ap = 0
        count_bp = 0
        for _ in range(N):
            outcome = np.random.choice(choice_list, p=probs)
            if outcome == 'pp': A, B = 1, 1
            elif outcome == 'mm': A, B = -1, -1
            elif outcome == 'pm': A, B = 1, -1
            else: A, B = -1, 1

            f_a_up = 0.0
            f_a_down = 0.0
            if pa < 0.5: f_a_down = 1 - 2*pa
            elif pa > 0.5: f_a_up = 2*pa - 1
            A = flip_outcome(A, f_a_up, f_a_down)

            f_b_up = 0.0
            f_b_down = 0.0
            if pb < 0.5: f_b_down = 1 - 2*pb
            elif pb > 0.5: f_b_up = 2*pb - 1
            B = flip_outcome(B, f_b_up, f_b_down)

            sum_ab += A * B
            if A == 1: count_ap += 1
            if B == 1: count_bp += 1

        return sum_ab/N, count_ap/N, count_bp/N

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
# Double-Slit S2 (Bloch) - fonctionnelle
# ========================================
def simulate_double_slit_s2(distribution='uniform', N=100000, gamma=3*np.pi, k=1.1, g=0.5):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros(100)
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
        p1 = np.array([np.cos(alpha), np.sin(alpha), 0])
        p2 = np.array([np.cos(alpha+delta), np.sin(alpha+delta), 0])
        dot1 = lambd_x*p1[0] + lambd_y*p1[1] + lambd_z*p1[2]
        dot2 = lambd_x*p2[0] + lambd_y*p2[1] + lambd_z*p2[2]
        delta_phi = g*k*(dot1 - dot2) + gamma*np.sin(alpha)
        I[i] = 1 + np.mean(np.cos(delta_phi))

    Imax, Imin = np.max(I), np.min(I)
    V = (Imax - Imin)/(Imax + Imin)
    fringes = round(gamma / np.pi)
    return Imax, Imin, V, fringes

# ========================================
# Double-Slit S1 (Poincaré)
# ========================================
def simulate_double_slit_s1(distribution='uniform', N=100000, gamma=5*np.pi, k=0.551, noise_sigma=np.pi/3):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros(100)
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

    Imax, Imin = np.max(I), np.min(I)
    V = (Imax - Imin)/(Imax + Imin)
    fringes = round(gamma / np.pi)
    return Imax, Imin, V, fringes

# ========================================
# GHZ enhanced (NO post-selection) - version améliorée
# ========================================
def simulate_ghz_enhanced(model='S2', alpha=0.5, beta=0.5,
                          sigma_common=0.02, sigma_delta=0.005, N=100000):
    """
    Retourne (M, p_avg, efficiency_average)
    sigma_common : variance composante commune (eta)
    sigma_delta  : variance bruit individuel
    """
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
            p1 = np.clip(R1/4.0 + 0.5, 0, 1)
            A1 = (np.random.random(N) < p1).astype(int) * 2 - 1

            s1_2 = sign_dot(base, n2)
            s2_2 = sign_dot(l2, n2)
            s3_2 = sign_dot(l2pp, n2)
            R2 = s1_2 * s2_2 + s3_2
            p2 = np.clip(R2/4.0 + 0.5, 0, 1)
            A2 = (np.random.random(N) < p2).astype(int) * 2 - 1

            s1_3 = sign_dot(base, n3)
            s2_3 = sign_dot(l3, n3)
            s3_3 = sign_dot(l3pp, n3)
            R3 = s1_3 * s2_3 + s3_3
            p3 = np.clip(R3/4.0 + 0.5, 0, 1)
            A3 = (np.random.random(N) < p3).astype(int) * 2 - 1

            product = A1 * A2 * A3
            E = np.mean(product)
            eff = np.mean((A1 == 1) & (A2 == 1) & (A3 == 1))
            p_avg = (np.mean(p1) + np.mean(p2) + np.mean(p3)) / 3.0
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
            p1 = np.clip(R1/4.0 + 0.5, 0, 1)
            A1 = (np.random.random(N) < p1).astype(int) * 2 - 1

            s1_2 = sign_cos(n2, lambda_angle)
            s2_2 = sign_cos(n2, l2)
            s3_2 = sign_cos(n2, l2pp)
            R2 = s1_2 * s2_2 + s3_2
            p2 = np.clip(R2/4.0 + 0.5, 0, 1)
            A2 = (np.random.random(N) < p2).astype(int) * 2 - 1

            s1_3 = sign_cos(n3, lambda_angle)
            s2_3 = sign_cos(n3, l3)
            s3_3 = sign_cos(n3, l3pp)
            R3 = s1_3 * s2_3 + s3_3
            p3 = np.clip(R3/4.0 + 0.5, 0, 1)
            A3 = (np.random.random(N) < p3).astype(int) * 2 - 1

            product = A1 * A2 * A3
            E = np.mean(product)
            eff = np.mean((A1 == 1) & (A2 == 1) & (A3 == 1))
            p_avg = (np.mean(p1) + np.mean(p2) + np.mean(p3)) / 3.0
            return E, p_avg, eff, (A1, A2, A3, p1, p2, p3)

        E_xxx, p_xxx, eff_xxx, data_xxx = get_outcomes(x, x, x)
        E_xyy, p_xyy, eff_xyy, data_xyy = get_outcomes(x, y, y)
        E_yxy, p_yxy, eff_yxy, data_yxy = get_outcomes(y, x, y)
        E_yyx, p_yyx, eff_yyx, data_yyx = get_outcomes(y, y, x)

    # M and average p/eff:
    M = abs(E_xyy + E_yxy + E_yyx - E_xxx)
    p_avg = (p_xxx + p_xyy + p_yxy + p_yyx) / 4.0
    eff_avg = (eff_xxx + eff_xyy + eff_yxy + eff_yyx) / 4.0

    # Return also raw data for potential post-selection processing
    raw_data = {
        'xxx': data_xxx,
        'xyy': data_xyy,
        'yxy': data_yxy,
        'yyx': data_yyx
    }

    return M, p_avg, eff_avg, raw_data

# ========================================
# GHZ with post-selection minimale automatique
# ========================================
def ghz_minimal_postselection_from_raw(raw_data, eps_target=0.999, max_fraction=0.9):
    """
    raw_data: dict with keys 'xxx','xyy','yxy','yyx' each mapping to tuple
              (A1,A2,A3,p1,p2,p3) arrays for N trials.
    eps_target: target conditioned mean close to expected_sign (>= eps_target * expected_sign)
    max_fraction: do not allow selecting more than this fraction per setting (safety)
    Returns:
      results dict containing per-setting selection fractions, conditioned E, and M_postselected
    Behavior:
      For each setting, compute a "score" estimating the probability the product equals expected_sign:
        score = prod_i (p_i if expected_sign==1 else (1 - p_i))
      Then sort trials by score descending and find minimal top-k fraction such that conditioned mean(product[top_k]) 
      >= eps_target (for expected_sign=1) or <= -eps_target (for expected_sign=-1).
      If not achievable with fraction <= max_fraction, fall back to mask product==expected_sign.
    """
    expected_signs = {'xxx': 1, 'xyy': -1, 'yxy': -1, 'yyx': -1}
    N = raw_data['xxx'][0].shape[0]
    E_cond = {}
    fractions = {}
    selected_indices = {}

    # helper to compute product array
    def product_from_arrays(arr_tuple):
        A1, A2, A3, p1, p2, p3 = arr_tuple
        return A1 * A2 * A3, p1, p2, p3

    for key in ['xxx','xyy','yxy','yyx']:
        prod_arr, p1, p2, p3 = product_from_arrays(raw_data[key])
        expected = expected_signs[key]
        # compute score estimating chance that product == expected (based on p's)
        if expected == 1:
            score = p1 * p2 * p3
        else:
            # chance product == -1 roughly when one or three of the Ai are -1;
            # approximate by (1-p1)*(p2)*(p3) + ... but simpler: use
            # score = (1-p1)*(1-p2)*(1-p3) + (1-p1)*p2*p3 + p1*(1-p2)*p3 + p1*p2*(1-p3)
            # This equals probability that product = -1 under independent Bernoulli approx
            a = (1-p1)*(1-p2)*(1-p3)
            b = (1-p1)*p2*p3
            c = p1*(1-p2)*p3
            d = p1*p2*(1-p3)
            score = a + b + c + d

        # sort by score descending
        idx_sorted = np.argsort(-score)  # descending
        prod_sorted = prod_arr[idx_sorted]

        # try cumulative top-k to reach eps_target
        achieved = False
        # convert target to inequality depending on expected sign
        # For expected 1: want mean(prod_selected) >= eps_target
        # For expected -1: want mean(prod_selected) <= -eps_target
        # We'll iterate over fractions to find minimal fraction
        fractions_to_try = np.concatenate((np.linspace(0.001, 0.01, 10),
                                           np.linspace(0.01, 0.1, 10),
                                           np.linspace(0.1, max_fraction, 20)))
        for frac in fractions_to_try:
            k = max(1, int(np.ceil(frac * N)))
            sel_mean = np.mean(prod_sorted[:k])
            if expected == 1 and sel_mean >= eps_target:
                fractions[key] = k / N
                selected_indices[key] = idx_sorted[:k]
                E_cond[key] = sel_mean
                achieved = True
                break
            if expected == -1 and sel_mean <= -eps_target:
                fractions[key] = k / N
                selected_indices[key] = idx_sorted[:k]
                E_cond[key] = sel_mean
                achieved = True
                break
        if not achieved:
            # fallback: trivial selection where product == expected
            mask = (prod_arr == expected)
            k = np.sum(mask)
            if k == 0:
                # can't select any -> choose no selection (E conditional is NaN). Use mean original
                fractions[key] = 0.0
                selected_indices[key] = np.array([], dtype=int)
                E_cond[key] = np.mean(prod_arr)  # not good, but fallback
            else:
                fractions[key] = k / N
                selected_indices[key] = np.where(mask)[0]
                E_cond[key] = np.mean(prod_arr[mask])

    # Build final conditioned E values for all settings and compute M_postselected
    E_values = {}
    for key in ['xxx','xyy','yxy','yyx']:
        idx = selected_indices.get(key, np.array([], dtype=int))
        prod_arr = product_from_arrays(raw_data[key])[0]
        if idx.size == 0:
            # no selection -> use original mean
            E_values[key] = float(np.mean(prod_arr))
        else:
            E_values[key] = float(np.mean(prod_arr[idx]))

    M_post = abs(E_values['xyy'] + E_values['yxy'] + E_values['yyx'] - E_values['xxx'])
    # overall efficiency = average selected fraction
    overall_eff = np.mean([fractions.get(k, 0.0) for k in ['xxx','xyy','yxy','yyx']])

    return {
        'fractions': fractions,
        'E_cond': E_values,
        'M_post': M_post,
        'overall_efficiency': overall_eff
    }

# ========================================
# Tentative reproduction WITHOUT post-selection:
# grid search over sigma_common / sigma_delta to maximize M (no post-selection)
# ========================================
def reproduce_without_postselection_grid(model='S2', alpha=0.5, beta=0.5, N=80000):
    # small grid (kept coarse to be fast)
    sigma_common_list = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08]
    sigma_delta_list = [0.0, 0.0005, 0.001, 0.005, 0.01]
    best = None
    for sc in sigma_common_list:
        for sd in sigma_delta_list:
            M, p_avg, eff_avg, _ = simulate_ghz_enhanced(model=model, alpha=alpha, beta=beta,
                                                         sigma_common=sc, sigma_delta=sd, N=N)
            if best is None or M > best[0]:
                best = (M, sc, sd, p_avg, eff_avg)
    return best

# ========================================
# Run everything and print results (complete)
# ========================================
if __name__ == "__main__":
    # 1) CHSH quick run (as you did)
    print("CHSH (Bell) RESULTS:")
    for pa, pb in [(0.5,0.5), (0.4,0.6), (0.3,0.7), (0.35,0.35)]:
        S, pa_avg, pb_avg = compute_S(pa, pb)
        print(f"pa={pa}, pb={pb}: |S|={S:.3f}, P(A+)={pa_avg:.3f}, P(B+)={pb_avg:.3f}")

    # 2) Double-slit baseline + small optimizations
    print("\nDOUBLE-SLIT S2 baseline & optimization:")
    Imax_u, Imin_u, V_u, fr_u = simulate_double_slit_s2('uniform', N=100000)
    print(f"S2 uniform baseline: Imax={Imax_u:.3f}, Imin={Imin_u:.3f}, V={V_u:.3f}, Fringes={fr_u}")
    # small grid search (kept small)
    best_s2 = None
    gamma_list = [2.5*np.pi, 3.0*np.pi, 3.5*np.pi]
    k_list = [0.9, 1.0, 1.1, 1.2]
    g_list = [0.4, 0.5, 0.6]
    for gamma in gamma_list:
        for k in k_list:
            for g in g_list:
                Imax_t, Imin_t, V_t, fr_t = simulate_double_slit_s2('uniform', N=40000, gamma=gamma, k=k, g=g)
                if best_s2 is None or V_t > best_s2[0]:
                    best_s2 = (V_t, gamma, k, g, Imax_t, Imin_t, fr_t)
    print("S2 best found (small grid): V={:.3f}, gamma={:.3f}π, k={}, g={}".format(best_s2[0], best_s2[1]/np.pi, best_s2[2], best_s2[3]))
    print(" --> Imax, Imin, fringes =", best_s2[4], best_s2[5], best_s2[6])

    print("\nDOUBLE-SLIT S1 baseline & optimization:")
    Imax1_u, Imin1_u, V1_u, fr1_u = simulate_double_slit_s1('uniform', N=100000, noise_sigma=np.pi/3)
    print(f"S1 uniform baseline: Imax={Imax1_u:.3f}, Imin={Imin1_u:.3f}, V={V1_u:.3f}, Fringes={fr1_u}")
    # small grid search for S1
    best_s1 = None
    gamma_list_s1 = [4.0*np.pi, 5.0*np.pi, 6.0*np.pi]
    k_list_s1 = [0.3, 0.5, 0.55, 0.7]
    noise_list = [np.pi/4, np.pi/3, np.pi/2]
    for gamma in gamma_list_s1:
        for k in k_list_s1:
            for noise in noise_list:
                Imax_t, Imin_t, V_t, fr_t = simulate_double_slit_s1('uniform', N=40000, gamma=gamma, k=k, noise_sigma=noise)
                if best_s1 is None or V_t > best_s1[0]:
                    best_s1 = (V_t, gamma, k, noise, Imax_t, Imin_t, fr_t)
    print("S1 best found (small grid): V={:.3f}, gamma={:.3f}π, k={}, noise_sigma={:.3f}".format(best_s1[0], best_s1[1]/np.pi, best_s1[2], best_s1[3]))
    print(" --> Imax, Imin, fringes =", best_s1[4], best_s1[5], best_s1[6])

    # 3) GHZ enhanced (no post-selection) sweep (small)
    print("\nGHZ enhanced - sweep sigma_common (keep sigma_delta small):")
    for sigma_common in [0.0, 0.005, 0.01, 0.02, 0.05]:
        M, p_avg, terms, raw = simulate_ghz_enhanced('S2', sigma_common=sigma_common, sigma_delta=0.001, N=100000)
        print(f"sigma_common={sigma_common:.3f} | M={M:.3f}, P(±)={p_avg:.3f}, eff={terms:.3f}")

    print("\nGHZ enhanced - sweep sigma_delta (keep sigma_common moderate):")
    for sigma_delta in [0.000, 0.001, 0.005, 0.01, 0.02]:
        M, p_avg, terms, raw = simulate_ghz_enhanced('S2', sigma_common=0.02, sigma_delta=sigma_delta, N=100000)
        print(f"sigma_delta={sigma_delta:.4f} | M={M:.3f}, P(±)={p_avg:.3f}, eff={terms:.3f}")

    print("\nGHZ tuned attempt (strong common, tiny individual):")
    M_tuned, p_tuned, eff_tuned, raw_tuned = simulate_ghz_enhanced('S2', sigma_common=0.08, sigma_delta=0.0005, N=150000)
    print(f"Tuned S2: M={M_tuned:.3f}, P(±)={p_tuned:.3f}, eff={eff_tuned:.3f}")

    # 4) GHZ with minimal post-selection computed from raw_tuned (or raw from a run)
    # Use raw_tuned if available, else fallback to a fresh run
    print("\nGHZ POST-SELECTION minimal attempt (from tuned run raw data):")
    raw_for_post = raw_tuned if 'raw_tuned' in locals() else simulate_ghz_enhanced('S2', sigma_common=0.02, sigma_delta=0.001, N=150000)[3]
    post_results = ghz_minimal_postselection_from_raw(raw_for_post, eps_target=0.999, max_fraction=0.9)
    print("Per-setting selected fractions:", {k: round(v,4) for k,v in post_results['fractions'].items()})
    print("Conditioned E values (post-selected):", {k: round(v,4) for k,v in post_results['E_cond'].items()})
    print("M_postselected =", round(post_results['M_post'],4))
    print("Overall average selection fraction (efficiency) =", round(post_results['overall_efficiency'],4))

    # 5) Try to reproduce the post-selected M WITHOUT post-selection (small grid search)
    print("\nTentative reproduction WITHOUT post-selection (small grid search):")
    best_repro = reproduce_without_postselection_grid(model='S2', N=80000)
    print("Best found (no post-selection): M={:.3f}, sigma_common={}, sigma_delta={}, Pavg={:.3f}, effavg={:.3f}".format(
        best_repro[0], best_repro[1], best_repro[2], best_repro[3], best_repro[4]
    ))

    # Summary message
    print("\n--- FINISHED: GHZ comparison (no-post vs minimal post-selection) ---")
    print("Note: post-selection can trivially reach M≈4 by selecting trials with product == expected_sign.")
    print("The minimal-postselection routine above attempts to find a light selection per setting that yields E≈±1.")
    print("If you want a stricter 'minimal' criterion (e.g., maximize M while keeping average fraction >= X%), tell me X% and j'adapte.")
