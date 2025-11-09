# improved_simulations.py
import numpy as np

# Seed global pour reproductibilité
np.random.seed(42)

# -----------------------------
# CHSH (Bell) - unchanged
# -----------------------------
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

# -----------------------------
# Double-slit S2 (Bloch)
# -----------------------------
def simulate_double_slit_s2(distribution='uniform', N=100000, gamma=3*np.pi, k=1.1, g=0.5):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros(100)

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

    delta = np.pi/2
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

# -----------------------------
# Double-slit S1 (Poincaré)
# -----------------------------
def simulate_double_slit_s1(distribution='uniform', N=100000, gamma=5*np.pi, k=0.551, noise_sigma=np.pi/3):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros(100)

    if distribution == 'uniform':
        lambd = np.random.uniform(0, 2*np.pi, N)
        lambd2 = lambd.copy()
    else:
        z = np.random.beta(2,5,N)
        lambd = 2*np.pi*z
        lambd2 = (lambd + np.random.normal(0, noise_sigma, N)) % (2*np.pi)

    delta = np.pi/2
    for i, alpha in enumerate(alphas):
        delta_phi = 2*gamma*np.sin(alpha) + k*(np.cos(alpha - lambd) - np.cos(alpha + delta - lambd2))
        I[i] = 1 + np.mean(np.cos(delta_phi))

    Imax, Imin = np.max(I), np.min(I)
    V = (Imax - Imin)/(Imax + Imin)
    fringes = round(gamma / np.pi)
    return Imax, Imin, V, fringes

# -----------------------------
# Enhanced GHZ (NO post-selection)
# - Introduce: sigma_common (eta) and sigma_delta (per-particle noise).
# - When sigma_common >> sigma_delta, lambda_i are strongly correlated.
# -----------------------------
def simulate_ghz_enhanced(model='S2', alpha=0.5, beta=0.5,
                          sigma_common=0.05, sigma_delta=0.005, N=100000):
    """
    sigma_common : écart-type de la composante commune (eta)
    sigma_delta  : écart-type du bruit individuel sur chaque particule
    """
    if model == 'S2':
        # baseline shared lambda_vec (structure commune)
        z = np.random.beta(alpha, beta, N)
        cos_t = 2*z - 1
        theta = np.arccos(cos_t)
        phi = 2*np.pi*np.random.uniform(0,1,N)
        sin_t = np.sin(theta)
        base = np.column_stack((sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t))

        # common fluctuation (eta) same for all particles but different per trial
        eta_common = np.random.normal(0, sigma_common, (N,3))

        # small independent fluctuations for each particle
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

        # extra random secondary vectors
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
            return E, p_avg, eff

        E_xxx, p_xxx, eff_xxx = get_outcomes(x, x, x)
        E_xyy, p_xyy, eff_xyy = get_outcomes(x, y, y)
        E_yxy, p_yxy, eff_yxy = get_outcomes(y, x, y)
        E_yyx, p_yyx, eff_yyx = get_outcomes(y, y, x)

    elif model == 'S1':
        # S1 analogue: angles on circle
        z = np.random.beta(alpha, beta, N)
        lambda_angle = 2*np.pi*z

        eta_common = np.random.normal(0, sigma_common, N)  # scalar common noise
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
            return E, p_avg, eff

        E_xxx, p_xxx, eff_xxx = get_outcomes(x, x, x)
        E_xyy, p_xyy, eff_xyy = get_outcomes(x, y, y)
        E_yxy, p_yxy, eff_yxy = get_outcomes(y, x, y)
        E_yyx, p_yyx, eff_yyx = get_outcomes(y, y, x)

    # M, average p and average efficiency
    M = abs(E_xyy + E_yxy + E_yyx - E_xxx)
    p_avg = (p_xxx + p_xyy + p_yxy + p_yyx) / 4.0
    terms = (eff_xxx + eff_xyy + eff_yxy + eff_yyx) / 4.0
    return M, p_avg, terms

# -----------------------------
# Small grid search to "improve" double-slit visibilities
# (keeps your model but searches a few parameter combos)
# -----------------------------
def optimize_double_slit_S2(distribution='uniform', N=50000):
    # small grid around typical params
    gamma_list = [2.5*np.pi, 3.0*np.pi, 3.5*np.pi]
    k_list = [0.9, 1.0, 1.1, 1.2]
    g_list = [0.4, 0.5, 0.6]
    best = None
    for gamma in gamma_list:
        for k in k_list:
            for g in g_list:
                Imax, Imin, V, fr = simulate_double_slit_s2(distribution=distribution, N=N, gamma=gamma, k=k, g=g)
                if best is None or V > best[0]:
                    best = (V, gamma, k, g, Imax, Imin, fr)
    return best  # returns tuple (V, gamma, k, g, Imax, Imin, fr)

def optimize_double_slit_S1(distribution='uniform', N=50000):
    gamma_list = [4.0*np.pi, 5.0*np.pi, 6.0*np.pi]
    k_list = [0.3, 0.5, 0.55, 0.7]
    noise_list = [np.pi/4, np.pi/3, np.pi/2]
    best = None
    for gamma in gamma_list:
        for k in k_list:
            for noise in noise_list:
                Imax, Imin, V, fr = simulate_double_slit_s1(distribution=distribution, N=N, gamma=gamma, k=k, noise_sigma=noise)
                # choose the params that produce the target behavior (high or low V depending on taste)
                if best is None or V > best[0]:
                    best = (V, gamma, k, noise, Imax, Imin, fr)
    return best

# -----------------------------
# Run comparisons
# -----------------------------
if __name__ == "__main__":
    # CHSH quick run (unchanged)
    print("CHSH (Bell) RESULTS:")
    for pa, pb in [(0.5,0.5), (0.4,0.6), (0.3,0.7), (0.35,0.35)]:
        S, pa_avg, pb_avg = compute_S(pa, pb)
        print(f"pa={pa}, pb={pb}: |S|={S:.3f}, P(A+)={pa_avg:.3f}, P(B+)={pb_avg:.3f}")

    # Double-slit baseline and optimized (S2)
    print("\nDOUBLE-SLIT S2 baseline & optimization:")
    Imax_u, Imin_u, V_u, fr_u = simulate_double_slit_s2('uniform', N=100000)
    print(f"S2 uniform baseline: Imax={Imax_u:.3f}, Imin={Imin_u:.3f}, V={V_u:.3f}, Fringes={fr_u}")
    best_s2 = optimize_double_slit_S2(distribution='uniform', N=40000)
    print("S2 best found (small grid): V={:.3f}, gamma={:.3f}π, k={}, g={}".format(best_s2[0], best_s2[1]/np.pi, best_s2[2], best_s2[3]))
    print(" --> Imax, Imin, fringes =", best_s2[4], best_s2[5], best_s2[6])

    # Double-slit S1 baseline & optimized
    print("\nDOUBLE-SLIT S1 baseline & optimization:")
    Imax1_u, Imin1_u, V1_u, fr1_u = simulate_double_slit_s1('uniform', N=100000, noise_sigma=np.pi/3)
    print(f"S1 uniform baseline: Imax={Imax1_u:.3f}, Imin={Imin1_u:.3f}, V={V1_u:.3f}, Fringes={fr1_u}")
    best_s1 = optimize_double_slit_S1(distribution='uniform', N=40000)
    print("S1 best found (small grid): V={:.3f}, gamma={:.3f}π, k={}, noise_sigma={:.3f}".format(best_s1[0], best_s1[1]/np.pi, best_s1[2], best_s1[3]))
    print(" --> Imax, Imin, fringes =", best_s1[4], best_s1[5], best_s1[6])

    # GHZ enhanced - tests: vary sigma_common / sigma_delta to see effect
    print("\nGHZ enhanced - sweep sigma_common (keep sigma_delta small):")
    for sigma_common in [0.0, 0.005, 0.01, 0.02, 0.05]:
        M, p_avg, terms = simulate_ghz_enhanced('S2', sigma_common=sigma_common, sigma_delta=0.001, N=100000)
        print(f"sigma_common={sigma_common:.3f} | M={M:.3f}, P(±)={p_avg:.3f}, eff={terms:.3f}")

    print("\nGHZ enhanced - sweep sigma_delta (keep sigma_common moderate):")
    for sigma_delta in [0.000, 0.001, 0.005, 0.01, 0.02]:
        M, p_avg, terms = simulate_ghz_enhanced('S2', sigma_common=0.02, sigma_delta=sigma_delta, N=100000)
        print(f"sigma_delta={sigma_delta:.4f} | M={M:.3f}, P(±)={p_avg:.3f}, eff={terms:.3f}")

    # Final sample: a tuned attempt (strong common noise, tiny individual noise)
    print("\nGHZ tuned attempt (strong common, tiny individual):")
    M_tuned, p_tuned, eff_tuned = simulate_ghz_enhanced('S2', sigma_common=0.08, sigma_delta=0.0005, N=150000)
    print(f"Tuned S2: M={M_tuned:.3f}, P(±)={p_tuned:.3f}, eff={eff_tuned:.3f}")

    # Optional: S1 versions
    M_s1, p_s1, eff_s1 = simulate_ghz_enhanced('S1', sigma_common=0.02, sigma_delta=0.001, N=100000)
    print(f"Sample S1: M={M_s1:.3f}, P(±)={p_s1:.3f}, eff={eff_s1:.3f}")
