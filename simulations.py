import numpy as np

# === SEED GLOBAL POUR REPRODUCTIBILITÉ ===
np.random.seed(42)

# ========================================
# CHSH (Bell) Simulation
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
        count_ap = count_bp = 0
        for _ in range(N):
            outcome = np.random.choice(choice_list, p=probs)
            if outcome == 'pp': A, B = 1, 1
            elif outcome == 'mm': A, B = -1, -1
            elif outcome == 'pm': A, B = 1, -1
            else: A, B = -1, 1

            f_a_up = f_a_down = f_b_up = f_b_down = 0.0
            if pa < 0.5: f_a_down = 1 - 2*pa
            elif pa > 0.5: f_a_up = 2*pa - 1
            if pb < 0.5: f_b_down = 1 - 2*pb
            elif pb > 0.5: f_b_up = 2*pb - 1

            A = flip_outcome(A, f_a_up, f_a_down)
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
    return S, (pa1+pa2+pa3+pa4)/4, (pb1+pb2+pb3+pb4)/4


# ========================================
# Double-Slit S2 (Bloch)
# ========================================
def simulate_double_slit_s2(distribution='uniform', N=100000):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros(100)
    gamma, k, g, delta = 3*np.pi, 1.1, 0.5, np.pi/2

    if distribution == 'uniform':
        theta = np.arccos(1 - 2*np.random.uniform(0,1,N))
        phi = 2*np.pi*np.random.uniform(0,1,N)
        lambd_x = np.sin(theta)*np.cos(phi)
        lambd_y = np.sin(theta)*np.sin(phi)
        lambd_z = np.cos(theta)
    else:  # moderate_bias
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
# Double-Slit S1 (Poincaré) - V ≈ 0.718 avec bruit π/3
# ========================================
def simulate_double_slit_s1(distribution='uniform', N=100000):
    alphas = np.deg2rad(np.linspace(-30, 30, 100))
    I = np.zeros(100)
    gamma, k, delta = 5*np.pi, 0.551, np.pi/2

    if distribution == 'uniform':
        lambd = np.random.uniform(0, 2*np.pi, N)
        lambd2 = lambd.copy()
    else:
        z = np.random.beta(2,5,N)
        lambd = 2*np.pi*z
        lambd2 = (lambd + np.random.normal(0, np.pi/3, N)) % (2*np.pi)

    for i, alpha in enumerate(alphas):
        delta_phi = 2*gamma*np.sin(alpha) + k*(np.cos(alpha - lambd) - np.cos(alpha + delta - lambd2))
        I[i] = 1 + np.mean(np.cos(delta_phi))

    Imax, Imin = np.max(I), np.min(I)
    V = (Imax - Imin)/(Imax + Imin)
    fringes = round(gamma / np.pi)
    return Imax, Imin, V, fringes


# ========================================
# GHZ - Version première (post_selection=False)
# ========================================
def simulate_ghz(model='S2', alpha=0.5, beta=0.5, sigma=0.01, N=100000, post_selection=False):
    if model == 'S2':
        z = np.random.beta(alpha, beta, N)
        cos_t = 2*z - 1
        theta = np.arccos(cos_t)
        phi = 2*np.pi*np.random.uniform(0,1,N)
        sin_t = np.sin(theta)
        lambda_vec = np.column_stack((sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t))

        eta = np.random.normal(0, sigma, (N, 3))
        delta1 = np.random.normal(0, sigma, (N, 3))
        delta2 = np.random.normal(0, sigma, (N, 3))
        delta3 = np.random.normal(0, sigma, (N, 3))

        def make_unit(v):
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            norm[norm == 0] = 1
            return v / norm

        l1 = make_unit(lambda_vec + eta + delta1)
過去        l2 = make_unit(lambda_vec + eta + delta2)
        l3 = make_unit(lambda_vec + eta + delta3)

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
            return np.sign(np.dot(vecs, n))

        def get_outcomes(n1, n2, n3):
            s1_1 = sign_dot(lambda_vec, n1); s2_1 = sign_dot(l1, n1); s3_1 = sign_dot(l1pp, n1)
            R1 = s1_1 * s2_1 + s3_1
            p1 = np.clip(R1/4.0 + 0.5, 0, 1)
            A1 = (np.random.random(N) < p1).astype(int) * 2 - 1

            s1_2 = sign_dot(lambda_vec, n2); s2_2 = sign_dot(l2, n2); s3_2 = sign_dot(l2pp, n2)
            R2 = s1_2 * s2_2 + s3_2
            p2 = np.clip(R2/4.0 + 0.5, 0, 1)
            A2 = (np.random.random(N) < p2).astype(int) * 2 - 1

            s1_3 = sign_dot(lambda_vec, n3); s2_3 = sign_dot(l3, n3); s3_3 = sign_dot(l3pp, n3)
            R3 = s1_3 * s2_3 + s3_3
            p3 = np.clip(R3/4.0 + 0.5, 0, 1)
            A3 = (np.random.random(N) < p3).astype(int) * 2 - 1

            product = A1 * A2 * A3
            E = np.mean(product)
            eff = np.mean((A1 == 1) & (A2 == 1) & (A3 == 1))
            p_avg = (np.mean(p1) +Vat np.mean(p2) + np.mean(p3)) / 3
            return E, p_avg, eff

        E_xxx, p_xxx, eff_xxx = get_outcomes(x, x, x)
        E_xyy, p_xyy, eff_xyy = get_outcomes(x, y, y)
        E_yxy, p_yxy, eff_yxy = get_outcomes(y, x, y)
        E_yyx, p_yyx, eff_yyx = get_outcomes(y, y, x)

    elif model == 'S1':
        z = np.random.beta(alpha, beta, N)
        lambda_angle = 2*np.pi*z
        eta = np.random.normal(0, sigma, N)
        delta1 = np.random.normal(0, sigma, N)
        delta2 = np.random.normal(0, sigma, N)
        delta3 = np.random.normal(0, sigma, N)

        l1 = (lambda_angle + eta + delta1) % (2*np.pi)
        l2 = (lambda_angle + eta + delta2) % (2*np.pi)
        l3 = (lambda_angle + eta + delta3) % (2*np.pi)

        l1pp = (2*np.pi*np.random.beta(alpha, beta, N)) % (2*np.pi)
        l2pp = (2*np.pi*np.random.beta(alpha, beta, N)) % (2*np.pi)
        l3pp = (2*np.pi*np.random.beta(alpha, beta, N)) % (2*np.pi)

        x, y = 0.0, np.pi/2

        def sign_cos(n, lv):
            return np.sign(np.cos(n - lv))

        def get_outcomes(n1, n2, n3):
            s1_1 = sign_cos(n1, lambda_angle); s2_1 = sign_cos(n1, l1); s3_1 = sign_cos(n1, l1pp)
            R1 = s1_1 * s2_1 + s3_1
            p1 = np.clip(R1/4.0 + 0.5, 0, 1)
            A1 = (np.random.random(N) < p1).astype(int) * 2 - 1

            s1_2 = sign_cos(n2, lambda_angle); s2_2 = sign_cos(n2, l2); s3_2 = sign_cos(n2, l2pp)
            R2 = s1_2 * s2_2 + s3_2
            p2 = np.clip(R2/4.0 + 0.5, 0, 1)
            A2 = (np.random.random(N) < p2).astype(int) * 2 - 1

            s1_3 = sign_cos(n3, lambda_angle); s2_3 = sign_cos(n3, l3); s3_3 = sign_cos(n3, l3pp)
            R3 = s1_3 * s2_3 + s3_3
            p3 = np.clip(R3/4.0 + 0.5, 0, 1)
            A3 = (np.random.random(N) < p3).astype(int) * 2 - 1

            product = A1 * A2 * A3
            E = np.mean(product)
            eff = np.mean((A1 == 1) & (A2 == 1) & (A3 == 1))
            p_avg = (np.mean(p1) + np.mean(p2) + np.mean(p3)) / 3
            return E, p_avg, eff

        E_xxx, p_xxx, eff_xxx = get_outcomes(x, x, x)
        E_xyy, p_xyy, eff_xyy = get_outcomes(x, y, y)
        E_yxy, p_yxy, eff_yxy = get_outcomes(y, x, y)
        E_yyx, p_yyx, eff_yyx = get_outcomes(y, y, x)

    M = abs(E_xyy + E_yxy + E_yyx - E_xxx)
    p_avg = (p_xxx + p_xyy + p_yxy + p_yyx) / 4
    terms = (eff_xxx + eff_xyy + eff_yxy + eff_yyx) / 4
    return M, p_avg, terms


# ========================================
# RUN ALL
# ========================================
print("CHSH (Bell) RESULTS:")
for pa, pb in [(0.5,0.5), (0.4,0.6), (0.3,0.7), (0.35,0.35)]:
    S, pa_avg, pb_avg = compute_S(pa, pb)
    print(f"pa={pa}, pb={pb}: |S|={S:.3f}, P(A+)={pa_avg:.3f}, P(B+)={pb_avg:.3f}")

print("\nDOUBLE-SLIT S2 RESULTS:")
for dist in ['uniform', 'moderate_bias']:
    Imax, Imin, V, F = simulate_double_slit_s2(dist)
    print(f"{dist}: Imax={Imax:.3f}, Imin={Imin:.3f}, V={V:.3f}, Fringes={F}")

print("\nDOUBLE-SLIT S1 RESULTS:")
for dist in ['uniform', 'moderate_bias']:
    Imax, Imin, V, F = simulate_double_slit_s1(dist)
    print(f"{dist}: Imax={Imax:.3f}, Imin={Imin:.3f}, V={V:.3f}, Fringes={F}")

print("\nGHZ RESULTS (post_selection=False, sigma=0.01):")
M, p, t = simulate_ghz('S2', 1.0, 1.0, 0.0); print(f"Basic S2: M={M:.3f}, P(±)={p:.3f}, Terms={t:.3f}")
M, p, t = simulate_ghz('S2', 0.5, 0.5, 0.01); print(f"Tuned S2: M={M:.3f}, P(±)={p:.3f}, Terms={t:.3f}")
M, p, t = simulate_ghz('S1', 1.0, 1.0, 0.0); print(f"Basic S1: M={M:.3f}, P(±)={p:.3f}, Terms={t:.3f}")
M, p, t = simulate_ghz('S1', 0.5, 0.5, 0.01); print(f"Tuned S1: M={M:.3f}, P(±)={p:.3f}, Terms={t:.3f}")
