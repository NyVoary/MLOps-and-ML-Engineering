import numpy as np

def compute_hessian(f, point, h=1e-4):
    """
    Numerically computes the Hessian matrix of a scalar function f
    at the evaluation point 'point', using centered finite differences.

    Parameters:
        f     : scalar function that takes a NumPy vector (size n) and returns a real number
        point : NumPy vector or list [x1, x2, ..., xn] input point
        h     : step size for the finite differences (1e-4 is ideal for f'')

    Returns:
        H     : NumPy Hessian matrix of shape (n, n)
    """
    point = np.array(point, dtype=float)
    n = len(point)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # 1. Pure second derivative: d^2f / dx_i^2
                # Formula: (f(x + h) - 2*f(x) + f(x - h)) / h^2
                point_plus = np.copy(point)
                point_minus = np.copy(point)

                point_plus[i] += h
                point_minus[i] -= h

                f_plus = f(point_plus)
                f_ref = f(point)
                f_minus = f(point_minus)

                H[i, i] = (f_plus - 2 * f_ref + f_minus) / (h ** 2)
            else:
                # 2. Mixed second derivative: d^2f / (dx_i dx_j)
                # 4-point formula for the centered mixed finite difference
                p_pp = np.copy(point)  # +h_i, +h_j
                p_pm = np.copy(point)  # +h_i, -h_j
                p_mp = np.copy(point)  # -h_i, +h_j
                p_mm = np.copy(point)  # -h_i, -h_j

                p_pp[i] += h; p_pp[j] += h
                p_pm[i] += h; p_pm[j] -= h
                p_mp[i] -= h; p_mp[j] += h
                p_mm[i] -= h; p_mm[j] -= h

                H[i, j] = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)) / (4 * (h ** 2))

    return H

# =====================================================================
# Demonstration and validation
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" HESSIAN MATRIX CALCULATOR (NUMPY)")
    print("==================================================\n")

    # Our test function: f(x, y) = 3*(x^2)*y + 2*(y^3) - 5*x
    def loss_function(v):
        x, y = v[0], v[1]
        return 3 * (x**2) * y + 2 * (y**3) - 5 * x

    pt = [1.0, 2.0]
    H_calc = compute_hessian(loss_function, pt)

    print("--- EXAMPLE: f(x, y) = 3x^2y + 2y^3 - 5x ---")
    print(f"Evaluation point: (x={pt[0]}, y={pt[1]})")
    print(f"Hessian matrix shape: {H_calc.shape}")
    print("\nComputed Hessian matrix H(1, 2):")
    print(np.round(H_calc, 4))

    print("\nExpected theoretical matrix:")
    print("[[12.  6.]\n [ 6. 24.]]")

    # Symmetry check (Schwarz's theorem)
    is_symmetric = np.allclose(H_calc, H_calc.T)
    print(f"\nIs the matrix symmetric? {is_symmetric}")
    print("==================================================")
