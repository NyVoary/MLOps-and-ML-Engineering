import numpy as np

# =====================================================================
# 1. Vector calculus operators (numerical)
# =====================================================================

def compute_divergence_2d(F_x, F_y, x, y, h=1e-5):
    """
    Computes the divergence of a vector field F = (F_x, F_y) at point (x, y).
    div(F) = dF_x/dx + dF_y/dy
    """
    # Centered finite differences for dF_x / dx
    dFx_dx = (F_x(x + h, y) - F_x(x - h, y)) / (2 * h)

    # Centered finite differences for dF_y / dy
    dFy_dy = (F_y(x, y + h) - F_y(x, y - h)) / (2 * h)

    return dFx_dx + dFy_dy

def compute_laplacian_2d(f, x, y, h=1e-5):
    """
    Computes the Laplacian of a scalar function f(x, y) at point (x, y).
    Laplacian(f) = d^2f/dx^2 + d^2f/dy^2
    """
    # Second-order centered finite differences for d^2f / dx^2
    d2f_dx2 = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h**2)

    # Second-order centered finite differences for d^2f / dy^2
    d2f_dy2 = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h**2)

    return d2f_dx2 + d2f_dy2


# =====================================================================
# 2. Demonstration & validation
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" VECTOR CALCULUS: DIVERGENCE & LAPLACIAN")
    print("==================================================\n")

    # Test evaluation point
    x_test, y_test = 2.0, 3.0

    # -----------------------------------------------------------------
    # Test 1: Divergence of the field F(x, y) = [3x^2, 2y]
    # -----------------------------------------------------------------
    F_x = lambda x, y: 3 * x**2
    F_y = lambda x, y: 2 * y

    div_num = compute_divergence_2d(F_x, F_y, x_test, y_test)
    div_exact = 6 * x_test + 2  # exact result worked out by hand

    print("--- DIVERGENCE ---")
    print(f"Vector field: F(x, y) = [3x^2, 2y]")
    print(f"Evaluation point (x, y): ({x_test}, {y_test})")
    print(f"Numerical divergence: {div_num:.6f}")
    print(f"Exact divergence (6x + 2): {div_exact:.6f}")
    print(f"Error: {abs(div_num - div_exact):.2e}\n")

    # -----------------------------------------------------------------
    # Test 2: Laplacian of f(x, y) = x^3 + y^3
    # -----------------------------------------------------------------
    f_scalar = lambda x, y: x**3 + y**3

    lap_num = compute_laplacian_2d(f_scalar, x_test, y_test)
    lap_exact = 6 * x_test + 6 * y_test  # exact result worked out by hand

    print("--- LAPLACIAN ---")
    print(f"Scalar function: f(x, y) = x^3 + y^3")
    print(f"Evaluation point (x, y): ({x_test}, {y_test})")
    print(f"Numerical Laplacian: {lap_num:.6f}")
    print(f"Exact Laplacian (6x + 6y): {lap_exact:.6f}")
    print(f"Error: {abs(lap_num - lap_exact):.2e}")
    print("==================================================")
