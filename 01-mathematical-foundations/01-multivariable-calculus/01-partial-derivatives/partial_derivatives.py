import numpy as np

# 1. Function under study: f(x, y) = x^2 * y + 4 * y^2
def f(x: float, y: float) -> float:
    return (x**2) * y + 4 * (y**2)

# 2. Pseudo-code applied to the partial derivative with respect to X
def partial_derivative_x(func, x: float, y: float, h: float = 1e-5) -> float:
    """
    Computes the partial derivative ∂f/∂x at point (x, y) via centered finite differences.
    Only 'x' varies by +/- h, 'y' stays fixed.
    """
    f_plus = func(x + h, y)
    f_minus = func(x - h, y)

    df_dx = (f_plus - f_minus) / (2 * h)
    return df_dx

# 3. Pseudo-code applied to the partial derivative with respect to Y
def partial_derivative_y(func, x: float, y: float, h: float = 1e-5) -> float:
    """
    Computes the partial derivative ∂f/∂y at point (x, y) via centered finite differences.
    Only 'y' varies by +/- h, 'x' stays fixed.
    """
    f_plus = func(x, y + h)
    f_minus = func(x, y - h)

    df_dy = (f_plus - f_minus) / (2 * h)
    return df_dy


if __name__ == "__main__":
    # Test point
    x_val, y_val = 1.0, 2.0

    # Numerical calculations via my functions
    df_dx_num = partial_derivative_x(f, x_val, y_val)
    df_dy_num = partial_derivative_y(f, x_val, y_val)

    # Exact theoretical values (computed by hand: ∂f/∂x = 2xy = 4, ∂f/∂y = x^2 + 8y = 17)
    df_dx_exact = 4.0
    df_dy_exact = 17.0

    print("==================================================")
    print(" PARTIAL DERIVATIVES CALCULATION (NUMERICAL)")
    print("==================================================")
    print(f"Evaluation point: (x = {x_val}, y = {y_val})\n")

    print(f"df/dx computed: {df_dx_num:.5f} | exact value: {df_dx_exact}")
    print(f"df/dy computed: {df_dy_num:.5f} | exact value: {df_dy_exact}\n")

    # Building the gradient vector with NumPy
    gradient_vector = np.array([df_dx_num, df_dy_num])
    print(f"Gradient vector grad(f)(1, 2) = {gradient_vector}")
    print("==================================================")
