import numpy as np

def f(x: float, y: float) -> float:
    """
    Scalar function f(x, y) = x^2 * y + 4 * y^2
    """
    return (x**2) * y + 4 * (y**2)


def analytical_gradient(x: float, y: float) -> np.ndarray:
    """
    Computes the exact (analytical) gradient of f(x, y).

    grad f(x, y) = [ df/dx, df/dy ]
                 = [ 2*x*y, x^2 + 8*y ]
    """
    df_dx = 2 * x * y
    df_dy = x**2 + 8 * y
    return np.array([df_dx, df_dy], dtype=float)


def numerical_gradient(func, point: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Computes the numerical gradient of a function f via centered finite differences.

    Args:
        func: the scalar function to differentiate (takes an array of coordinates).
        point: the point [x, y] at which to evaluate the gradient.
        h: the infinitesimal step size.

    Returns:
        np.ndarray: the approximated gradient vector [df/dx, df/dy].
    """
    grad = np.zeros_like(point, dtype=float)

    for i in range(len(point)):
        # Point perturbed in the +h direction
        point_plus = np.array(point, dtype=float)
        point_plus[i] += h

        # Point perturbed in the -h direction
        point_minus = np.array(point, dtype=float)
        point_minus[i] -= h

        # Centered finite-difference formula
        grad[i] = (func(*point_plus) - func(*point_minus)) / (2 * h)

    return grad


if __name__ == "__main__":
    # Test point
    x_test, y_test = 1.0, 2.0
    point_test = np.array([x_test, y_test])

    print("==================================================")
    print(" PARTIAL DERIVATIVES & GRADIENT CALCULATION")
    print("==================================================")
    print(f"Function under study: f(x, y) = x^2 * y + 4 * y^2")
    print(f"Evaluation point: (x = {x_test}, y = {y_test})\n")

    # 1. Analytical (exact) computation
    grad_exact = analytical_gradient(x_test, y_test)
    print(f"1. Analytical (exact) gradient:")
    print(f"   df/dx = {grad_exact[0]}")
    print(f"   df/dy = {grad_exact[1]}")
    print(f"   grad(f) vector = {grad_exact}\n")

    # 2. Numerical computation (finite differences)
    grad_num = numerical_gradient(f, point_test)
    print(f"2. Numerical gradient (approximated with h=1e-5):")
    print(f"   grad(f) vector = {grad_num}\n")

    # 3. Accuracy check (relative error)
    error = np.linalg.norm(grad_exact - grad_num)
    print(f"3. Validation:")
    print(f"   Absolute error (L2 norm) = {error:.8f}")

    if np.allclose(grad_exact, grad_num, atol=1e-4):
        print("   Success: the numerical method validates the analytical computation!")
    else:
        print("   Error: the gap between the two results is too large.")
    print("==================================================")
