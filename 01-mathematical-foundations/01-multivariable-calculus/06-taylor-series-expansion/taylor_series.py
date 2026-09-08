import numpy as np
import math

def numerical_derivative(f, x, n, h=1e-3):
    """
    Computes the n-th numerical derivative of a function f at point x
    using finite differences (successive approximations).
    """
    if n == 0:
        return f(x)
    elif n == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif n == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    elif n == 3:
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * (h**3))
    else:
        # For n >= 4, recursive order-1 approach
        return (numerical_derivative(f, x + h, n - 1, h) -
                numerical_derivative(f, x - h, n - 1, h)) / (2 * h)

def taylor_expansion_1d(f, a, x, order_N):
    """
    Computes the Taylor series approximation of f(x) around point 'a'
    up to order order_N, numerically.
    """
    approx = 0.0
    terms = []

    for n in range(order_N + 1):
        # 1. Compute the n-th derivative f^(n)(a)
        df_n = numerical_derivative(f, a, n)

        # 2. Compute the term (f^(n)(a) / n!) * (x - a)^n
        fact = math.factorial(n)
        term = (df_n / fact) * ((x - a)**n)

        approx += term
        terms.append(term)

    return approx, terms

# =====================================================================
# Demonstration and validation
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" TAYLOR SERIES APPROXIMATION (NUMERICAL)")
    print("==================================================\n")

    # Test function: exp(x) around a = 0
    f_test = np.exp
    a = 0.0
    x_eval = 0.1
    max_order = 3

    print(f"Function: f(x) = e^x")
    print(f"Expansion point (a): {a}")
    print(f"Evaluation point (x): {x_eval}\n")

    true_val = f_test(x_eval)
    approx_val, terms = taylor_expansion_1d(f_test, a, x_eval, max_order)

    print("--- Approximation results by order ---")
    cum_sum = 0
    for order, term in enumerate(terms):
        cum_sum += term
        print(f"Order {order}: term = {term:.8f} | cumulative = {cum_sum:.8f}")

    print("\n--------------------------------------------------")
    print(f"True value of e^({x_eval}): {true_val:.8f}")
    print(f"Estimated value (order {max_order}): {approx_val:.8f}")
    print(f"Absolute error: {abs(true_val - approx_val):.8e}")
    print("==================================================")
