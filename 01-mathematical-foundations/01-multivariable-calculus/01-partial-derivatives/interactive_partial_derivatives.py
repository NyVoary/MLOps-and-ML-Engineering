import sympy as sp
import numpy as np

# 1. SymPy handles the symbols and the symbolic display
x, y = sp.symbols('x y')
f_expr = x**2 * y + 4 * y**2

# Symbolic formulas, for on-screen display
df_dx_expr = sp.diff(f_expr, x)
df_dy_expr = sp.diff(f_expr, y)

print(f"SymPy display: df/dx = {df_dx_expr}")

# 2. Converting to an ultra-fast Python/NumPy function
f_num = sp.lambdify((x, y), f_expr, 'numpy')

# 3. NumPy performs the numerical computation via finite differences at point (1.0, 2.0)
x_val, y_val = 1.0, 2.0
h = 1e-5

df_dx_numpy = (f_num(x_val + h, y_val) - f_num(x_val - h, y_val)) / (2 * h)
df_dy_numpy = (f_num(x_val, y_val + h) - f_num(x_val, y_val - h)) / (2 * h)

gradient_vector = np.array([df_dx_numpy, df_dy_numpy])
print(f"Gradient evaluated via NumPy: {gradient_vector}")
