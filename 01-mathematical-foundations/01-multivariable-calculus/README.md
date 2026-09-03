# Partial Derivatives and Gradient Vector

## 1. The Partial Derivative 🔍

When a function depends on several variables — for example $f(x, y)$ — the **partial derivative** measures how the function changes when we vary **a single variable**, keeping all others constant.

### Notation

The partial derivative with respect to $x$ is denoted:

$$\frac{\partial f}{\partial x} \quad \text{or} \quad f'_x$$

The partial derivative with respect to $y$ is denoted:

$$\frac{\partial f}{\partial y} \quad \text{or} \quad f'_y$$

### Worked example

Consider the following simplified cost function:

$$f(x, y) = 3x^2 + 2xy + y^3$$

To compute $\dfrac{\partial f}{\partial x}$, we treat $y$ as a plain constant (like any fixed number: 2, 5, etc.):

$$\frac{\partial f}{\partial x} = 6x + 2y$$

To compute $\dfrac{\partial f}{\partial y}$, we treat $x$ as a constant:

$$\frac{\partial f}{\partial y} = 2x + 3y^2$$

## 2. The Gradient Vector 🧭

The **gradient** gathers all the partial derivatives of a function into a single vector.

### Definition

For a function $f(x, y)$, the gradient vector (denoted $\nabla f$, read "nabla f") is:

$$\nabla f(x, y) = \begin{pmatrix} \dfrac{\partial f}{\partial x} \\[6pt] \dfrac{\partial f}{\partial y} \end{pmatrix}$$

### Key geometric property

The gradient vector has two characteristics that are crucial for Machine Learning:

- **Direction**: it always points toward the direction where the function increases the fastest.
- **Norm (magnitude)**: its size indicates how steep the slope is.

> 💡 **The link to optimization (Gradient Descent):**
> To reduce an error (minimize a cost function), we move in the direction **opposite** to the gradient: $-\nabla f$.

### Your turn! ✏️

Let's apply this theory directly to a small practical case before moving to Python.

Consider the following function:

$$f(x, y) = x^2 y + 4y^2$$

- What is the partial derivative $\dfrac{\partial f}{\partial x}$?
- What is the partial derivative $\dfrac{\partial f}{\partial y}$?

Try to work out both calculations!

## 3. Script structure

To keep software-engineering rigor (with doctests, typing, and docstrings):

1. **Analytical (exact) approach**: a Python function that computes the exact value of the function and its gradient vector at point $(x, y)$.
2. **Numerical approach (finite differences)**: a generic function that computes the gradient of any function using the centered finite-difference approximation:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x - h, y)}{2h}$$

3. **Validation**: comparing the exact (analytical) value against the numerical approximation to verify our implementation is correct.

### The intuition behind this pseudo-code 💡

- **The small variation $h$**: we take a very small step (e.g. $10^{-5}$) to zoom in on the curve around the point $(x, y)$.
- **Isolating the variable**:
  - To get $\dfrac{\partial f}{\partial x}$, we evaluate the function slightly to the right $f(x+h, y)$ and slightly to the left $f(x-h, y)$. We don't touch $y$!
  - To get $\dfrac{\partial f}{\partial y}$, it's the opposite: $x$ stays fixed, and we vary $y$.
- **Computing the slope**: the difference $(f_{plus} - f_{minus})$ divided by the total distance $2h$ gives us the exact slope of the tangent (the partial derivative).

### Two ways to compute this in software

| Type of computation | What the program does | Example input | Example output | Main Python tool |
|---|---|---|---|---|
| **Numerical** | Computes exact or approximated values at a specific point $(x, y)$. | $x=1, y=2$ | The number $17.0$ | NumPy |
| **Symbolic** | Manipulates variables like on paper to find the mathematical expression of the derivative. | $f(x, y) = x^2 y + 4y^2$ | The expression $x^2 + 8y$ | SymPy |

### Comparing the methods

| Method | How it works | Advantage | Drawback | Used where? |
|---|---|---|---|---|
| **Symbolic** (SymPy) | Manipulates algebraic expressions to produce a formal formula. | Gives the exact equation $2xy$. | Very slow, requires a substitution step to evaluate at a point. | Theoretical math, formal computation, proof verification. |
| **Numerical** (NumPy) | Uses $\dfrac{f(x+h) - f(x-h)}{2h}$. | Very simple to implement. | Approximation (round-off errors), requires evaluating $f$ multiple times per parameter. | Gradient validation tests ("Gradient Checking"). |
| **Delegated / Automatic** (PyTorch / Autograd) | Computes the exact gradient value at a given point by building a computation graph. | Ultra-fast, exact up to machine precision, ideal for millions of parameters. | Requires a dedicated autodiff framework. | All of modern Deep Learning (Backpropagation). |

## 4. A note on the step size $h = 10^{-5}$

In numerical finite-difference computation, the choice of $h = 10^{-5}$ (i.e. `1e-5`) is an ideal compromise for balancing two opposing types of computer error.

### The two errors at war ⚔️

**A. Truncation Error**

- Theory: mathematically, the smaller $h$ is, the closer the formula $\dfrac{f(x+h) - f(x-h)}{2h}$ gets to the true derivative.
- Problem: if $h$ is too large (e.g. $h = 0.1$), the tangent approximation is very inaccurate because the curve varies too much over that interval.

**B. Round-off Error (Floating-Point Limitation)**

- Problem: computers store decimal numbers with limited precision (typically 64 bits in Python, i.e. about 15 to 17 significant digits).
- Consequence: if $h$ is too small (e.g. $h = 10^{-15}$), the difference $f(x+h) - f(x-h)$ becomes a number extremely close to zero. The computer then loses precision when subtracting two nearly identical numbers (a phenomenon called **catastrophic cancellation**), and then dividing by a tiny $2h$ makes it worse.

### Why exactly $10^{-5}$ or $10^{-8}$? ⚖️

For a standard floating-point representation (64-bit double precision, with $\epsilon \approx 10^{-16}$), for the centered finite-difference formula, the theoretical optimal error is reached when:

$$h \approx \sqrt[3]{\epsilon} \approx \sqrt[3]{10^{-16}} \approx 10^{-5.3}$$

This is why $h = 10^{-5}$ (or $10^{-6}$) is the standard default setting in engineering: it minimizes both the geometric error of the curve and the round-off error of the computer. 💡

---

## Pseudo-code

### Pseudo-code: Partial Derivative (Central Difference)

```
ALGORITHM: Calculate Partial Derivative

INPUT:
    - f: scalar function f(x, y)
    - point: vector containing [x, y]
    - var_index: index of the variable to differentiate (0 for x, 1 for y)
    - h: step size (default = 1e-5)

OUTPUT:
    - df_dvar: partial derivative value at the given point

BEGIN
    1. Create a copy of 'point' named 'point_plus'
    2. Create a copy of 'point' named 'point_minus'

    3. Add 'h' to 'point_plus' at position 'var_index'      // (e.g., [x + h, y] if var_index == 0)
    4. Subtract 'h' from 'point_minus' at position 'var_index' // (e.g., [x - h, y] if var_index == 0)

    5. Evaluate f_plus = f(point_plus)
    6. Evaluate f_minus = f(point_minus)

    7. Compute df_dvar = (f_plus - f_minus) / (2 * h)

    RETURN df_dvar
END
```

### Pseudo-code: Gradient Vector

```
ALGORITHM: Calculate Gradient Vector

INPUT:
    - f: scalar function
    - point: vector [x1, x2, ..., xn]
    - h: step size

OUTPUT:
    - grad: vector containing all partial derivatives [df/dx1, df/dx2, ..., df/dxn]

BEGIN
    1. Initialize 'grad' array of the same size as 'point' with zeros
    2. FOR each index 'i' from 0 to length(point) - 1 DO:
           grad[i] = Calculate Partial Derivative(f, point, var_index=i, h)
       END FOR

    RETURN grad
END
```

### Pseudo-code: Interactive Partial Derivatives & Gradient Calculator

```
ALGORITHM: Interactive Partial Derivatives & Gradient Calculator

INPUT:
    - expr_str: mathematical expression entered by user as text (e.g., "x**2 * y + 4*y**2")
    - x_val: numeric float value for x
    - y_val: numeric float value for y
    - h: step size for numerical derivative (default = 1e-5)

OUTPUT:
    - Symbolic partial derivatives (text formulas via SymPy)
    - Numeric values of partial derivatives evaluated at (x_val, y_val) via NumPy
    - Gradient Vector: ∇f(x, y) = [df/dx, df/dy]

BEGIN
    1. Define symbolic variables 'x' and 'y' using SymPy
    2. Parse 'expr_str' into a symbolic expression f_expr

    3. Compute SYMBOLIC partial derivatives (for display purpose):
          df_dx_expr = diff(f_expr, x)
          df_dy_expr = diff(f_expr, y)

    4. Convert symbolic expression f_expr into a fast NumPy-compatible function:
          f_num = lambdify((x, y), f_expr, 'numpy')

    5. Compute NUMERICAL partial derivatives at point (x_val, y_val) using central differences with NumPy:
          df_dx_num = (f_num(x_val + h, y_val) - f_num(x_val - h, y_val)) / (2 * h)
          df_dy_num = (f_num(x_val, y_val + h) - f_num(x_val, y_val - h)) / (2 * h)

    6. Construct the Gradient Vector with NumPy:
          gradient_vector = array([df_dx_num, df_dy_num])

    7. Display symbolic formulas (SymPy), numerical evaluations (NumPy), and the Gradient Vector
END
```
