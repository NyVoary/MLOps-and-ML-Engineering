# Multivariable Calculus

This is the roadmap sub-section where I learn how calculus extends to functions of several variables. Each topic below has its own folder with a focused `README.md` (theory, worked example, pseudo-code, a plot) and my implementation script(s).

## Topics

1. [Partial Derivatives](01-partial-derivatives/README.md) 🔍 - how a function changes when only one variable moves.
2. [Gradient Vector](02-gradient-vector/README.md) 🧭 - all partial derivatives gathered into one vector; the direction of steepest ascent.
3. [Jacobian Matrix](03-jacobian-matrix/README.md) 🧮 - the gradient generalized to functions with multiple outputs; the engine behind backpropagation.
4. [Hessian Matrix](04-hessian-matrix/README.md) 🧠 - the gradient of the gradient; the curvature that tells a minimum from a saddle point.
5. [Lagrangian Multipliers](05-lagrangian-multipliers/README.md) ⚖️ - optimizing a function under a constraint, by finding where the gradients of the objective and the constraint line up.

## Shared conventions

### Script structure

Across all topics, I use the same software-engineering approach (doctests, typing, docstrings):

1. **Analytical (exact) approach**: a Python function that computes the exact value from the known formula.
2. **Numerical approach (finite differences)**: a generic function that approximates the same quantity using the centered finite-difference formula:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x - h, y)}{2h}$$

3. **Validation**: comparing the exact (analytical) value against the numerical approximation to check my implementation is correct.
4. **Visualization**: whenever it helps me build intuition, a `plot_*.py` script using Matplotlib - a 2D/3D picture of the function, its slope, or its curvature.

### A note on the step size (h)

In numerical finite-difference computation, the choice of $h = 10^{-5}$ (i.e. `1e-5`) is an ideal compromise for balancing two opposing types of computer error.

**A. Truncation Error**

- Theory: mathematically, the smaller $h$ is, the closer the formula $\dfrac{f(x+h) - f(x-h)}{2h}$ gets to the true derivative.
- Problem: if $h$ is too large (e.g. $h = 0.1$), the tangent approximation is very inaccurate because the curve varies too much over that interval.

**B. Round-off Error (Floating-Point Limitation)**

- Problem: computers store decimal numbers with limited precision (typically 64 bits in Python, i.e. about 15 to 17 significant digits).
- Consequence: if $h$ is too small (e.g. $h = 10^{-15}$), the difference $f(x+h) - f(x-h)$ becomes a number extremely close to zero. The computer then loses precision when subtracting two nearly identical numbers (a phenomenon called **catastrophic cancellation**), and then dividing by a tiny $2h$ makes it worse.

**Why exactly $10^{-5}$ or $10^{-8}$?** ⚖️

For a standard floating-point representation (64-bit double precision, with $\epsilon \approx 10^{-16}$), for the centered finite-difference formula, the theoretical optimal error is reached when:

$$h \approx \sqrt[3]{\epsilon} \approx \sqrt[3]{10^{-16}} \approx 10^{-5.3}$$

This is why $h = 10^{-5}$ (or $10^{-6}$) is the standard default setting in engineering: it minimizes both the geometric error of the curve and the round-off error of the computer. 💡

Note: for the Hessian (second derivatives), I use a slightly larger $h = 10^{-4}$ instead - the theoretical optimum for a second-order centered difference scales as $\epsilon^{1/4}$ rather than $\epsilon^{1/3}$, since the formula divides by $h^2$ and is more sensitive to round-off error.
