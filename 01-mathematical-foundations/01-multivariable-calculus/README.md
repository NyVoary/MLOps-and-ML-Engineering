# Multivariable Calculus — Partial Derivatives, Gradient Vector & Jacobian Matrix

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

## 3. Script Structure

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

## 4. A Note on the Step Size $h = 10^{-5}$

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

## 5. The Jacobian Matrix 🧮

### 5.1 From gradient to Jacobian

So far, we've worked with functions that take several inputs $(x, y)$ but produce a single scalar output: $f(x, y) \in \mathbb{R}$. The gradient then gathers all the partial derivatives into a single vector.

The **Jacobian matrix** is simply the generalization of the gradient for functions with several inputs **and** several outputs:

$$\mathbf{F}(x, y) = \begin{pmatrix} f_1(x, y) \\ f_2(x, y) \end{pmatrix}$$

Each **row** of the Jacobian matrix is the gradient of one of the output functions:

$$\mathbf{J} = \begin{pmatrix} \dfrac{\partial f_1}{\partial x} & \dfrac{\partial f_1}{\partial y} \\[6pt] \dfrac{\partial f_2}{\partial x} & \dfrac{\partial f_2}{\partial y} \end{pmatrix}$$

### 5.2 Worked example

Consider a function $\mathbf{F}$ that takes a 2D point $(x, y)$ as input and returns a 2D vector as output:

$$\mathbf{F}(x, y) = \begin{pmatrix} f_1(x, y) \\ f_2(x, y) \end{pmatrix} = \begin{pmatrix} x^2 y \\ x + 3y \end{pmatrix}$$

**Symbolic partial derivatives**, computed one at a time:

- $\dfrac{\partial f_1}{\partial x} = 2xy$
- $\dfrac{\partial f_1}{\partial y} = x^2$
- $\dfrac{\partial f_2}{\partial x} = 1$
- $\dfrac{\partial f_2}{\partial y} = 3$

Which gives the symbolic Jacobian matrix:

$$J(x, y) = \begin{pmatrix} 2xy & x^2 \\ 1 & 3 \end{pmatrix}$$

**Evaluating at a specific point** $(x = 2, y = 3)$:

- $2xy = 2 \cdot 2 \cdot 3 = 12$
- $x^2 = 2^2 = 4$

$$J(2, 3) = \begin{pmatrix} 12 & 4 \\ 1 & 3 \end{pmatrix}$$

> 💡 **Reading the matrix**: row 1 corresponds to output $f_1$, column 1 corresponds to input $x$. The entry $J_{11} = 12$ is the partial derivative $\dfrac{\partial f_1}{\partial x}$ — it tells us $f_1$'s *sensitivity* to $x$. If we nudge $x$ by $+0.01$ (keeping $y = 3$ fixed), $f_1$ increases by roughly $12 \times 0.01 = +0.12$: each entry of the Jacobian is a local rate of change, exactly like a single partial derivative.

### 5.3 General shape rule

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ (with $n$ inputs and $m$ outputs), the Jacobian matrix has shape $m \times n$: **one row per output, one column per input**.

For example, with 3 input variables and 2 output functions:

$$J = \begin{pmatrix} \dfrac{\partial f_1}{\partial x_1} & \dfrac{\partial f_1}{\partial x_2} & \dfrac{\partial f_1}{\partial x_3} \\[6pt] \dfrac{\partial f_2}{\partial x_1} & \dfrac{\partial f_2}{\partial x_2} & \dfrac{\partial f_2}{\partial x_3} \end{pmatrix} \quad (2 \times 3 \text{ matrix})$$

### 5.4 Two ways to read the Jacobian

The same object is used very differently depending on the field:

- **Geometry / analysis** 📐: the Jacobian measures how a transformation deforms space (change of variables, volume computation via the Jacobian determinant, surface deformation). More formally, it's the linear map that gives the best local approximation of a differentiable function between two vector spaces.
- **Machine Learning** 🤖: the Jacobian measures how sensitive each output of a model is to each input. If a neural network layer takes a 3D input vector $X$ and produces a 2D output vector $Y$, the Jacobian matrix contains every rate of change needed to propagate the error backward (**backpropagation**).

### 5.5 The Jacobian inside a neural network layer

Consider a dense (fully-connected) layer with 3 inputs ($x_1, x_2, x_3$) and 2 output neurons ($z_1, z_2$), each computed as a weighted sum plus a bias:

$$z_1 = w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1$$
$$z_2 = w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2$$

During training, backpropagation needs to answer: *"If I slightly change input $x_1$, how does it affect $z_1$ **and** $z_2$?"* That information is exactly the Jacobian of the layer, $\mathbf{J} = \dfrac{\partial \mathbf{z}}{\partial \mathbf{x}}$:

$$J = \begin{pmatrix} \dfrac{\partial z_1}{\partial x_1} & \dfrac{\partial z_1}{\partial x_2} & \dfrac{\partial z_1}{\partial x_3} \\[6pt] \dfrac{\partial z_2}{\partial x_1} & \dfrac{\partial z_2}{\partial x_2} & \dfrac{\partial z_2}{\partial x_3} \end{pmatrix} = \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{pmatrix}$$

**Key takeaway**: for a linear layer $Z = WX + B$, the Jacobian with respect to the input $X$ is nothing but the weight matrix $W$ itself. This is why libraries like PyTorch and TensorFlow are, under the hood, continuously multiplying Jacobian matrices together (via the chain rule) to propagate the error from layer to layer.

## 6. Backpropagation & Gradient Descent — Putting It All Together 🔄

### 6.1 The learning cycle

Training a neural network loops through three phases:

1. **Forward pass** ➡️: input data flows through the network, layer by layer, to produce a prediction.
2. **Loss calculation** 🎯: a cost function (e.g. squared error) compares the prediction to the true target to measure the overall error $\mathcal{L}$.
3. **Backpropagation** ⬅️: the error computed at the output is propagated backward through the network.

### 6.2 The chain rule and the Jacobian

During backpropagation, we need the gradient of the loss $\mathcal{L}$ with respect to every weight $w$ and bias $b$. For a simple chain $x \to (\text{neuron}, w) \to \hat{y} \to \mathcal{L}$, the chain rule multiplies local sensitivities:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial w}$$

In a deep network with several layers and neurons, these local-sensitivity computations become **Jacobian matrix multiplications**. Concretely, for a layer with 3 inputs and 2 outputs, the layer's Jacobian $\dfrac{\partial Z}{\partial X}$ has shape $2 \times 3$, and the incoming error vector from the layer above, $\dfrac{\partial \mathcal{L}}{\partial Z}$, has shape $1 \times 2$ (one value per output). Multiplying them:

$$\underbrace{\frac{\partial \mathcal{L}}{\partial X}}_{1 \times 3} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z}}_{1 \times 2} \times \underbrace{\frac{\partial Z}{\partial X}}_{2 \times 3}$$

redistributes the output error back onto each of the 3 inputs — this is exactly how the error "climbs back" through every layer of the network.

### 6.3 Gradient Descent recap

**Gradient Descent** is the optimization algorithm that uses these gradients to actually minimize the loss — like descending a mountain in the fog 🌫️: at each step, you look at the slope under your feet and step in the steepest downward direction.

| Component | Role |
|---|---|
| Cost function $\mathcal{L}$ | Measures the model's overall error; the quantity we want to minimize. |
| Gradient $\nabla \mathcal{L}$ | Vector of all partial derivatives of the loss; always points toward the steepest **increase**. |
| Learning rate $\eta$ | Small positive number (e.g. $0.01$) controlling the step size at each iteration. |

Update rule for a weight $w$:

$$w_{\text{new}} = w_{\text{current}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$

The **minus sign** is crucial: since the gradient points where the function increases, we step in the opposite direction to decrease it. For example, if $\dfrac{\partial \mathcal{L}}{\partial w} = +5$ (increasing $w$ increases the error), the update rule *decreases* $w$ — exactly what we want.

**Algorithm outline:**

1. Initialize weights (often randomly).
2. Compute the prediction and current error (forward pass).
3. Compute the gradient of the error with respect to every weight (backpropagation).
4. Update every weight using the rule above.
5. Repeat steps 2–4 until the error stops decreasing (convergence).

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

### Pseudo-code: Jacobian Matrix (Central Difference)

```
ALGORITHM: Calculate Jacobian Matrix

INPUT:
    - F: vector-valued function returning an array of size m
    - point: vector [x1, x2, ..., xn] of size n
    - h: step size (default = 1e-5)

OUTPUT:
    - J: Jacobian matrix of size (m x n)

BEGIN
    1. Evaluate F_ref = F(point)
    2. Set m = length(F_ref)
    3. Set n = length(point)

    4. Initialize J as a zero matrix of shape (m, n)

    5. FOR each column index 'j' from 0 to n - 1 DO:
           Create copy 'point_plus' = point
           Create copy 'point_minus' = point

           point_plus[j] = point_plus[j] + h
           point_minus[j] = point_minus[j] - h

           F_plus = F(point_plus)
           F_minus = F(point_minus)

           // Compute numerical partial derivative for all outputs with respect to variable j
           J[:, j] = (F_plus - F_minus) / (2 * h)
       END FOR

    RETURN J
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
