"""
Visualizing the partial derivatives of f(x, y) = x^2 * y + 4 * y^2.

I plot:
    1. The 3D surface of f(x, y), with the evaluation point marked.
    2. The "slice" in x (y held fixed): a cut through the surface where
       only x moves. The slope of this curve at the evaluation point
       is exactly df/dx.
    3. The "slice" in y (x held fixed): same idea, but y is the one moving.
       The slope of this curve is df/dy.

This is exactly the intuition from the theory: freeze one variable and
look only at the effect of the other.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for projection='3d')


def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Function under study: f(x, y) = x^2 * y + 4 * y^2."""
    return (x**2) * y + 4 * (y**2)


def partial_derivative_x(func, x: float, y: float, h: float = 1e-5) -> float:
    return (func(x + h, y) - func(x - h, y)) / (2 * h)


def partial_derivative_y(func, x: float, y: float, h: float = 1e-5) -> float:
    return (func(x, y + h) - func(x, y - h)) / (2 * h)


def plot(x0: float = 1.0, y0: float = 2.0, save_path: str = "partial_derivatives_plot.png") -> None:
    df_dx = partial_derivative_x(f, x0, y0)
    df_dy = partial_derivative_y(f, x0, y0)
    z0 = f(x0, y0)

    fig = plt.figure(figsize=(15, 5))

    # --- 1. 3D surface with the evaluation point ---
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    x_range = np.linspace(x0 - 2, x0 + 2, 60)
    y_range = np.linspace(y0 - 2, y0 + 2, 60)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)
    ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.75, linewidth=0)
    ax1.scatter([x0], [y0], [z0], color="red", s=60, label=f"({x0}, {y0}, {z0:.1f})")
    ax1.set_title("f(x, y) = x²y + 4y² — surface")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")
    ax1.legend()

    # --- 2. Slice in x: y = y0 fixed, x varies ---
    ax2 = fig.add_subplot(1, 3, 2)
    x_slice = np.linspace(x0 - 2, x0 + 2, 200)
    z_slice_x = f(x_slice, y0)
    ax2.plot(x_slice, z_slice_x, color="steelblue")
    ax2.scatter([x0], [z0], color="red", zorder=5)
    tangent_x = z0 + df_dx * (x_slice - x0)
    ax2.plot(x_slice, tangent_x, "r--", label=f"tangent, slope = ∂f/∂x = {df_dx:.2f}")
    ax2.set_title(f"Slice in x (y = {y0} fixed)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x, y0)")
    ax2.legend()

    # --- 3. Slice in y: x = x0 fixed, y varies ---
    ax3 = fig.add_subplot(1, 3, 3)
    y_slice = np.linspace(y0 - 2, y0 + 2, 200)
    z_slice_y = f(x0, y_slice)
    ax3.plot(y_slice, z_slice_y, color="seagreen")
    ax3.scatter([y0], [z0], color="red", zorder=5)
    tangent_y = z0 + df_dy * (y_slice - y0)
    ax3.plot(y_slice, tangent_y, "r--", label=f"tangent, slope = ∂f/∂y = {df_dy:.2f}")
    ax3.set_title(f"Slice in y (x = {x0} fixed)")
    ax3.set_xlabel("y")
    ax3.set_ylabel("f(x0, y)")
    ax3.legend()

    fig.suptitle("Partial derivatives = slopes of the surface's slices", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
