"""
Visualizing the divergence of F(x, y) = [3x^2, 2y] and the Laplacian of
f(x, y) = x^3 + y^3.

I plot:
    1. The vector field F as arrows, over a background colored by its
       divergence: red where div(F) > 0 (a "source"), blue where
       div(F) < 0 (a "sink"), with the div(F) = 0 boundary (x = -1/3)
       drawn as a dashed line.
    2. The Laplacian of f as a heatmap over the same kind of grid -
       since Laplacian(f) = 6x + 6y here, it's a plane, but seeing it
       as a heatmap makes the "curvature" reading concrete.
"""

import numpy as np
import matplotlib.pyplot as plt

from vector_calculus import compute_divergence_2d, compute_laplacian_2d


def F_x(x, y):
    return 3 * x**2


def F_y(x, y):
    return 2 * y


def f_scalar(x, y):
    return x**3 + y**3


def plot(save_path: str = "vector_calculus_plot.png") -> None:
    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x_range, y_range)

    # --- Divergence background, evaluated exactly (6x + 2) for a smooth plot ---
    DIV = 6 * X + 2

    # Coarser grid for the arrows themselves (otherwise unreadable)
    xg = np.linspace(-2, 2, 14)
    yg = np.linspace(-2, 2, 14)
    XG, YG = np.meshgrid(xg, yg)
    UG, VG = F_x(XG, YG), F_y(XG, YG)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- 1. Vector field + divergence (source vs. sink) ---
    ax1 = axes[0]
    mesh = ax1.pcolormesh(X, Y, DIV, cmap="coolwarm", vmin=-10, vmax=10, shading="auto")
    fig.colorbar(mesh, ax=ax1, label="div(F) = 6x + 2")
    # Normalize arrow lengths so the field is readable across the whole grid
    norm = np.sqrt(UG**2 + VG**2)
    norm[norm == 0] = 1
    ax1.quiver(XG, YG, UG / norm, VG / norm, color="black", scale=25, width=0.004)
    ax1.axvline(-1 / 3, color="black", linestyle="--", linewidth=1.5, label="div(F) = 0  (x = -1/3)")
    ax1.set_title("Vector field F(x, y) = [3x², 2y]\ncolored by divergence: source (red) vs. sink (blue)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_aspect("equal")

    # --- 2. Laplacian of f(x, y) = x^3 + y^3 ---
    ax2 = axes[1]
    LAP = 6 * X + 6 * Y
    mesh2 = ax2.pcolormesh(X, Y, LAP, cmap="PRGn", shading="auto")
    fig.colorbar(mesh2, ax=ax2, label="Laplacian(f) = 6x + 6y")
    ax2.contour(X, Y, LAP, levels=[0], colors="black", linewidths=1.5)
    ax2.set_title("Laplacian of f(x, y) = x³ + y³\n(local curvature of f)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    fig.suptitle("Vector calculus operators: divergence (source/sink) and Laplacian (curvature)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot()
