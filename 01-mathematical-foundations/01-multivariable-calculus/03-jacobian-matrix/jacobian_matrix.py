import numpy as np

def compute_jacobian(F, point, h=1e-5):
    """
    Numerically computes the Jacobian matrix of a vector-valued function F
    at the evaluation point 'point', using centered finite differences.

    Parameters:
        F     : function that takes a NumPy vector (size n) and returns a NumPy vector (size m)
        point : NumPy vector or list [x1, x2, ..., xn] input point
        h     : step size for the finite differences (default 1e-5)

    Returns:
        J     : NumPy Jacobian matrix of shape (m, n)
    """
    point = np.array(point, dtype=float)

    # 1. Reference evaluation to know the output size (m)
    F_ref = F(point)
    m = len(F_ref)
    n = len(point)

    # 2. Initialize the Jacobian matrix (m x n) filled with zeros
    J = np.zeros((m, n))

    # 3. Compute each column j of the Jacobian (sensitivity with respect to variable x_j)
    for j in range(n):
        point_plus = np.copy(point)
        point_minus = np.copy(point)

        # Perturbation of the j-th variable
        point_plus[j] += h
        point_minus[j] -= h

        # Perturbed outputs
        F_plus = F(point_plus)
        F_minus = F(point_minus)

        # Centered finite difference for column j: (F(x+h) - F(x-h)) / (2h)
        J[:, j] = (F_plus - F_minus) / (2 * h)

    return J

# =====================================================================
# Demonstration examples
# =====================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" JACOBIAN MATRIX CALCULATOR (NUMPY)")
    print("==================================================\n")

    # --- Example 1: R^2 -> R^2 function (our hand-computed example) ---
    # F(x, y) = [x^2 * y, x + 3y]
    def example_function(v):
        x, y = v[0], v[1]
        f1 = (x**2) * y
        f2 = x + 3 * y
        return np.array([f1, f2])

    point_1 = [2.0, 3.0]
    J1 = compute_jacobian(example_function, point_1)

    print("--- EXAMPLE 1: F(x, y) = [x^2 * y, x + 3y] ---")
    print(f"Evaluation point: {point_1}")
    print(f"Jacobian matrix shape (m x n): {J1.shape}")
    print("Jacobian matrix J(2, 3):")
    print(J1)
    print("Expected theoretical result: [[12.  4.]")
    print("                              [ 1.  3.]]\n")

    # --- Example 2: Neural Network Layer R^3 -> R^2 ---
    # Z = W * X + B (2 output neurons, 3 input variables)
    W = np.array([
        [0.5, -1.2,  2.0],  # Neuron 1 weights
        [1.5,  0.8, -0.4]   # Neuron 2 weights
    ])
    b = np.array([0.1, -0.5])

    def neural_layer(X):
        return np.dot(W, X) + b

    X_in = [1.0, 2.0, -1.0]
    J2 = compute_jacobian(neural_layer, X_in)

    print("--- EXAMPLE 2: Dense Layer R^3 -> R^2 (Neural Network) ---")
    print(f"Input vector X: {X_in}")
    print(f"Jacobian matrix shape: {J2.shape} (2 outputs, 3 inputs)")
    print("Jacobian matrix dZ/dX:")
    print(J2)
    print("\nInitial weight matrix W:")
    print(W)
    print("-> Note: the Jacobian of a linear layer matches the weight matrix W exactly!")
    print("==================================================")
