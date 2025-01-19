from mod_class import Mod
import numpy as np


def gauss_elimination_inverse_mod(matrix):
    n = matrix.shape[0]

    # Ensure the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Only square matrices can be inverted.")

    # Create an augmented matrix with the identity matrix on the right-hand side
    prime = matrix[0, 0].n  # Assuming all elements are Mod with the same prime
    identity = np.array([[Mod(1 if i == j else 0, prime) for j in range(n)] for i in range(n)], dtype=object)
    augmented = np.hstack((matrix, identity))

    for i in range(n):
        # Pivot: Make sure the diagonal element is not zero
        if augmented[i, i] == Mod(0, prime):
            # Find a row below the current one with a nonzero value in this column and swap
            for j in range(i + 1, n):
                if augmented[j, i] != Mod(0, prime):
                    augmented[[i, j]] = augmented[[j, i]]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")

        # Normalize the pivot row
        augmented[i] = [x / augmented[i, i] for x in augmented[i]]

        # Eliminate the other entries in this column
        for j in range(n):
            if i != j:
                factor = augmented[j, i]
                augmented[j] = [augmented[j, k] - factor * augmented[i, k] for k in range(2 * n)]

    # The right-hand side of the augmented matrix is now the inverse
    inverse = augmented[:, n:]
    return inverse


if __name__ == "__main__":
    import numpy as np

    prime = 5
    A = np.array([[Mod(2, prime), Mod(1, prime), Mod(1, prime)],
                  [Mod(1, prime), Mod(3, prime), Mod(2, prime)],
                  [Mod(1, prime), Mod(0, prime), Mod(0, prime)]], dtype=object)

    try:
        A_inv = gauss_elimination_inverse_mod(A)
        print("Inverse of the matrix:")
        for row in A_inv:
            print(row)

        # Verify the result
        print("Verification (A @ A_inv):")
        result = A @ A_inv 
        print(result)

    except ValueError as e:
        print(e)
