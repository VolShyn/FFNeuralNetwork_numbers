import numpy as np
from one_hot import one_hot_encode

def RelU_deriv(Z: np.ndarray) -> np.ndarray:
    """
    function to return derivative of RelU

    return np.array
    """
    return Z > 0

def backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    """
    function realisation of back propagation algorithm
    for 3 layers perceptron

    return dWn, dbn
    """

    rows, cols = X.shape

    one_hot_Y = one_hot_encode(Y)

    dZ3 = A3 - one_hot_Y
    dW3 = 1 / rows * dZ3.dot(A2.T)
    db3 = 1 / rows * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * RelU_deriv(Z2)
    dW2 = 1 / rows * dZ2.dot(A1.T)
    db2 = 1 / rows * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * RelU_deriv(Z1)
    dW1 = 1 / rows * dZ1.dot(X.T)
    db1 = 1 / rows * np.sum(dZ1)

    return  dW1, db1, dW2, db2, dW3, db3