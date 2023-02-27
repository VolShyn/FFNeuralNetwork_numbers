import numpy as np
import timeit   
from one_hot import one_hot_encode
from init_params import init_params
from update_params import update_params
from FProp import forward_propagation
from predictions import get_accuracy, get_predictions

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
    one_hot_Y = one_hot_encode(Y)
    from read import ROWS 

    dZ3 = A3 - one_hot_Y
    dW3 = 1 / ROWS * dZ3.dot(A2.T)
    db3 = 1 / ROWS * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * RelU_deriv(Z2)
    dW2 = 1 / ROWS * dZ2.dot(A1.T)
    db2 = 1 / ROWS * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * RelU_deriv(Z1)
    dW1 = 1 / ROWS * dZ1.dot(X.T)
    db1 = 1 / ROWS * np.sum(dZ1)

    return  dW1, db1, dW2, db2, dW3, db3

def gradient_descent(X, Y, alpha, iterations):
    """
    gradient descent realisation

    return W1, b1, W2, b2, W3, b3
    """

    W1, b1, W2, b2, W3, b3 = init_params()

    starttime = timeit.default_timer()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 10 == 0:
            print("Ітерації:  ", i)
            print(f"{timeit.default_timer() - starttime:.3f}c")
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))

    return W1, b1, W2, b2, W3, b3