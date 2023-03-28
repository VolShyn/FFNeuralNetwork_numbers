import numpy as np
import timeit   
from one_hot import one_hot_encode
from init_params import init_params, init_adagrad_params
from update_params import update_params
from FProp import forward_propagation
from predictions import get_accuracy, get_predictions

def RelU_deriv(Z: np.ndarray) -> np.ndarray:
    return Z > 0

def standardize_weights(W):
    mean = np.mean(W, axis=1, keepdims=True)
    std = np.std(W, axis=1, keepdims=True) + 1e-7
    W_standardized = (W - mean) / std

    return W_standardized

def backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot_encode(Y, 10)
    from read import ROWS 

    dZ3 = A3 - one_hot_Y
    dW3 = 1 / ROWS * dZ3.dot(A2.T)
    db3 = 1 / ROWS * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * RelU_deriv(Z2)
    dW2 = 1 / ROWS * dZ2.dot(A1.T)
    db2 = 1 / ROWS * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * RelU_deriv(Z1)
    dW1 = 1 / ROWS * dZ1.dot(X)
    db1 = 1 / ROWS * np.sum(dZ1)

    return  dW1, db1, dW2, db2, dW3, db3

def stochastic_adagrad(X, Y, alpha, iterations, batch_size=64, epsilon=1e-8):
    W1, b1, W2, b2, W3, b3 = init_params()
    X = X.T
    accuracies = []
    # Initialize AdaGrad squared gradients
    s_W1, s_b1, s_W2, s_b2, s_W3, s_b3 = init_adagrad_params()

    starttime = timeit.default_timer()
    for i in range(iterations):
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        X_batch = X[indices]
        Y_batch = Y[indices]
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X_batch.T)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch)

        # Update AdaGrad squared gradients
        s_W1 += dW1 * dW1
        s_b1 += db1 * db1
        s_W2 += dW2 * dW2
        s_b2 += db2 * db2
        s_W3 += dW3 * dW3
        s_b3 += db3 * db3

        # Update parameters with adaptive learning rates
        W1 -= alpha * dW1 / (np.sqrt(s_W1) + epsilon)
        b1 -= alpha * db1 / (np.sqrt(s_b1) + epsilon)
        W2 -= alpha * dW2 / (np.sqrt(s_W2) + epsilon)
        b2 -= alpha * db2 / (np.sqrt(s_b2) + epsilon)
        W3 -= alpha * dW3 / (np.sqrt(s_W3) + epsilon)
        b3 -= alpha * db3 / (np.sqrt(s_b3) + epsilon)

        if i % 10 == 0:
            print("Ітерації:  ", i)
            print(f"{timeit.default_timer() - starttime:.3f}c")
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y_batch)
            print(accuracy)
            accuracies.append(accuracy)

    return W1, b1, W2, b2, W3, b3, accuracies


# def stochastic_gradient_descent(X, Y, alpha, iterations, batch_size=64):
    W1, b1, W2, b2, W3, b3 = init_params()
    X = X.T

    starttime = timeit.default_timer()
    for i in range(iterations):
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        X_batch = X[indices]
        Y_batch = Y[indices]
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(standardize_weights(W1), b1, standardize_weights(W2), b2, W3, b3, X_batch.T)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 10 == 0:
            print("Ітерації:  ", i)
            print(f"{timeit.default_timer() - starttime:.3f}c")
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y_batch))

    return W1, b1, W2, b2, W3, b3


# def gradient_descent(X, Y, alpha, iterations):
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