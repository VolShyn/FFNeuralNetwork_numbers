import numpy as np
import matplotlib.pyplot as plt
from FProp import forward_propagation
from BackProp import backward_propagation
from init_params import init_params
from update_params import update_params

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _1,_2, _3,_4, _5, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_prediction(index, X_tr, Y_tr, W1, b1, W2, b2, W3, b3):
    current_image = X_tr[:, index, None]
    prediction = make_predictions(X_tr[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_tr[index]
    print("Передбачили: ", prediction)
    print("Лейбл: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def gradient_descent(X, Y, alpha, iterations):
    """
    gradient descent realisation

    return W1, b1, W2, b2, W3, b3
    """

    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 10 == 0:
            print("Епоха: ", i / 10)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3