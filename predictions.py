import numpy as np
import matplotlib.pyplot as plt
from FProp import forward_propagation

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
    
