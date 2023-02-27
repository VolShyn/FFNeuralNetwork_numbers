import numpy as np
import matplotlib.pyplot as plt
from FProp import forward_propagation

def get_predictions(A3: np.ndarray) -> np.ndarray:
    """
    takes np.ndarray m x n

    return array 1 x n with index of max value along axis
    """
    return np.argmax(A3, 0)

def get_accuracy(preds: np.ndarray, Y: np.ndarray) -> np.int32:
    """
    take predictions and compare them with the ground truth 

    return mean sum of right predicted
    """
    print(preds, Y)
    return np.sum(preds == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2, W3, b3) -> np.ndarray:
    """
    last layer of NN
    getting predictions from NN

    return predictions
    """
    _1,_2, _3,_4, _5, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    preds = get_predictions(A3)
    return preds

def test_prediction(index, X_tr, Y_tr, W1, b1, W2, b2, W3, b3):
    """
    taking ground truth image, plotting it
    then comparing it label with predicted value
    """
    current_image = X_tr[:, index, None]
    pred = make_predictions(X_tr[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_tr[index]
    print("Передбачили: ", pred)
    print("Лейбл: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
