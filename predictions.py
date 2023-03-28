import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from FProp import forward_propagation
ITER_ACC_LIST = []

def get_predictions(A3: np.ndarray) -> np.ndarray:
    """
    takes np.ndarray m x n

    return array 1 x n with index of max value along axis
    """
    return np.argmax(A3, 0)

def get_accuracy(preds: np.ndarray, Y: np.ndarray, test = False) -> np.int32:
    """
    preds: predictions array
    Y: array
    test: boolean, if it's test or training 
    take predictions and compare them with the ground truth 

    return mean sum of right predicted
    """
    print(preds, Y)

    acc = np.sum(preds == Y) / Y.size

    if test == True:
        cf_matrix = confusion_matrix(Y, preds)
        sns.heatmap(cf_matrix, annot=True, fmt='g', cmap="mako")
        plt.show()

    return acc

def make_predictions(X, W1, b1, W2, b2, W3, b3) -> np.ndarray:
    """
    last layer of NN
    getting predictions from NN

    return predictions
    """
    _1,_2,_3,_4,_5, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)

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
    

def plot_accuracy_vs_iterations(accuracies, iterations_step=10):
    plt.plot(range(0, len(accuracies) * iterations_step, iterations_step), accuracies)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iterations")
    plt.show()
