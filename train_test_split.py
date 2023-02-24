from typing import Iterable
import logging 
import numpy as np

def train_test_split(data: Iterable, percentage = 0.9):
    """
    function to split data into training and testing sets

    return X_train, Y_train, X_test, Y_test
    """


    rows, cols = data.shape
    needed_len = int(len(data) * percentage)


    try:
        data = data.to_numpy()
    except Exception as e:
        logging.exception(e)

    # Shuffle before splitting
    np.random.shuffle(data)

    # create boolean mask to split data
    # mask = np.random.rand(len(data)) =< percentage
    
    #Test set
    test_data = data[needed_len: rows].T
    X_test = test_data[1:cols]
    Y_test = test_data[0]

    X_test = X_test / 255

    #Train set
    train_data = data[0:needed_len].T
    X_train = train_data[1:cols]
    Y_train = train_data[0]

    X_train = X_train / 255

    return X_train, Y_train, X_test, Y_test
