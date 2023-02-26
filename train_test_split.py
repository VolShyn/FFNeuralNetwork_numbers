import numpy as np

def train_test_split(data: np.ndarray, p = 0.2) -> tuple:
    """
    function to split data into training and testing sets
    data: data
    p: % part of data for test

    return X_train, Y_train, X_test, Y_test
    """


    rows, cols = data.shape
    _plen = int(len(data) * p)

    # Shuffle before splitting
    np.random.shuffle(data)

    # create boolean mask to split data
    # mask = np.random.rand(len(data)) =< percentage
    
    #Test set
    test_data = data[0:_plen].T
    X_test = test_data[1:cols]
    Y_test = test_data[0]

    X_test = X_test / 255

    #Train set
    train_data = data[_plen:rows].T
    X_train = train_data[1:cols]
    Y_train = train_data[0]

    X_train = X_train / 255

    return X_train, Y_train, X_test, Y_test
          