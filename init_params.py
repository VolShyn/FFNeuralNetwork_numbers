import numpy as np

def init_params():
    """
    function to initialize parameters for NN

    return params Wn,bn
    """


    # random from (-0.5 to 0.5) of given shape 
    W1 = np.random.rand(784,784) - 0.5
    b1 = np.random.rand(784,1) - 0.5

    W2 = np.random.rand(10,784) - 0.5
    b2 = np.random.rand(10,1) - 0.5

    W3 = np.random.rand(10,10) - 0.5
    b3 = np.random.rand(10,1) - 0.5

    return W1,b1, W2,b2, W3,b3
    