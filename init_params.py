import numpy as np

def init_params():
    """
    function to initialize parameters for NN

    return params Wn,bn
    """


    # random from (-0.5 to 0.5) of given shape 
    W1 = np.random.rand(128,784) - 0.5
    b1 = np.random.rand(128,1) - 0.5

    W2 = np.random.rand(64,128) - 0.5
    b2 = np.random.rand(64,1) - 0.5

    W3 = np.random.rand(10,64) - 0.5
    b3 = np.random.rand(10,1) - 0.5

    return W1,b1, W2,b2, W3,b3

def init_adagrad_params():
    s_W1 = np.zeros((128,784))
    s_b1 = np.zeros((128, 1))

    s_W2 = np.zeros((64, 128))
    s_b2 = np.zeros((64, 1))

    s_W3 = np.zeros((10, 64))
    s_b3 = np.zeros((10, 1))

    return s_W1, s_b1, s_W2, s_b2, s_W3, s_b3

