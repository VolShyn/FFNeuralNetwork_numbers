import numpy as np

def RelU(Z: np.array) -> np.array:
    """
    non-linear ReLU function

    return iterable of 0's and value  
    """
    return np.maximum(Z, 0)

def SoftMax(Z: np.array) -> np.array:
    """
    non-linear SoftMax function

    return iterable of values from 0 to 1
    """
    return np.exp(Z) / sum(np.exp(Z))

def forward_propagation(W1,b1,W2,b2,W3,b3,X):
    """
    realisation of 3 layers NN feed-forward propagation
    
    return Z's and A's
    """
    Z1 = W1.dot(X) + b1
    A1 = RelU(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = RelU(Z2)
    
    Z3 = W3.dot(A2) + b3
    A3 = SoftMax(Z3)

    return Z1, A1, Z2, A2, Z3, A3   

