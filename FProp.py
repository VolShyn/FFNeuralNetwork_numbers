import numpy as np
from typings import Iterable

def RelU(Z: Iterable) -> np.array:
    """
    non-linear ReLU function

    return iterable of 0's and value  
    """
    return np.maximum(Z, 0)

def SoftMax(Z: Iterable) -> np.array:
    """
    non-linear SoftMax function

    return iterable of values from 0 to 1
    """
    A = np.exp(Z) / sum(np.exp(Z))

    return A

def forward_propagation(W1,b1,W2,b2,W3,b3,X):
    """
    realisation of 3 layers NN feed-forward propagation
    
    return Z's and A's
    """
    Z1 = W1.dot(X) + b1
    A1 = RelU(Z1)

    Z2 = W2.dot(A2) + b2
    A2 = RelU(Z2)
    
    Z3 = W3.dot(A2) + b3
    A3 = SoftMax(Z3)

    return Z1, A1, Z2, A2, Z3, A3   

