def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    """
    updating weights and biases of NN

    return updated parameters
    """

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1 

    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2  

    W3 = W3 - alpha * dW3 
    b3 = b3 - alpha * db3  
    
    return W1, b1, W2, b2, W3, b3
