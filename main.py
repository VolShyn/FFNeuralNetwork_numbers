import numpy as np
import pandas as pd # just for comfort of reading csv 
from train_test_split import train_test_split
from grad_desc import gradient_descent, test_prediction


if __name__ == '__main__':

     data = pd.read_csv('/home/volodymyr/Projects/FFNeuralNetwork_numbers/digits_dataset/train.csv') 

     data = np.array(data)
     m, n = data.shape
     
     print(m, n)
     
     X_train, Y_train, X_test, Y_test = train_test_split(data)

     print(Y_train.shape)

     W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.5, 100)
     

     test_prediction(0,X_train, Y_train, W1, b1, W2, b2, W3, b3)
     test_prediction(1,X_train, Y_train, W1, b1, W2, b2, W3, b3)
     # test_prediction(2, W1, b1, W2, b2, W3, b3)
     # test_prediction(3, W1, b1, W2, b2, W3, b3)