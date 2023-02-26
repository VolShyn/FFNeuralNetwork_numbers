import numpy as np


def one_hot_encode(Y: np.ndarray) -> np.ndarray:
      """
      func realisation of one hot encoding
      
      return one hot encoded version of array
      """

      one_hot_y = np.zeros((Y.size, Y.max()+1))
      one_hot_y[np.arange(Y.size), Y] = 1
      
      return one_hot_y.T