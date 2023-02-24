import numpy as np


def one_hot_encode(Y: np.array) -> np.array:
	"""
	function to convert array into one-hot encoded array

	return one-hot encoded array
	"""

    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y