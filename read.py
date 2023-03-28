import numpy as np
import pandas as pd # just for comfort of reading csv files
import os 

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_file(path=f'{ROOT_DIR}/digits_dataset/train.csv') -> np.ndarray:
    """
    path = path to csv file
    function to read_csv file and convert to numpy array

    return np.array
    """
    global ROWS, COLS
    data = pd.read_csv(path) 
    data = np.array(data)
    ROWS, COLS = data.shape


    return data