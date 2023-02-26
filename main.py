import numpy as np
import pickle
import os 
import pandas as pd # just for comfort of reading csv 
from train_test_split import train_test_split
from grad_desc import gradient_descent, test_prediction


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def train():   
    """
    train model and save weights as pickle file
    """
    global X_train, Y_train
    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.05, 1000)

    data = {'weights':(W1,W2,W3), 'biases': (b1,b2,b3)}
    with open('weights.pickle', 'wb') as f:
        pickle.dump(data, f)

data = pd.read_csv(f'{ROOT_DIR}/digits_dataset/train.csv') 
data = np.array(data)

rows, cols = data.shape
print(f'Rows: {rows}, Col: {cols}')

X_train, Y_train, X_test, Y_test = train_test_split(data)
print(f'Y_train_shape: {Y_train.shape}')
print(Y_train)

try:
     # Завантажуємо ваги з пікла
     with open('weights.pickle', 'rb') as f:
          weights = pickle.load(f)
     W1, W2, W3 = weights['weights']
     b1, b2, b3 = weights['biases']

except FileNotFoundError:
     print('Не вдалося завантажити ваги моделі!')
     train()

try:
     for i in range(4):
          test_prediction(i, X_train, Y_train, W1, b1, W2, b2, W3, b3)
except NameError:
     print('Немає w')