import pickle
from train_test_split import train_test_split
from predictions import get_accuracy,make_predictions,test_prediction, plot_accuracy_vs_iterations

def train():   
    """
    train model and save weights target_backendas pickle file
    """
    global X_train, Y_train
    from BackProp import stochastic_adagrad

    W1, b1, W2, b2, W3, b3, accuracies = stochastic_adagrad(X_train, Y_train, 0.01, 5000)
    plot_accuracy_vs_iterations(accuracies)

    data = {'weights':(W1,W2,W3), 'biases': (b1,b2,b3)}
    with open('weights.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    from read import read_file
    # Зчитуємо дані
    data = read_file()
    ROWS, COLS = data.shape
    print(f'Rows: {ROWS}, Col: {COLS}') 

    # Ділимо на тестування/тренування
    X_train, Y_train, X_test, Y_test = train_test_split(data, 0.15)
    print(f'Y_train_shape: {Y_train.shape}')
    print(Y_train) 

    try:
        # Завантажуємо ваги з пікл файлу
        with open('weights.pickle', 'rb') as f:
            weights = pickle.load(f)
        W1, W2, W3 = weights['weights']
        b1, b2, b3 = weights['biases']
    except FileNotFoundError:
        print('Не вдалося завантажити ваги моделі!')
        train()
        
    
    try:
        i = 0
        while True:
            i+=1
            test_prediction(i, X_train, Y_train, W1, b1, W2, b2, W3, b3)
    except KeyboardInterrupt:
        print('Було натиснуто Ctrl + C')

    # Подивимось результати на тестовій виборці
    try:
        y_hat = make_predictions(X_test,W1, b1, W2, b2, W3, b3 )
        print(f'Точність генералізації {get_accuracy(y_hat, Y_test, test=True)}')
    except NameError:
        print('Не вийшло отримати точність на тестових даних')
    