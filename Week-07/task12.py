import numpy as np
from sklearn import datasets
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV

def main():

    digits_ds = datasets.load_digits()

    X = digits_ds.data
    y = np.where(digits_ds.target == 2, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    param_grid = {
        'C' : [0.1, 1, 10],
        'gamma' : [0.00001, 0.0001, 0.001, 0.01, 0.1]
    }

    svm = GridSearchCV(SVC(kernel='rbf'), param_grid).fit(X_train, y_train)

    print(f'Best CV params: {svm.best_params_}')
    print(f'Best CV accuracy: {svm.best_score_}')
    print(f'Test accuracy of best grid search hypers: {svm.score(X_test, y_test)}')


if __name__ =='__main__':
    main()

    # What do you notice about the value for the gamma hyperparameter this time? - Although increasing gamma means better accuracy,
    # combining it with best C, best acuracy goes on smaller gamma. 