import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

def main():

    digits_ds = datasets.load_digits()

    X = digits_ds.data
    y = np.where(digits_ds.target == 2, 1, 0)

    param_grid = {
        'gamma' : [0.00001, 0.0001, 0.001, 0.01, 0.1]
    }

    svm = GridSearchCV(SVC(kernel='rbf'), param_grid).fit(X, y)

    print(f'Best CV parameters: {svm.best_params_}')
    print(f'Best CV accuracy: {svm.best_score_}')


if __name__ =='__main__':
    main()

