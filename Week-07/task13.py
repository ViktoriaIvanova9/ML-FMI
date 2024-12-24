import numpy as np
from sklearn import datasets
from sklearn.linear_model import  SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,  GridSearchCV


def main():
    digits_ds = datasets.load_digits()

    X = digits_ds.data
    y = np.where(digits_ds.target == 2, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = { # don't understand when we need logreg__C and when only C
        'alpha' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        'loss' : ['hinge', 'log_loss']
    }

    # ! remember - the only difference between the linear regression and Linear SVC is the loss function
    classifier = GridSearchCV(SGDClassifier(random_state=0), param_grid).fit(X_train, y_train)

    print(f'Best CV params: {classifier.best_params_}')
    print(f'Best CV accuracy: {classifier.best_score_}')
    print(f'Test accuracy of best grid search hypers: {classifier.score(X_test, y_test)}')


if __name__ =='__main__':
    main()