import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)

def main():

    digits_ds = datasets.load_digits()

    X = digits_ds.data
    y = digits_ds.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    logreg = LogisticRegression().fit(X_train, y_train)
    svm = SVC().fit(X_train, y_train)

    print(f'Dataset shape: {digits_ds.data.shape}')
    print(f'Number of classes: {len(digits_ds.target_names)}')
    print(f'Training accuracy of logistic regression: {logreg.score(X_train, y_train)}')
    print(f'Validation accuracy of logistic regression: {logreg.score(X_test, y_test)}')
    print(f'Training accuracy of non-linear support vector classifier: {svm.score(X_train, y_train)}')
    print(f'Validation accuracy of non-linear support vector classifier: {svm.score(X_test, y_test)}')
    # don't understand why I receive the correct output for ligreg and not for svc

    _, ax = plt.subplots(1, 5, figsize=(10, 3), sharey=True)
    plt.gray()

    for i in range(5):
        current_num = np.random.randint(0, 7)
        ax[i].matshow(digits_ds.images[current_num])
        ax[i].set_title(current_num)

    plt.show()


if __name__ =='__main__':
    main()

    #  Which is the better classifier and why? - The non-linear SVC has better validation accuracy, so it is a better classifier