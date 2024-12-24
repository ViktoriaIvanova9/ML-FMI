import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    digits_ds = datasets.load_digits()

    X = digits_ds.data
    y = digits_ds.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    lr_ovr = LogisticRegression(multi_class='ovr').fit(X_train, y_train)
    lr_multinomial = LogisticRegression(multi_class='multinomial').fit(X_train, y_train)

    print(f'OVR training accuracy: {lr_ovr.score(X_train, y_train)}')
    print(f'OVR test accuracy    : {lr_ovr.score(X_test, y_test)}')
    print(f'Softmax training accuracy: {lr_multinomial.score(X_train, y_train)}')
    print(f'Softmax test accuracy    : {lr_multinomial.score(X_test, y_test)}') # here again I receive a different result


if __name__ =='__main__':
    main()