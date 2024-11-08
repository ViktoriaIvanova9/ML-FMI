import numpy as np
import pandas as pd
import math

from sklearn.neighbors import KNeighborsClassifier

class KNeighborsClassifierCustom:
    def __init__(self, n_neighbours):
        self.n_neighbours = n_neighbours

    def fit(self, matrix, array):
        self.X = matrix
        self.y = array

    def calc_distance(self, curr_coords, train_coords):
        dist = 0
        for i in range(len(train_coords)):
            dist += (train_coords[i]-curr_coords[i])**2

        return math.sqrt(dist)

    def divide_sets(self, point):
        distances = [self.calc_distance(point, train_point) for train_point in self.X]
        smallest_dists = np.argsort(distances)[:self.n_neighbours]
        y_list = [self.y[elem] for elem in smallest_dists]
        sum_y_list = np.sum(y_list)
        return 1 if sum_y_list > (len(y_list) / 2) else 0

    def predict(self, X_new):
        y = [self.divide_sets(row) for row in X_new]
        return y

    def score(self, X_train, y_train):
        y_pred = self.predict(X_train)
        return np.sum(y_pred == y_train) / len(y_train)
    
def custom_KNN_result(telecom_df):
    knn_custom = KNeighborsClassifierCustom(3)
    X_train = telecom_df[['account_length', 'customer_service_calls']].values
    y_train = telecom_df['churn'].values

    X_new = np.array([[30.0, 17.5],
                [107.0, 24.1],
                [213.0, 10.9]])

    knn_custom.fit(X_train, y_train)
    y_result_custom = knn_custom.predict(X_new)
    print(f' Predictions custom KNN: {y_result_custom}')
    print(f'Score custom KNN: {round(knn_custom.score(X_train, y_train), 4)}')


def sklearn_KNN_result(telecom_df):
    knn = KNeighborsClassifier(3)
    X_train = telecom_df[['account_length', 'customer_service_calls']].values
    y_train = telecom_df['churn'].values

    X_new = np.array([[30.0, 17.5],
            [107.0, 24.1],
            [213.0, 10.9]])

    knn.fit(X_train, y_train)
    y_result = knn.predict(X_new)
    print(f'Predictions sklearn KNN: {y_result}')
    print(f'Score sklearn KNN: {round(knn.score(X_train, y_train), 4)}')


def main():
    telecom_df = pd.read_csv('../DATA/telecom_churn_clean.csv', index_col=0)
    custom_KNN_result(telecom_df)
    sklearn_KNN_result(telecom_df)


if __name__ == '__main__':
    main()



