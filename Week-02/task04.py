import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def train_and_test_data():
    telecom_df = pd.read_csv('../DATA/telecom_churn_clean.csv', index_col=0)

    X = telecom_df.loc[:, telecom_df.columns != 'churn'].values
    y = telecom_df['churn'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print(f'Training Dataset Shape: {X_train.shape}')
    print(f'Accuracy when "n_neighbors"=5: {round(knn.score(X_test, y_test), 4)}')

    neighbours = np.arange(1, 13)
    print(f'neighbours={neighbours}')

    train_accuracies = {}
    test_accuracies = {}

    for n_neighbours_cnt in neighbours:
        knn = KNeighborsClassifier(n_neighbors=n_neighbours_cnt)
        knn.fit(X_train, y_train)

        train_accuracies[n_neighbours_cnt] = round(knn.score(X_train, y_train), 4)
        test_accuracies[n_neighbours_cnt] = round(knn.score(X_test, y_test), 4)

    print(f'train_accuracies={train_accuracies}')
    print(f'test_accuracies={test_accuracies}')

    plt.plot(train_accuracies.keys(), train_accuracies.values())
    plt.plot(test_accuracies.keys(), test_accuracies.values())

    plt.title('KNN: Varying Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')

    plt.legend(['Training accuracy', 'Testing accuracy'])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_and_test_data()