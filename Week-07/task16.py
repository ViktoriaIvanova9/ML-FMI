from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

def main():
    cancer_ds = datasets.load_breast_cancer()

    X = cancer_ds.data
    y = cancer_ds.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    dt_entropy = DecisionTreeClassifier(max_depth=4, random_state=1, criterion='entropy').fit(X_train, y_train)
    dt_gini = DecisionTreeClassifier(max_depth=4, random_state=1, criterion='gini').fit(X_train, y_train)

    print(f'Accuracy achieved by using entropy: {round(dt_entropy.score(X_test, y_test), 3)}') # why gini is default since entropy is better?
    print(f'Accuracy achieved by using the gini index: {round(dt_gini.score(X_test, y_test), 3)}')

if __name__ == '__main__':
    main()