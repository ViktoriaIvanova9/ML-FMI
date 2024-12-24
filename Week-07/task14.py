import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

def main():
    cancer_ds = datasets.load_breast_cancer()

    X = cancer_ds.data[:, [0,7]]
    y = cancer_ds.target

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y) # why no stratify=y??

    dt = DecisionTreeClassifier(max_depth=6, random_state=1)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    print(f'First 5 predictions: {y_pred[:5]}') 
    print(f'Test set accuracy: {round(dt.score(X_test, y_test), 2)}')

if __name__ == '__main__':
    main()