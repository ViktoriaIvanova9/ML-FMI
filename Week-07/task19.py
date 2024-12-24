import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

def main():
    column_names = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G ratio', 'has_liver_disease']
    liver_df = pd.read_csv('../DATA/indian_liver_patient_dataset.csv', names=column_names)
    print(liver_df)

    liver_dummies = pd.get_dummies(liver_df['Gender'], drop_first=True, dtype=int)
    liver_dummies = pd.concat([liver_df, liver_dummies], axis=1)
    liver_dummies = liver_dummies.drop(columns=['Gender'])

    X = liver_dummies.drop(['has_liver_disease'], axis=1)
    liver_dummies['has_liver_disease'] = np.where(liver_dummies['has_liver_disease'] == 2, 0, 1)
    y = liver_dummies['has_liver_disease'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    steps = [('imputer', SimpleImputer(strategy='most_frequent')),
             ('scaler', StandardScaler())]
    pipeline = Pipeline(steps)

    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared =  pipeline.transform(X_test)

    logreg = LogisticRegression(random_state=1).fit(X_train_prepared, y_train)
    dt = DecisionTreeClassifier(max_depth=4, random_state=1, min_samples_leaf=0.13).fit(X_train_prepared, y_train)
    knn = KNeighborsClassifier(n_neighbors=27).fit(X_train_prepared, y_train)

    voting = VotingClassifier(estimators=[('logreg', logreg), ('dt', dt), ('knn', knn)]).fit(X_train_prepared, y_train)

    y_pred_logreg = logreg.predict(X_test_prepared)
    y_pred_dt = dt.predict(X_test_prepared)
    y_pred_knn = knn.predict(X_test_prepared)

    y_pred_voting = voting.predict(X_test_prepared)

    print(f'Logistic Regression: {round(f1_score(y_test, y_pred_logreg), 3)}')
    print(f'K Nearest Neighbours: {round(f1_score(y_test, y_pred_knn), 3)}')
    print(f'Classification Tree: {round(f1_score(y_test, y_pred_dt), 3)}')
    print()
    print(f'Voting Classifier: {round(f1_score(y_test, y_pred_voting), 3)}')

if __name__ == '__main__':
    main()