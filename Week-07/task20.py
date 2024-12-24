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
from sklearn.ensemble import BaggingClassifier

def preprocess(liver_df):
    """
    Prepare the data for modeling.

    Parameters
    ----------
    liver_df: pandas DataFrame containing the data

    Returns
    -------
    X_train_prepared : ndarray with data for training
    X_test_prepared : ndarray with data for testing
    y_train : ndarray with target training data 
    y_test : ndarray with target testing data 

    """
    liver_dummies = pd.get_dummies(liver_df['Gender'], drop_first=True, dtype=int)
    liver_dummies = pd.concat([liver_df, liver_dummies], axis=1)
    liver_dummies = liver_dummies.drop(columns=['Gender'])

    X = liver_dummies.drop(['has_liver_disease'], axis=1)
    liver_dummies['has_liver_disease'] = np.where(liver_dummies['has_liver_disease'] == 2, 0, 1)
    y = liver_dummies['has_liver_disease'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
    pipeline = Pipeline(steps)

    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared =  pipeline.transform(X_test)

    return X_train_prepared, X_test_prepared, y_train, y_test


def main():
    column_names = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G ratio', 'has_liver_disease']
    liver_df = pd.read_csv('../DATA/indian_liver_patient_dataset.csv', names=column_names)

    X_train_prepared, X_test_prepared, y_train, y_test = preprocess(liver_df)

    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=1), 
                                n_estimators=50, 
                                random_state=1).fit(X_train_prepared, y_train)

    y_pred_bagging = bagging.predict(X_test_prepared)

    dt = DecisionTreeClassifier(random_state=1).fit(X_train_prepared, y_train)
    y_pred_dt = dt.predict(X_test_prepared)

    print(f'Test set f1-score of aggregator: {round(f1_score(y_test, y_pred_bagging), 2)}')
    print(f'Test set f1-score of single decision tree: {round(f1_score(y_test, y_pred_dt), 2)}')

if __name__ == '__main__':
    main()