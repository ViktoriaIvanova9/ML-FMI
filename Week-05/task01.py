import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    diabetes_df = pd.read_csv('../DATA/diabetes_clean.csv')

    X = diabetes_df.drop('diabetes', axis=1)
    y = diabetes_df['diabetes'].values

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    log_reg = LogisticRegression(max_iter=10000, random_state=42)
    log_reg.fit(X_train, y_train)
    print(log_reg.predict_proba(X_test)[:10, 1])

if __name__ == '__main__':
    main()