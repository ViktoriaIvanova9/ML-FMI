import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score

def main():
    diabetes_df = pd.read_csv('../DATA/diabetes_clean.csv')

    X = diabetes_df.drop('diabetes', axis=1)
    y = diabetes_df['diabetes'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    y_predict_knn = knn.predict(X_test)
    print("Model KNN trained!")

    print()

    log_reg = LogisticRegression(random_state=42).fit(X_train, y_train)
    y_predict_log_reg = log_reg.predict(X_test)
    print("Model LogisticRegression trained!")

    y_predict_prob_knn = knn.predict_proba(X_test)[:, 1]
    print(f'KNN AUC: {roc_auc_score(y_test, y_predict_prob_knn)}')
    print(f'KNN Metrics: ')
    print(classification_report(y_test, y_predict_knn))

    y_predict_prob__log_reg = log_reg.predict_proba(X_test)[:, 1]
    print(f'LogisticRegression AUC: {roc_auc_score(y_test, y_predict_prob__log_reg)}')
    print(f'LogisticRegression Metrics:')
    print(classification_report(y_test, y_predict_log_reg))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    display1 = ConfusionMatrixDisplay.from_predictions(y_test, y_predict_knn)
    display2 = ConfusionMatrixDisplay.from_predictions(y_test, y_predict_log_reg)
    display1.plot(ax=ax1)
    display2.plot(ax=ax2)
    ax1.set_title("KNN")
    ax2.set_title("LogisticRegresion")
    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()

    #  Which model performs better on the test set and why? - The Logistic regresion model is better firstly because the AUC is
    #  closer to 1 and precision, recall and f1 are more for Logistic Regression.