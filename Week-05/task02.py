import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main():
    diabetes_df = pd.read_csv('../DATA/diabetes_clean.csv')

    X = diabetes_df.drop('diabetes', axis=1)
    y = diabetes_df['diabetes'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    y_predict = log_reg.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_predict)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.title('ROC curve for Diabetes Prediction')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    #  What does the plot tell you about the model's performance? - C.