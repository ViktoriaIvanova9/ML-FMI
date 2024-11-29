import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

def main():
    diabetes_df = pd.read_csv('../DATA/diabetes_clean.csv')

    X = diabetes_df[['bmi', 'age']].values
    y = diabetes_df['diabetes'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train, y_train)
    y_predictions = knn.predict(X_test)

    print(classification_report(y_test, y_predictions, target_names=['No diabetes', 'Diabetes']))

    ConfusionMatrixDisplay.from_predictions(y_test, y_predictions, display_labels=['No diabetes', 'Diabetes'])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    main()

    # How many true positives were predicted? - 20
    # How many false positives were predicted? - 16
    # For which class is the f1-score higher? - for No diabetes