import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve

def main():
    with open("../DATA/music_dirty_missing_vals.txt") as music_file:
        music_to_json = json.load(music_file)

    music_df = pd.DataFrame(music_to_json)
    music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)

    X = music_df.drop('genre', axis=1)
    y = music_df['genre'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    steps = [('imputer', SimpleImputer(strategy="median")),
             ('knn', KNeighborsClassifier(n_neighbors=3))]
    pipeline = Pipeline(steps)

    y_predict = pipeline.fit(X_train, y_train).predict(X_test)

    print(classification_report(y_test, y_predict))
    ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_predict)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate(Positive label: 1)')
    plt.ylabel('True Positive Rate(Positive label: 1)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

    # Does the model perform well on unseen data? - No, the model is not improved much at all since the false positive and false negative 
    # amount is high and the graphics is just a bit better than without using a model.