import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    music_df = pd.read_csv('../DATA/music_clean.csv', index_col=0)

    X = music_df.drop('popularity', axis=1)
    music_df['popularity'] = np.where(music_df['popularity'] >= np.median(music_df['popularity']), 1, 0)
    y = music_df['popularity'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kf = KFold(n_splits=6, shuffle=True, random_state=12)

    log_reg = LogisticRegression()
    log_reg_score = cross_val_score(log_reg, X_train_scaled, y_train, cv=kf)

    knn = KNeighborsClassifier()
    knn_score = cross_val_score(knn, X_train_scaled, y_train, cv=kf)

    dec_tree = DecisionTreeClassifier()
    dec_tree_score = cross_val_score(dec_tree, X_train_scaled, y_train, cv=kf) 

    plt.boxplot([log_reg_score, knn_score, dec_tree_score], tick_labels=['Logistic regression', 'KNN', 'Decision Tree Classifier'])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    # Which model performs best? - Logistic regression