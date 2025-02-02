import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, naive_bayes
from sklearn.feature_extraction import text


def main():
    # df = pd.read_csv(os.path.join('DATA', 'fake_or_real_news.csv'), index_col=0)

    df = pd.read_csv('../DATA/fake_or_real_news.csv', index_col=0)

    y = df['label']
    X = df['text'].values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y,
                                                                        test_size=0.33,
                                                                        random_state=52,
                                                                        stratify=y)

    count_vectorizer = text.CountVectorizer(stop_words='english')
    X_train_vectorized = count_vectorizer.fit_transform(X_train)
    X_test_vectorized = count_vectorizer.transform(X_test)

    tfidf_vectorizer = text.TfidfVectorizer(stop_words='english')
    X_train_vectorized_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_vectorized_tfidf = tfidf_vectorizer.transform(X_test)

    nb_classifier_bow = naive_bayes.MultinomialNB()
    nb_classifier_bow.fit(X_train_vectorized, y_train)
    y_pred = nb_classifier_bow.predict(X_test_vectorized)
    print(f'Accuracy when using BoW: {nb_classifier_bow.score(X_test_vectorized, y_test)}')

    f, axs = plt.subplots(1, 2, figsize=(15, 8))

    f.suptitle('Confusion matrix: BoW (left) vs TF-IDF (right)')
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axs[0])

    nb_classifier_tfidf = naive_bayes.MultinomialNB()
    nb_classifier_tfidf.fit(X_train_vectorized_tfidf, y_train)
    y_pred = nb_classifier_tfidf.predict(X_test_vectorized_tfidf)
    print(
        f'Accuracy when using TF-IDF: {nb_classifier_tfidf.score(X_test_vectorized_tfidf, y_test)}'
    )
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axs[1])

    plt.tight_layout()
    plt.show()

    alphas = np.arange(0, 1.1, 0.1)
    scores = []
    for alpha in alphas:
        nb_classifier_i = naive_bayes.MultinomialNB(alpha=alpha)
        nb_classifier_i.fit(X_train_vectorized_tfidf, y_train)
        y_pred = nb_classifier_i.predict(X_test_vectorized_tfidf)
        scores.append(nb_classifier_i.score(X_test_vectorized_tfidf, y_test))

    plt.title('Accuracy per alpha value')
    plt.plot(alphas, scores)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.xticks(alphas)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    nb_classifier = naive_bayes.MultinomialNB(alpha=0.1)
    nb_classifier.fit(X_train_vectorized_tfidf, y_train)
    class_labels = nb_classifier.classes_
    feature_names = tfidf_vectorizer.get_feature_names_out()
    feat_with_weights = sorted(zip(nb_classifier.feature_log_prob_[0], feature_names),
                               reverse=True)
    print()
    print(class_labels[0], [word for coef, word in feat_with_weights[-20:]])
    print(class_labels[1], [word for coef, word in feat_with_weights[:20]])


if __name__ == '__main__':
    main()