import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def main():
    news_df = pd.read_csv('../DATA/fake_or_real_news.csv', index_col=0)

    print(news_df[:7])
    print()

    print('Distribution of labels')
    count = news_df['label'].value_counts()
    proportion = news_df['label'].value_counts(normalize=True)

    label_distribution = pd.DataFrame({ 'count': count, 'proportion': proportion })
    print(label_distribution)

    X = news_df['title'].values
    y = news_df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

    vectorizer = CountVectorizer(stop_words='english',  analyzer = 'word')
    X_vectorized = vectorizer.fit_transform(X_train)
    count_features = vectorizer.get_feature_names_out()

    print(f'First 10 tokens: {X_vectorized.toarray()[:10]}')
    print(f'Size of vocabulary: {len(count_features)}')

    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer = 'word', max_df=0.7)
    X_tf_idf_vectorized = tf_idf_vectorizer.fit_transform(X_train.tolist())
    print(X_tf_idf_vectorized.toarray())

    # Here I understand the idea of CountVectorizer and TfIdfVectorizer - to transform a text to bag-of-words/matrix
    # but don't know whether the preprocessing is bad or I do sth to not receive the correct result(only zeroes when I convert .toarray())

if __name__ == '__main__':
    main()
 