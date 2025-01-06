import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def main():
    news_df = pd.read_csv('../DATA/fake_or_real_news.csv')

    print(news_df[:8])
    print('Distribution of labels:')

    count_labels = news_df['label'].value_counts(normalize=False)
    proportion_labels = news_df['label'].value_counts(normalize=True)

    label_distribution = pd.DataFrame({
        'count': count_labels,
        'proportion': proportion_labels
    })
    print(label_distribution)

    X = news_df[['title', 'text']]
    y = news_df['label'].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

    vectorizer = CountVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(news_df)
    count_features = vectorizer.get_feature_names_out()

    print(f'First 10 tokens: {X_vectorized.toarray()}')
    print(f'Size of vocabulary: {len(count_features)}')

if __name__ == '__main__':
    main()
 