import collections
from nltk.corpus import stopwords
import ast
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel


def main():
    with open('../DATA/messy_articles.txt', encoding="utf8") as articles_file:
        content = articles_file.read()
        articles_list = ast.literal_eval(content)

    dictionary = Dictionary(articles_list)
    print(f'Id of "computer": {dictionary.token2id['computer']}')

    corpus = [dictionary.doc2bow(doc) for doc in articles_list]
    print(f'First 10 ids and frequency counts from the 5th document: {corpus[4][:10]}')

    word_counts_fifth = collections.Counter({dictionary[word_id] : frequency for word_id, frequency in corpus[4]})
    most_common_fifth = word_counts_fifth.most_common(5)
    words_fifth = [word for word, _ in most_common_fifth]
    print(f'Top 5 words in the 5th document: {words_fifth}')

    word_counts = collections.Counter()
    for current_elem in corpus:
        word_counts.update({dictionary[word_id] : frequency for word_id, frequency in current_elem})
    most_common = word_counts.most_common(5)
    print(f'Top 5 words across all documents: {most_common}')

    tfidf = TfidfModel(corpus)
    tfidf_fifth = tfidf[corpus[4]]
    print(f'First 5 term ids with their weights: {tfidf_fifth[:5]}')

    tfidf_fifth_sorted = sorted(tfidf_fifth, key=lambda x: x[1], reverse=True)[:5]
    words_sorted = [dictionary[tf_id] for tf_id, _ in tfidf_fifth_sorted]
    print(f'Top 5 words in the 5th document when using tf-idf: {words_sorted}')

if __name__ == '__main__':
    main()
