import numpy as np
import collections
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
import nltk
from nltk import pos_tag, ne_chunk_sents

def main():
    with open('../DATA/news_articles.txt') as article_file:
        article = article_file.read()


    tokenized_sent = nltk.word_tokenize(article)
    print(tokenized_sent)
    # tagged_sent = nltk.pos_tag(tokenized_sent)
    # chucks_list = nltk.ne_chunk(tagged_sent)
    # print(chucks_list)

    # for elem in 
    # fig, ax = plt.subplots()
    # ax.pie(counts, labels=labels)
    # plt.title("Distribution of NER categories")
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()
