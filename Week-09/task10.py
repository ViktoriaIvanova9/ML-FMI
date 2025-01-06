import numpy as np
import collections
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
import nltk
from nltk import pos_tag, ne_chunk_sents
import spacy
from polyglot.text import Text

def main():
    with open('../DATA/french.txt') as article_file:
        article = article_file.read()

    ptext = Text(article)
    
    print(f'All recognized entities: {ptext.entities}')
    # print(f'Percentage of entities referring to Gabo in gabo.txt: {}')

if __name__ == '__main__':
    main()
 