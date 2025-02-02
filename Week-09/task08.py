import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import spacy
import nltk
from nltk import pos_tag, ne_chunk_sents

def main():
    with open('../DATA/news_articles.txt', encoding="utf8") as article_file:
        article = article_file.read()

    nlp_pipeline = spacy.load('en_core_web_sm')
    nlp_pipeline.get_pipe('ner')

    doc = nlp_pipeline(article)
    entities = [doc.ents[i].label_  for i, _ in enumerate(doc.ents)]

    counts = Counter(entities)
    labels = list(counts.keys())
    number_of_elements = list(counts.values())

    plt.pie(number_of_elements, labels=labels)
    plt.title("Distribution of NER categories")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
