from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
import nltk
from nltk import pos_tag, ne_chunk_sents

def main():
    with open('../DATA/article_uber.txt', encoding="utf8") as article_file:
        article = article_file.read()
    tokenized_article = nltk.sent_tokenize(article)

    tokenized_last_sent = nltk.word_tokenize(tokenized_article[-1])
    tagged_sent = nltk.pos_tag(tokenized_last_sent)
    print(f'Last sentence POS: {tagged_sent}')

    tokenized_first_sent = nltk.word_tokenize(tokenized_article[0])
    tagged_first_sent = nltk.pos_tag(tokenized_first_sent)
    # print(f'First sentence with NER applied: {nltk.ne_chunk_sents(tagged_first_sent, binary=True)}') 
    # don't know what different is expected here
    print(f'First sentence with NER applied: {nltk.ne_chunk(tagged_first_sent)}')

    # it is unclear for me what approach I should use for the last one
    list_NE = []
    for sentence in tokenized_article:
        tokenized_words = nltk.word_tokenize(sentence)
        tagged_sent = nltk.pos_tag(tokenized_words)
        for tagged in tagged_sent:
            if tagged[1] == "NNP":
                list_NE.append(tagged)
    print(f'All chunks with label NE: {list_NE}')

if __name__ == '__main__':
    main()
