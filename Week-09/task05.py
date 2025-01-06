import re
import nltk
import collections
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def main():
    with open("../DATA/article.txt") as wiki_file:
        wiki_text = wiki_file.read()
    
    counter = collections.Counter(tokenize.word_tokenize(wiki_text))

    print(f'Top 10 most common tokens: {counter.most_common(10)}')
    # By looking at the most common tokens, what topic does this article cover? - it is not clear

    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+') # here it reads n from \n as a separate word 
    tokenized_text = tokenizer.tokenize(wiki_text)
    
    filtered_text = []
    for word in tokenized_text:
        if word not in stop_words:
            filtered_text.append(word.lower())

    preprocessed_counter = collections.Counter(filtered_text)
    print(f'Top 10 most common tokens after preprocessing: {preprocessed_counter.most_common(10)}')

if __name__ == '__main__':
    main()
