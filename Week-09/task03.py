import re
import nltk
import collections
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer

def main():
    tweets = ['This is the best #nlp exercise ive found online! #python', '#NLP is super fun! <3 #learning', 'Thanks @datacamp :) #nlp #python']    
    
    tokenizer = RegexpTokenizer(r'[#@]\w+')
    tweet_token = TweetTokenizer()

    tokenized_tweets = []
    for tweet in tweets:
        current_tokenized = tweet_token.tokenize(tweet)
        tokenized_tweets.append(current_tokenized)

    print(f'All hashtags in first tweet: {tokenizer.tokenize(tweets[0])}')
    print(f'All mentions and hashtags in last tweet: {tokenizer.tokenize(tweets[-1])}')
    print(f'All tokens: {tokenized_tweets}')

if __name__ == '__main__':
    main()