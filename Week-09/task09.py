from nltk.corpus import stopwords
import spacy

def main():
    with open('../DATA/article_uber.txt', encoding="utf-8") as article_file:
        article = article_file.read()

    nlp_pipeline = spacy.load('en_core_web_sm')
    nlp_pipeline.get_pipe('ner')

    doc = nlp_pipeline(article)
    for entity in doc.ents:
        print(f'{entity.label_} {entity}')

if __name__ == '__main__':
    main()

    # Which are the extra categories that spacy uses compared to nltk in its named-entity recognition? - C. NORP, CARDINAL, MONEY, WORKOFART, LANGUAGE, EVENT
 