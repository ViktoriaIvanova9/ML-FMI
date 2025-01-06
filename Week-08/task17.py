import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

def main():    
    document = ['cats say meow', 'dogs say woof', 'dogs chase cats']

    tf_idf_vec = TfidfVectorizer().fit(document)
    transformed_tf_idf = tf_idf_vec.transform(document)

    print(transformed_tf_idf.toarray())
    print(tf_idf_vec.get_feature_names_out())


if __name__ == '__main__':
    main()