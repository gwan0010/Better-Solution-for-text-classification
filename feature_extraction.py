__author__ = ''
__version__ = 'latest'
__doc__ = """ this is the third file to be run
create 'feature_vectors' directory to store generated feature files
"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.io


train = pd.read_csv('./data/preprocessed_training_docs.csv', )

test = pd.read_csv('./data/preprocessed_testing_docs.csv')
test['Text'].fillna('', inplace=True)


def parse_vocabs(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        vocabs_content = f.read()

    vocabs_dict = dict(map(lambda each:
                           (each.split(':')[0], int(each.split(':')[1])), vocabs_content.split('\n')))
    return vocabs_dict


#unigrams_dict = parse_vocabs('./feature_vectors/unigram_vocabs.txt')
combine_vocabs = parse_vocabs('./feature_vectors/combine_vocabs.txt')

# unigrams
# uni_tfidf_vec = TfidfVectorizer(vocabulary=unigrams_dict, ngram_range=(1, 1))

# uni_tr_X = uni_tfidf_vec.fit_transform(train['Text'].values)
# uni_te_X = uni_tfidf_vec.transform(test['Text'].values)
# scipy.io.mmwrite('./feature_vectors/training_unigram_dtm.mtx', uni_tr_X)
# scipy.io.mmwrite('./feature_vectors/testing_unigram_dtm.mtx', uni_te_X)

# combined (uni + bi + tri)
combine_tfidf_vec = TfidfVectorizer(vocabulary=combine_vocabs, ngram_range=(1, 3))

combine_tr_X = combine_tfidf_vec.fit_transform(train['Text'].values)
combine_te_X = combine_tfidf_vec.transform(test['Text'].values)
scipy.io.mmwrite('./feature_vectors/training_combine_dtm.mtx', combine_tr_X)
scipy.io.mmwrite('./feature_vectors/testing_combine_dtm.mtx', combine_te_X)

#pd.DataFrame(train['Class']).to_csv('./feature_vectors/training_labels.csv', index=False)

# del combine_tr_tfidf_dtm
# del combine_te_tfidf_dtm
# del combine_tfidf_vec
# del combine_tr_X
# del combine_te_X
