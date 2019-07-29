__author__ = ''
__version__ = 'latest'
__doc__ = """ this is the second file to be run
create 'feature_vectors' directory in advance to store generated vocabs
"""


import pandas as pd
from itertools import chain
from collections import Counter
import nltk



# unigrams without lemmatization
def get_unigram_vocabs(training_docs_df, threshold=0.05):

    vocabs = set()
    
    for current_label in training_docs_df['Class'].unique():
        mask = training_docs_df['Class'] == current_label
        current_df = pd.DataFrame(training_docs_df.loc[mask, ])
        current_tokens_counter = Counter(chain.from_iterable(current_df['Text'].apply(set).values))
        current_vocab = \
            list(filter(lambda token:
                        len(token) >= 3 and token.isalpha() and
                        current_tokens_counter[token] >= threshold*current_df.shape[0], current_tokens_counter))
        vocabs = vocabs.union(current_vocab)
    vocabs = sorted(vocabs)
    return vocabs


def get_bigram_vocabs(training_docs_df, threshold=0.05):
    vocabs = set()

    for current_label in training_docs_df['Class'].unique():
        mask = training_docs_df['Class'] == current_label
        current_df = pd.DataFrame(training_docs_df.loc[mask,])

        current_df['Text'] = current_df['Text'].apply(lambda x: list(nltk.bigrams(x)))
        current_tokens_counter = Counter(chain.from_iterable(current_df['Text'].apply(set).values))
        current_vocab = \
            list(filter(lambda token:
                        token[0].isalpha() and token[1].isalpha() and 
                        current_tokens_counter[token] >= threshold*current_df.shape[0], current_tokens_counter))
        vocabs = vocabs.union(current_vocab)

    vocabs = sorted(map(lambda tup: "{} {}".format(*tup), vocabs))
    return vocabs


def get_trigram_vocabs(training_docs_df, threshold=0.05):
    vocabs = set()
    
    for current_label in training_docs_df['Class'].unique():
        mask = training_docs_df['Class'] == current_label
        current_df = pd.DataFrame(training_docs_df.loc[mask,])
        
        current_df['Text'] = current_df['Text'].apply(lambda x: list(nltk.trigrams(x)))
        current_tokens_counter = Counter(chain.from_iterable(current_df['Text'].apply(set).values))
        current_vocab = \
            list(filter(lambda token:
                        token[0].isalpha() and token[1].isalpha() and token[2].isalpha() and
                        current_tokens_counter[token] >= threshold*current_df.shape[0], current_tokens_counter))
        vocabs = vocabs.union(current_vocab)

    vocabs = sorted(map(lambda tup: "{} {} {}".format(*tup), vocabs))
    return vocabs


training_docs = pd.read_csv('./data/preprocessed_training_docs.csv', encoding='utf-8')
training_docs['Text'] = training_docs['Text'].apply(lambda x: x.split(' '))

# unigrams
unigram_vocabs = get_unigram_vocabs(training_docs, 0.001)
# unigram_vocabs_content = "\n".join(map(lambda x: "{1}:{0}".format(x[0], x[1]), enumerate(unigram_vocabs)))
# with open('./feature_vectors/unigram_vocabs.txt', mode='w', encoding='utf-8') as f:
    # f.write(unigram_vocabs_content)

# pd.DataFrame({"Class": labels}).to_csv('./data/model_training_labels.csv', index=False)

# bigrams
bigram_vocabs = get_bigram_vocabs(training_docs, 0.001)

# trigrams
trigram_vocabs = get_trigram_vocabs(training_docs, 0.001)

combine_vocabs = sorted(set(chain.from_iterable([unigram_vocabs, bigram_vocabs, trigram_vocabs])))
combine_vocabs_content = "\n".join(map(lambda x: "{1}:{0}".format(x[0], x[1]), enumerate(combine_vocabs)))
with open('./feature_vectors/combine_vocabs.txt', mode='w', encoding='utf-8') as f:
    f.write(combine_vocabs_content)
