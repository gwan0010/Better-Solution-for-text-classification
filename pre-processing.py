__author__ = ''
__version__ = 'latest'
__doc__ = """ this is the first python file should be run
put the original data set under 'data' directory
"""

import pandas as pd
import nltk
import re
import string


# remove punctuation
def remove_punc(doc):
    table = str.maketrans({p: ' ' for p in string.punctuation})
    return doc.translate(table)


# loading stopwords
with open('./data/stopwords_en.txt') as f:
    stopwords = f.read().splitlines()


# preprocess (remove punctuation, digits, to lower case and token by white space)
def preprocess(text, tokenize_fun=nltk.word_tokenize):
    digit_regex = r'\d+'
    return list(filter(lambda word: word not in set(stopwords), filter(lambda word: not re.search(digit_regex, word),
       tokenize_fun(remove_punc(text.lower())))))


# get pos tag
def get_pos_tag(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.wordnet.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.wordnet.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.wordnet.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.wordnet.wordnet.ADV
    else:
        return nltk.wordnet.wordnet.NOUN


# lemmateize tokens based on pos tag
def lemmatize(doc_tokens):
    lmmer = nltk.stem.WordNetLemmatizer()
    return list(map(lambda tag: lmmer.lemmatize(tag[0], get_pos_tag(tag[1])), nltk.pos_tag(doc_tokens)))


# load the training docs
with open('./data/training_docs.txt', encoding='utf-8', mode='r') as f:
    origin_train_docs = f.read()

# load the training labels
with open('./data/training_labels_final.txt', mode='r', encoding='utf-8') as f:
    origin_train_labels = f.read()

# sort training labels according to doc id and grep the labels as list
train_labels = origin_train_labels.split('\n')
train_labels.pop()
train_labels.sort(key=lambda each: int(each.split(' ')[0].split('_')[2]), reverse=False)
labels = list(map(lambda each: each.split(' ')[1], train_labels))

# sort training docs according to doc id and grep the text content as list
doc_regex = r'ID [\w\W]+?\nEOD'
doc_id_regex = r'ID ([\w\W]+?)\nTEXT'
text_regex = r'TEXT ([\w\W]+?)\nEOD'
train_texts = re.findall(doc_regex, origin_train_docs)
train_texts.sort(key=lambda each: int(re.search(doc_id_regex, each).group(1).split('_')[2]), reverse=False)
texts = list(map(lambda each: re.search(text_regex, each).group(1).replace('\n', ' ').strip(), train_texts))

# create training docs data frame
training_docs = pd.DataFrame()
training_docs['Class'] = labels
training_docs['Text'] = texts
# remove from the memory to save space
del labels
del texts
del origin_train_docs
del origin_train_labels
# training_docs = training_docs.iloc[:10, ] (for develop only)

# apply pre-processing process on training set
training_docs['Text'] = training_docs['Text'].apply(lambda x: lemmatize(preprocess(x)))
# remove empty docs
training_docs = training_docs[training_docs['Text'].apply(lambda x: ' '.join(x)).str.strip() != ''].reset_index(drop=True)
# concatenate the tokens as a string for the following operation
training_docs['Text'] = training_docs['Text'].apply(lambda x: ' '.join(x))
# store the pre-processed docs
training_docs.to_csv('./data/preprocessed_training_docs.csv', index=False, encoding='utf-8')
# remove from the memory to save space
del training_docs

# for testing docs
with open('./data/testing_docs.txt', mode='r', encoding='utf-8') as f:
    origin_test_docs = f.read()

test_texts = re.findall(doc_regex, origin_test_docs)
texts_te = list(map(lambda each: re.search(text_regex, each).group(1).replace('\n', ' ').strip(), test_texts))
testing_docs = pd.DataFrame()
testing_docs['Text'] = texts_te
testing_docs['Text'] = testing_docs['Text'].apply(lambda x: lemmatize(preprocess(x)))
testing_docs['Text'] = testing_docs['Text'].apply(lambda x: ' '.join(x))
del texts_te
testing_docs.to_csv('./data/preprocessed_testing_docs.csv', index=False, encoding='utf-8')
del testing_docs
