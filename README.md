# Better-Solution-for-text-classification

--------------------
Env requirements
--------------------
python version 3.6
R version >= 3.4

packages:
• LibLineaR (R library)
• Matrix (R library)
• SparseM (R library)

• re (Python package)
• string (Python package)
• nltk (Python package)
• scipy (Python package)
• scikit-learn (Python package)
• pandas (Python package)

--------------------
Directory Path
--------------------
root
- data (directory)
  under data directory
  - training_docs.txt
  - training_labels_final.txt
  - testing_docs.txt
  - stopwords_en.txt 
  - preprocessed_training_docs.csv (generated file)
  - preprocessed_testing_docs.csv (generated file)
  - testing_labels_pred.txt (* generated file/predict results)

- feature_vectors (directory needs to be created)
  under feature_vectors directory
  - combine_vocabs.txt (generated file)
  - training_combine_dtm.mtx (generated file)
  - testing_combine_dtm.mtx (generated file)
  - training_labels.csv

- pre-processing.py
- vocabs_generation.py
- feature_extraction.py
- best_model.R

----------------------
Data Files and Preparation
----------------------
- put all the original data files under 'data' directory
- create 'feature_vectors' in advance to store some generated files

----------------------
Codes running Operations
----------------------
assume already in the root directory (same level as this file)
using following order
$ python pre-processing.py (linux/unix like command line)
$ python vocabs_generateion.py (linux/unix like command line)
$ python feature_extraction.py (linux/unix like command line)
$ Rscript best_model.R (linux/unix like command line)
after running above codes, the final predict results will be stored in testing_labels_pred.txt under data directory

--------------------
Algorithmic Analysis
--------------------

- linear SVM classifier is used for the better effective and efficient computing
- In the process of text preprocessing, the lemmatisation is used combining with pos tag (part of speech)
- TF-IDF is used to for the adding weights to each genearted feature(word)
- sparse matrix is used to improve the speed of I/O and model fitting


