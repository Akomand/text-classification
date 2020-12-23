# Multinomial Naive Bayes Classifier for Text Classification
# Aneesh Komanduri


# Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Datasets
from sklearn.datasets import fetch_20newsgroups
import utils



news = True
dataset = 'amazon'
data = pd.read_pickle(f'../data/{dataset}_preprocessed')



if news:
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    Y_train = newsgroups_train.target
    Y_test = newsgroups_test.target

    X_train = pd.read_pickle(f'../data/news_train_preprocessed')
    X_test = pd.read_pickle(f'../data/news_test_preprocessed')
    X_train = X_train['review']
    X_test = X_test['review']
else:
    X_train, X_test, Y_train, Y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=5)



text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf.fit(X_train, Y_train)
predicted = text_clf.predict(X_test)

print(metrics.classification_report(Y_test, predicted))

