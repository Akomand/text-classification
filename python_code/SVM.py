# Support Vector Machine w/ Linear Kernel for Text Classification
# Aneesh Komanduri


# Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
# 
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import RegexpTokenizer


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
                     ('clf', LinearSVC()),
                     ])



text_clf.fit(X_train, Y_train)
predicted = text_clf.predict(X_test)

print(metrics.classification_report(Y_test, predicted))





