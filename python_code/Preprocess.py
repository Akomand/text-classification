# Preprocessing for Text Classification
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



tokenizer = ToktokTokenizer()

# Stop words
stopword_list = stopwords.words('english')


def load_data(dataset_str):
    # Dataset of 400,000 samples
    if dataset_str == 'news':
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')
        
        return newsgroups_train.data, newsgroups_train.target, newsgroups_test.data, newsgroups_test.target
    
    if dataset_str == 'amazon':
        with open('../data/test.ft.txt', 'r') as f:
            labels, reviews = [], []
            for line in f:
                content = line.split()
                labels.append(content[0])
                reviews.append(' '.join(content[1:]))
        f.close()
        
        data = pd.DataFrame()
        data['review'] = reviews
        data['sentiment'] = labels 
        
    # Dataset of 50,000 samples
    if dataset_str == 'imdb':
        data = pd.read_csv('../data/IMDB_Dataset.csv')
    
    encoder = LabelEncoder()
    data['sentiment'] = encoder.fit_transform(data['sentiment'])
    
    return data


# Preprocessing Functions
# functions for removing html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# removing square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# removing noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# remove special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

# Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# Removing stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text



def preprocessing(data):
    data = data.apply(denoise_text)
    data = data.apply(remove_special_characters)
    data = data.apply(simple_stemmer)
    data = data.apply(remove_stopwords)
    
    return data


dataset = 'news'


trainData, trainLabel, testData, testLabel = load_data(dataset)

train = pd.DataFrame(data=trainData, columns=["review"])


train['sentiment'] = trainLabel



train = preprocessing(train)


test = pd.DataFrame(data=testData, columns=["review"])


test['sentiment'] = testLabel


test = preprocessing(test)


train.to_pickle(f'../data/{dataset}_train_preprocessed')


test.to_pickle(f'../data/{dataset}_test_preprocessed')




