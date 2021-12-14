from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.lib.function_base import average, vectorize
import pandas as pd
import numpy as np
import re, scipy, nltk, unidecode, contractions, string, re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Doc2Vec
# nltk.download()pip 
from bs4 import BeautifulSoup
import NLP
from word2number import w2n
# from nlg.utils import load_spacy_model
import pickle
import itertools
import spacy

# nlp = load_spacy_model()

# soup = BeautifulSoup(html_doc, 'html.parser')

# path = 'drive/My Drive/EECS_595/'
data = pd.read_excel('/Users/Jason/Desktop/Computer_Science/Lectures/Michigan/NLP/Project/595-final-project/subtask-A/semeval2015_task3_trial_data.xls')
# data.head()


stop_words = stopwords.words('english')
web_regex = "(http(s)*://)*(www\.)*\w+(\.\w+)?\.[a-z]{2,3}/*\w*[?$%&^*@!]*(\.)?\w*"
map = {"r" : "are", "u" : "you", "ur" : "you are", "Iam" : "I am", "any1" : "anyone", "thx" : "thanks"}
comments = list(data['comment_body'])
questions = list(data['full_question'])
labels = list(data['Gold annotation'])
lemmatizer = WordNetLemmatizer()
porterstemmer = PorterStemmer()

# q_tagged_list = list()
# tokenized_comments = []
# tokenized_questions = []
# print(comments)
label_map = {'bad' : -1, 'potential' : 0, 'good' : 1, 'repetition' : -1, 'diaglog' : -1, 'author' : -1}

def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text

with open('Emoticon_Dict.p', 'rb') as fp:
    Emoticon_Dict = pickle.load(fp)

def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in Emoticon_Dict) + u')')
    return emoticon_pattern.sub(r'', text)
  
def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "",  text)

def decontracted(phrase):
  """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""

  # specific
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  phrase = re.sub(r"won\’t", "will not", phrase)
  phrase = re.sub(r"can\’t", "can not", phrase)

  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)

  phrase = re.sub(r"n\’t", " not", phrase)
  phrase = re.sub(r"\’re", " are", phrase)
  phrase = re.sub(r"\’s", " is", phrase)
  phrase = re.sub(r"\’d", " would", phrase)
  phrase = re.sub(r"\’ll", " will", phrase)
  phrase = re.sub(r"\’t", " not", phrase)
  phrase = re.sub(r"\’ve", " have", phrase)
  phrase = re.sub(r"\’m", " am", phrase)

  return phrase
# print(comments)
# print(strip_html_tags('http://www.thedailyq.org/blog/2011/02/09/larger-georgetown-library-welcomes-public/ lo'))
def preprocessing(texts):
  tokenized_texts = []
  text_tagged_list = list()
# Preprocessing 1
  for i, text in enumerate(texts):
  # tokenization
    texts[i] = strip_html_tags(texts[i])
    texts[i] = remove_emoticons(texts[i])
    texts[i] = remove_URL(texts[i])
    texts[i] = decontracted(texts[i])
    texts[i] = texts[i].lower()
    # texts[i] = text.replace('\n', '')
    # texts[i] = text.replace(':O(', '')
    print(texts[i])
    # print(texts[i])
    text_words_list = word_tokenize(text)
    # Seet lower case and get rid of stopwords and punctuation
    # text_tagged_list = [word for word in text_words_list if word not in string.punctuation and word != '?']
    text_words_list = [remove_emoticons(word) for word in text_words_list]
    # text_words_list = [decontracted(word) for word in text_words_list]  # https://www.einfochips.com/blog/nlp-text-preprocessing/
    text_words_list = [remove_accented_chars(word) for word in text_words_list]
    # text_words_list = [remove_emoticons(word) for ]
    # text_words_list = [word.lower() for word in text_words_list if word not in stop_words]
    text_words_list = [expand_contractions(word) for word in text_words_list]
    # text_words_list = [w2n.word_to_num(word.text) if word.pos_ == 'NUM' else word for word in nlp(text)]
    text_words_list = [porterstemmer.stem(word) for word in text_words_list]
    text_words_list = [lemmatizer.lemmatize(word) for word in text_words_list]
    for j, word in enumerate(text_words_list):
      # set all words to lower case
      text_words_list[j] = text_words_list[j].lower() 
      # map the informal words with formal words
      if word in map:
        text_words_list[j] = map[word]
    text_tagged = nltk.pos_tag(text_words_list)
    text_tagged_list.append(text_tagged)
      
    texts[i] = ' '.join(text_words_list)
    tokenized_texts.append(text_words_list)
  # print(texts)
  return texts, tokenized_texts, text_tagged_list