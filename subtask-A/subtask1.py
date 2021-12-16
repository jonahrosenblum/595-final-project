from typing import OrderedDict
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
from bs4 import BeautifulSoup
# import NLP
from word2number import w2n
import pickle
import itertools
# import spacy
# from sentence_transformers import SentenceTransformer


import xmltodict, json








semeval_map = {'Potential': 0, 'Good': 1, 'Bad' : -1}


def read_xml(filename):
  semeval_list = []
  with open(filename) as f:
    doc = xmltodict.parse(f.read())
    for q in doc['root']['Question']:
      for comment in q['Comment']:
        # print((comment))
        if isinstance(comment, OrderedDict):
          if comment['@CGOLD'] not in semeval_map:
            label = -1
          else:
            label = semeval_map[comment['@CGOLD']]
          semeval_list.append({"question": q['QBody'], 
                                "comment": comment['CBody'], 
                                "label": label
                              })

    return semeval_list
data = read_xml('CQA-QL-train.xml')
comments = [d['comment'] for d in data if not d['comment'] is None]
questions = [d['question'] for d in data if not d['comment'] is None]
labels = [d['label'] for d in data if not d['comment'] is None]
# print(comments)
max_length = max([max([len(comment) for comment in comments])] + [max([len(question) for question in questions])])
print('max', max_length)
# print(train)
# data = pd.read_excel('/Users/Jason/Desktop/Computer_Science/Lectures/Michigan/NLP/Project/595-final-project/subtask-A/semeval2015_task3_trial_data.xls')

stop_words = stopwords.words('english')
web_regex = "(http(s)*://)*(www\.)*\w+(\.\w+)?\.[a-z]{2,3}/*\w*[?$%&^*@!]*(\.)?\w*"
map = {"r" : "are", "u" : "you", "ur" : "you are", "Iam" : "I am", "any1" : "anyone", "thx" : "thanks"}
lemmatizer = WordNetLemmatizer()
porterstemmer = PorterStemmer()
# label_map = {'bad' : -1, 'potential' : 0, 'good' : 1, 'repetition' : -1, 'diaglog' : -1, 'author' : -1}



q_c_df = pd.DataFrame(list(zip(questions,comments)), columns=['questions', 'comments'], index = range(len(comments)))

# stop_words_l=stopwords.words('english')
q_c_df['questions']=q_c_df.questions.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words) )
q_c_df['comments']=q_c_df.comments.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words) )


# 
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

document_embeddings = sbert_model.encode(q_c_df['questions'])

pairwise_similarities=cosine_similarity(document_embeddings)
print(pairwise_similarities)
# pairwise_differences=euclidean_distances(document_embeddings)

# most_similar(0,pairwise_similarities,'Cosine Similarity')
# most_similar(0,pairwise_differences,'Euclidean Distance')












def strip_html_tags(text):
    """remove html tags from text"""
    # if (text)
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
def preprocessing(texts):
  tokenized_texts = []
  text_tagged_list = list()
# Preprocessing 1
  for i, text in enumerate(texts):
  # tokenization
    if text is None:
      continue
    texts[i] = strip_html_tags(texts[i])
    texts[i] = remove_emoticons(texts[i])
    texts[i] = remove_URL(texts[i])
    texts[i] = decontracted(texts[i])
    texts[i] = texts[i].lower()
    text_words_list = word_tokenize(text)
    # print(texts[i])
    # Seet lower case and get rid of stopwords and punctuation
    text_words_list = [remove_emoticons(word) for word in text_words_list]
    text_words_list = [remove_accented_chars(word) for word in text_words_list]
    text_words_list = [expand_contractions(word) for word in text_words_list]
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
  return texts, tokenized_texts, text_tagged_list

comments, tokenized_comments, c_tagged_list = preprocessing(comments)
questions, tokenized_questiions, q_tagged_list = preprocessing(questions)

q = list()

c = list()





def count_tag(tagged, tag):
  return sum([1 for word in tagged if word[1] in tag])

for i in range(len(labels)):  
  # if labels[i] in label_map :
  c.append(labels[i])


for i, comment_words_list in enumerate(tokenized_comments):
  feature = list()
  feature.append(max([len(word) for word in comment_words_list]))
  feature.append(average([len(word) for word in comment_words_list]))
  feature.append(len(comment_words_list))
  feature.append(comments[i].count('?'))
  feature.append(count_tag(c_tagged_list[i], ['NN']))
  feature.append(count_tag(c_tagged_list[i], ['NN']) / len(comment_words_list))
  feature.append(count_tag(c_tagged_list[i], ['VB', 'VBD', 'VBP']) / len(comment_words_list))
  q.append(feature)
  



breakpoint = len(q) * 1 // 1000
X_train = q[:breakpoint]
Y_train = c[:breakpoint]
X_test  = q[breakpoint+1:]
Y_test  = c[breakpoint+1:]

lin_clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
print(Y_train)
lin_clf.fit(X_train, Y_train)
pred = lin_clf.predict(X_test)

print(accuracy_score(pred, Y_test))
