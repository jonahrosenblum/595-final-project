from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import re
import scipy
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.metrics import accuracy_score
# nltk.download()pip 

# path = 'drive/My Drive/EECS_595/'
data = pd.read_excel('/Users/Jason/Desktop/Computer_Science/Lectures/Michigan/NLP/Project/595-final-project/subtask-A/semeval2015_task3_trial_data.xls')
data.head()

import string
import re
stop_words = stopwords.words('english')
web_regex = "(http(s)*://)*(www\.)*\w+(\.\w+)?\.[a-z]{2,3}/*\w*[?$%&^*@!]*(\.)?\w*"
map = {"r" : "are", "u" : "you", "ur" : "you are", "Iam" : "I am", "any1" : "anyone", "thx" : "thanks"}
comments = list(data['comment_body'])
labels = list(data['Gold annotation'])
lemmatizer = WordNetLemmatizer()
porterstemmer = PorterStemmer()
tagged_list = list()
tokenized_comments = []
# print(comments)
label_map = {'bad' : -1, 'potential' : 0, 'good' : 1, 'dialog' : 0, 'ad' : 0, 'author' : 0, 'repetition': -1}

for i, comment in enumerate(comments):
  comments[i] = comment.replace('\n', '')
  comments[i] = comment.replace(':O(', '')
  comments[i] = re.sub(web_regex, "", comments[i])
  # print(comments[i])

for i, comment in enumerate(comments):
# tokenization
  comment_words_list = word_tokenize(comment)
  # Seet lower case and get rid of stopwords and punctuation  
  comment_words_list = [word.lower() for word in comment_words_list if word not in stop_words]
  comment_words_list = [porterstemmer.stem(word) for word in comment_words_list]
  comment_words_list = [lemmatizer.lemmatize(word) for word in comment_words_list]
  for j, word in enumerate(comment_words_list):
    # set all words to lower case
    comment_words_list[j] = comment_words_list[j].lower() 
    # map the informal words with formal words
    if word in map:
      comment_words_list[j] = map[word]
  tagged = nltk.pos_tag(comment_words_list)
  tagged_list.append(tagged)
    
  comments[i] = ' '.join(comment_words_list)
  tokenized_comments.append(comment_words_list)
  
X = list()

Y = list()


def count_tag(tagged, tag):
      return sum([1 for word in tagged if word[1] in tag])
for i in range(len(comments)):  
  if pd.isna(labels[i]) :
        labels[i] = 'potential'
  # try:
  
  Y.append(label_map[labels[i]])
  # except:
  #   print(comments[i], i)


# print (pd.isna(labels[5]))

# print(Y)
for i, comment_words_list in enumerate(tokenized_comments):
      feature = list()
      # for j, word in enumerate(comment_words_list):
      feature.append(max([len(word) for word in comment_words_list]))
      feature.append(average([len(word) for word in comment_words_list]))
      feature.append(len(comment_words_list))
      feature.append(comments[i].count('?'))
      feature.append(count_tag(tagged_list[i], ['NN']))
      feature.append(count_tag(tagged_list[i], ['NN']) / len(comment_words_list))
      feature.append(count_tag(tagged_list[i], ['VB', 'VBD', 'VBP']) / len(comment_words_list))
      X.append(feature)
      # Y.append(label_map[labels[i]])

      
print(len(Y))
X_train = X[:300]
Y_train = Y[:300]
X_test  = X[301:]
Y_test  = Y[301:]
# print(X)
lin_clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
lin_clf.fit(X_train, Y_train)
pred = lin_clf.predict(X_test)
# print(len())
# print(len(pred))
print(accuracy_score(pred, Y_test))
# print(len(X_train))
# print(len(X_train[0]))
# print(len(Y_train))

# print(tagged_list)