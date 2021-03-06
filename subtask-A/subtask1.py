from typing import OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.lib.function_base import average, vectorize
import pandas as pd
import numpy as np
import re, nltk, unidecode, contractions, string, re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import pickle
from sentence_transformers import SentenceTransformer


import xmltodict


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

semeval_map = {'Potential': 0, 'Good': 1, 'Bad' : -1}





def read_xml(filename):
  semeval_list = []
  with open(filename) as f:
    doc = xmltodict.parse(f.read())
    for q in doc['root']['Question']:
      for comment in q['Comment']:
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
test = read_xml('CQA-QL-devel.xml')
comments = [d['comment'] for d in data if not d['comment'] is None]
questions = [d['question'] for d in data if not d['comment'] is None]
Y_train = [d['label'] for d in data if not d['comment'] is None]

test_comments = [d['comment'] for d in test if not d['comment'] is None]
test_questions = [d['question'] for d in test if not d['comment'] is None]
Y_test = [d['label'] for d in test if not d['comment'] is None]



stop_words = stopwords.words('english')
web_regex = "(http(s)*://)*(www\.)*\w+(\.\w+)?\.[a-z]{2,3}/*\w*[?$%&^*@!]*(\.)?\w*"
map = {"r" : "are", "u" : "you", "ur" : "you are", "Iam" : "I am", "any1" : "anyone", "thx" : "thanks"}
lemmatizer = WordNetLemmatizer()
porterstemmer = PorterStemmer()


# Embedding reference: https://github.com/varun21290/medium/blob/master/Document%20Similarities/Document_Similarities.ipynb

q_c_df = pd.DataFrame(list(zip(questions,comments)), columns=['questions', 'comments'], index = range(len(comments)))

q_c_df['questions']=q_c_df.questions.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words) )
q_c_df['comments']=q_c_df.comments.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words) )

test_q_c_df = pd.DataFrame(list(zip(questions,comments)), columns=['questions', 'comments'], index = range(len(comments)))
test_q_c_df['questions']=test_q_c_df.questions.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words) )
test_q_c_df['comments']=test_q_c_df.comments.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words) )

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

questions_embeddings = sbert_model.encode(q_c_df['questions'])
comments_embeddings = sbert_model.encode(q_c_df['comments'])


test_questions_embeddings = sbert_model.encode(test_q_c_df['questions'])
test_comments_embeddings = sbert_model.encode(test_q_c_df['comments'])









# Preprocessing reference: https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79
def strip_html_tags(text):
    """remove html tags from text"""
    
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_accented_chars(text):
    """remove accented characters from text, e.g. caf??"""
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

# Decontract reference: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490
def decontracted(phrase):
  """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""

  # specific
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  phrase = re.sub(r"won\???t", "will not", phrase)
  phrase = re.sub(r"can\???t", "can not", phrase)

  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)

  phrase = re.sub(r"n\???t", " not", phrase)
  phrase = re.sub(r"\???re", " are", phrase)
  phrase = re.sub(r"\???s", " is", phrase)
  phrase = re.sub(r"\???d", " would", phrase)
  phrase = re.sub(r"\???ll", " will", phrase)
  phrase = re.sub(r"\???t", " not", phrase)
  phrase = re.sub(r"\???ve", " have", phrase)
  phrase = re.sub(r"\???m", " am", phrase)

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

test_comments, test_tokenized_comments, test_c_tagged_list = preprocessing(test_comments)
test_questions, test_tokenized_questiions, test_q_tagged_list = preprocessing(test_questions)

X_train = list()


X_test = list()



def count_tag(tagged, tag):
  return sum([1 for word in tagged if word[1] in tag])





for i, comment_words_list in enumerate(tokenized_comments):
  feature = list()
  pairwise_similarities=cosine_similarity([questions_embeddings[i], comments_embeddings[i]])
  feature.append(pairwise_similarities[0][1])
  feature.append(max([len(word) for word in comment_words_list]))
  feature.append(average([len(word) for word in comment_words_list]))
  feature.append(len(comment_words_list))
  feature.append(comments[i].count('?'))
  feature.append(count_tag(c_tagged_list[i], ['NN']))
  feature.append(count_tag(c_tagged_list[i], ['NN']) / len(comment_words_list))
  feature.append(count_tag(c_tagged_list[i], ['VB', 'VBD', 'VBP']) / len(comment_words_list))
  X_train.append(feature)
  

for i, test_comment_words_list in enumerate(test_tokenized_comments):
  test_feature = list()
  test_pairwise_similarities=cosine_similarity([test_questions_embeddings[i], test_comments_embeddings[i]])
  test_feature.append(test_pairwise_similarities[0][1])
  test_feature.append(max([len(word) for word in test_comment_words_list]))
  test_feature.append(average([len(word) for word in test_comment_words_list]))
  test_feature.append(len(test_comment_words_list))
  test_feature.append(test_comments[i].count('?'))
  test_feature.append(count_tag(test_c_tagged_list[i], ['NN']))
  test_feature.append(count_tag(test_c_tagged_list[i], ['NN']) / len(test_comment_words_list))
  test_feature.append(count_tag(test_c_tagged_list[i], ['VB', 'VBD', 'VBP']) / len(test_comment_words_list))
  X_test.append(test_feature)





lin_clf = svm.SVC(decision_function_shape='ovo', kernel='linear')

lin_clf.fit(X_train, Y_train)
pred = lin_clf.predict(X_test)


print('Accuracy is', accuracy_score(pred, Y_test))
print('F1 score is', f1_score(Y_test, pred, average = 'macro'))