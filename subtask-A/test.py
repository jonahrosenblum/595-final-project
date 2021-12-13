from bs4 import BeautifulSoup
import re
import pickle
def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "",  text)

with open('Emoticon_Dict.p', 'rb') as fp:
    Emoticon_Dict = pickle.load(fp)
def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in Emoticon_Dict) + u')')
    return emoticon_pattern.sub(r'', text)

texts = ['LOL Nishant - childs play eh. guess that explains your profile saying u r under 18...:0(', 'Anyway, the link is http://www.canadainternational.gc.ca/kuwait-koweit/index.aspx']
for i, text in enumerate(texts):
    texts[i] = remove_emoticons(texts[i])
    texts[i] = remove_URL(texts[i])
    
    print(texts[i])