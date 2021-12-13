from bs4 import BeautifulSoup
def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

text = 'http://abc.com k i know'
print(strip_html_tags(text))