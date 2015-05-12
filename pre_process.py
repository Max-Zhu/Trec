import re
import nltk

stopwords = nltk.corpus.stopwords.words('english')
porter_stemmer = nltk.stem.porter.PorterStemmer()
re_pattern = r'''([A-Z]\.)+[A-Z]?|\s+[^&]\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''

def normalize_twitter(text):
    """normalization for twitter"""
    text = re.sub(r'(RT|@|https?:\/\/)[^ ]+|#', '', text)
    text = re.sub(r'(^| )[:;x]-?[\(\)dop]($| )', ' ', text)  # facemark
    text = re.sub(r'(^| )(rt[ :]+)*', ' ', text)
    text = re.sub(r'([hj])+([aieo])+(\1+\2+){1,}', r'\1\2\1\2', text, re.IGNORECASE)  # laugh
    text = re.sub(r' +(via|live on) *$', '', text)
    return text

def tokenize_stemm(text):
	text = normalize_twitter(text)
	words = nltk.regexp_tokenize(text, re_pattern)
	stem_words = [porter_stemmer.stem_word(word) for word in words if word.lower() not in stopwords]
	return stem_words
