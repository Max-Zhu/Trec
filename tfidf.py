#-*-coding:utf-8-*-
import re, sys, codecs, pprint
import nltk
from gensim import corpora, models, similarities
import logging
from nltk.corpus import wordnet as wn

all_noun = [lemma.name for synset in wn.all_synsets('n') for lemma in synset.lemmas]
#all_verb = [lemma.name for synset in wn.all_synsets('v') for lemma in synset.lemmas]
corpus = all_noun
#corpus.extend(all_verb)

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
re_pattern = r'''([A-Z]\.)+[A-Z]?|\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''

def normalize_twitter(text):
    """normalization for twitter"""
    text = re.sub(r'(@|https?:\/\/)[^ ]+|#|RT', '', text)
    text = re.sub(r'(^| )[:;x]-?[\(\)dop]($| )', ' ', text)  # facemark
    text = re.sub(r'(^| )(rt[ :]+)*', ' ', text)
    text = re.sub(r'\s+&\w+', ' ', text)
    text = re.sub(r'([hj])+([aieo])+(\1+\2+){1,}', r'\1\2\1\2', text, re.IGNORECASE)  # laugh
    text = re.sub(r' +(via|live on) *$', '', text)
    return text

def preprocess(text):
	text = normalize_twitter(text)
	words = [word.lower() for word in nltk.regexp_tokenize(text, re_pattern)]
	words_stemmed = [stemmer.stem(word) for word in words if word not in stopwords]
	words_filtered = [word for word in words_stemmed if word in corpus]
	return words_filtered

def init_model(input_profiles):
	profiles = [preprocess(profile) for profile in input_profiles]
	#pprint.pprint(texts)
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	dictionary = corpora.Dictionary([corpus])
	docs = [dictionary.doc2bow(profile) for profile in profiles]
	tfidf = models.TfidfModel(docs)

	docs_tfidf = tfidf[docs]
	index = similarities.SparseMatrixSimilarity(docs_tfidf, num_features = len(corpus))
	index.save('./profiles.index')

	return dictionary, tfidf

def tfidf_test():
	profiles = loadProfile(sys.argv[1])
	dictionary, tfidf = init_model()
	index = similarities.SparseMatrixSimilarity.load('./profiles.index')

	for tweet in sys.stdin:
	    query = tokenize_stemm(tweet)
	    query_vec = dictionary.doc2bow(query)
	    sims = index[tfidf[query_vec]]
	    result = sorted(enumerate(sims), key=lambda item: -item[1])
	    pprint.pprint(result)
