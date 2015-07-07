# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from gensim import corpora, models, similarities
from nltk.corpus import stopwords
import re, logging, os, pprint
import collections
import math
import nltk


class TopicModel:
    def __init__(self, corpus):
        self.corpus = corpus
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def init_model(self):
        self.dictionary = corpora.Dictionary([self.corpus])
        docs = [self.dictionary.doc2bow(corp) for corp in self.corpus]
        self.tfidf = models.TfidfModel(docs)
        docs_tfidf = self.tfidf[docs]
        self.lsi = models.lsimodel.LsiModel(docs_tfidf, id2word=self.dictionary, num_topics = len(self.corpus))

    def save_model(self):
        index = similarities.MatrixSimilarity(self.lsi[self.corpus])
        index.save('lsi.index')
        # index = similarities.SparseMatrixSimilarity(docs_tfidf, num_features = len(self.corpus))
        # index.save('tfidf.index')

    def run(self):
        if not os.path.exists('lsi.index'):
            self.save_model()
        # index = similarities.SparseMatrixSimilarity.load('tfidf.index')
        index = similarities.MatrixSimilarity.load('lsi.index')
        en_stopwords = stopwords.words('english')

        for tweet in sys.stdin:
            query = normalize_tweet(tweet, en_stopwords)
            query_vec = self.dictionary.doc2bow(query)
            sims = index[self.lsi[query_vec]]
            result = sorted(enumerate(sims), key=lambda item: -item[1])
            pprint.pprint(result)


class ProbModel:
    def __init__(self, docs):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.dictionary = corpora.Dictionary(docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]
        self.rs = self.cal_rs(docs)
        self.idf = self.cal_idf(docs)
    
    def cal_rs(self, docs):
        tfs = []
        for corpu in self.corpus:
            tf = collections.defaultdict(int)
            for token_id, val in corpu:
                tf[token_id] = val
            tfs.append(tf)

        doc_lengths = [len(doc) for doc in docs]
        avg_dl = sum(doc_lengths)/len(docs)
        dl_norm = [dl/avg_dl for dl in doc_lengths]

        rs = [{} for i in range(len(docs))]
        for index, doc in enumerate(docs):
            K = 0.5 + 1.5 * dl_norm[index]
            for token_id in self.dictionary.values():
                fi = tfs[index][token_id]
                rs[index][token_id] = 3 * fi/(fi + K)
        return rs

    def cal_idf(self, docs):
        doc_word = []
        for corpu in self.corpus:
            for word_id, cnt in corpu:
                doc_word.append(word_id)
        idf = {}
        N = len(docs)
        for token_id in self.dictionary.keys():
            df = doc_word.count(token_id)
            idf[token_id] = math.log((N-df+0.5)/(df+0.5))
        return idf

    def run(self, docs):
        en_stopwords = stopwords.words('english')

        for tweet in sys.stdin:
            query = normalize_tweet(tweet, en_stopwords)
            topic_id, similarity = self.classify(query, docs)
            logging.info('Best match to profile: ' + str(topic_id+1))
    
    def classify(self, query, profiles):
        max_score = 0.0; match_index = -1
        query_vec = self.dictionary.doc2bow(query)
        for index, profile in enumerate(profiles):
            score = 0.0
            for word_id, tf in query_vec: 
                score += self.idf[word_id] * self.rs[index][word_id] 
            if score > max_score:
                max_score = score
                match_index = index
        return match_index, max_score


def normalize_tweet(line, stopwords): 
    re_pattern = r'''([A-Z]\.)+[A-Z]?|\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''
    text = re.sub(r'\$?\d+(\.\d+)?%?\w*|(https?:\/\/)[^ ]+|#|RT|@', '', line)
    words = [word for word in nltk.regexp_tokenize(text.lower(), re_pattern) if word not in stopwords]
    # words = [wn.morphy(word) for word in words]
    return words


def read_profiles():
    profiles = []
    with open('profiles_expanded', 'r') as fr:
        line = fr.readline()
        profile = line.split(' ')
        profiles.append(profile)
    return profiles


def test():
    profiles = read_profiles()
    tm = TopicModel(profiles)
    tm.init_model()
    tm.run()
    pm = ProbModel(profiles)
    pm.rum(profiles)
