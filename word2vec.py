#-*-coding:utf-8-*-
'''
Created on May 24, 2015

@author Max
'''

from gensim import corpora, models
from nltk.corpus import wordnet as wn
import nltk
import threading
import time, datetime
import sys, os
import logging

class MyThread(threading.Thread):
    """docstring for classify"""
    def __init__(self, threadName, event):
        super(MyThread, self).__init__()
        self.name = threadName
        self.threadEvent = event
        self.threshold = 0.5 # 相似性阈值，动态调整

    def run(self):
        print '%s is waiting...\n' % self.name
        self.threadEvent.wait()
        print '%s is running...\n' % self.name
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        process()

    def loadWord2VecModel(modelName):
        print 'loading model %s ...\n' % modelName
        time.sleep(5)
        self.w2vModel = gensim.models.Word2Vec.load_word2vec_format(modelName, binary=False)
        print 'model %s loaded\n' % modelName

    def loadTfidfModel(corpus, profiles):
        self.dictionary = corpora.Dictionary([corpus])
        profiles_bow = [dictionary.doc2bow(profile) for profile in profiles]
        tfidf = models.TfidfModel(profiles_bow)
        return tfidf

    def process():
        profiles = loadProfiles("")
        corpus = loadCorpus()
        tfidf = loadTfidfModel(corpus, profiles)
        profiles_keywords = extrKeywords(profiles, tfidf, 20)
        for tweet_text in fileinput.input():
            tweet = normalizeTweet(tweet_text, corpus)
            if tweet != "":
                tweet_tfidf = tfidf[self.dictionary.doc2bow(tweet)] #对每条推文进行tfidf化处理
                topicID = classify(tweet_tfidf, profiles_keywords)
                if topicID != -1: print topicID

    def classify(tweet, profiles):
        bestMatch = -1; maxSimilarity = 0.0
        for index, profile in enumerate(profiles):
            similarity = calcSimilarity(tweet, profile)
            if similarity > maxSimilarity:
                maxSimilarity = similarity
                bestMatch = index
        if maxSimilarity > self.threshold:
            return bestMatch
        return -1

    def calcSimilarity(tweet, profile_keywords):
        ''''利用Word2Vec模型计算每条推文的词和profile的每个关键词相似度，以推文词的tfidf值作为权值'''
        similarity = 0.0
        for token_id, weight in tweet:
            token = self.dictionary.id2token[token_id]
            for profile_keyword in profile_keywords:
                similarity += self.w2vModel.similarity(token, profile_keyword)*weight
        return similarity

def preprocess():
    signal = threading.Event()
    classifier = MyThread("classify_thread", signal)
    classifier.loadWord2VecModel("wiki.en.text.vector")
    classifier.start()

    isReady = detectTweetStream(2015, 5, 24, 12, 54, 0, 0) # 开始时间
    if isReady:
        signal.set()

def detectTweetStream(year, month, d, h, m, s, ms):
    start = datetime.datetime(year, month, d, h, m, s, ms)
    delta = (start - datetime.datetime.now()).seconds
    print 'waiting %d secondes\n' %delta
    time.sleep(delta)
    print 'tweet stream is ready\n'
    isReady = True
    return isReady

def loadProfiles(profileFilePath):
    '''每个profile文件代表一个user_profile'''
    profiles = []
    profileFiles = os.listdir(profileFilePath)
    for file in profileFiles:
        fr = open(os.path.join(profileFilePath, file))
        content = fr.readlines()
        profiles.append(content)
        fr.close()
    return profiles

def extrKeywords(profiles, tfidf, topK):
    ''''从profile中抽取出代表话题的关键字（依据tfidf值）'''
    profiles_tfidf = tfidf[profles]
    profiles_keywords = []
    for profile_tfidf in profiles_tfidf:
        profileSorted = sorted(profile_tfidf, key = lambda item: -item[1])
        if len(profileSorted) > topK:
            profiles_keywords.append(profileSorted[:topK])
        else:
            profiles_keywords.append(profileSorted)
    return profiles_keywords

def loadCorpus():
    all_noun = [lemma.name for synset in wn.all_synsets('n') for lemma in synset.lemmas]
    all_verb = [lemma.name for synset in wn.all_synsets('v') for lemma in synset.lemmas]
    corpus = all_noun
    corpus.extend(all_verb)
    return corpus

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
re_pattern = r'''([A-Z]\.)+[A-Z]?|\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''

def normalizeTweet(line, corpus):
    text = re.sub(r'(@|https?:\/\/)[^ ]+|#|RT', '', line)
    words = [word.lower() for word in nltk.regexp_tokenize(text, re_pattern)]
    words_stemmed = [stemmer.stem(word) for word in words if word not in stopwords]
    words_filtered = [word for word in words_stemmed if word in corpus]
    return words_filtered
