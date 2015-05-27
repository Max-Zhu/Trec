#-*-coding:utf-8-*-
'''
Created on May 24, 2015
@author Max
'''

from gensim import corpora, models
from nltk.corpus import wordnet as wn
import gensim
import nltk
import threading
import time, datetime
import sys, os, codecs, json, re, fileinput
import logging

reload(sys)
sys.setdefaultencoding('utf-8')

class MyThread(threading.Thread):
    """docstring for classify"""
    def __init__(self, threadName, event):
        super(MyThread, self).__init__()
        self.name = threadName
        self.threadEvent = event
        self.threshold = 0.5 # 相似性阈值，动态调整
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def run(self):
        logging.info('%s is waiting...\n' % self.name)
        self.threadEvent.wait()
        logging.info('%s is running...\n' % self.name)
        self.process()

    def loadWord2VecModel(self, modelName):
        logging.info('loading model %s ...\n' % modelName)
        #time.sleep(5)
        self.w2vModel = gensim.models.Word2Vec.load_word2vec_format('wiki.en.text.vector', binary=False)
        #self.w2vModel = "test"
        logging.info('model %s loaded\n' % modelName)

    def loadTfidfModel(self, corpus, profiles):
        self.dictionary = corpora.Dictionary([corpus])
        profiles_bow = [self.dictionary.doc2bow(profile) for profile in profiles]
        tfidf = models.TfidfModel(profiles_bow)
        return tfidf, profiles_bow

    def process(self):
        corpus = loadCorpus()
        profiles = loadProfiles("test_profile", corpus)
        tfidf, profiles_bow = self.loadTfidfModel(corpus, profiles)
        profiles_keywords = extrKeywords(profiles_bow, tfidf, 20)
        for tweet_text in sys.stdin:
            logging.info('input tweet: ' + tweet_text)
            tweet = normalizeTweet(tweet_text, corpus)
            logging.info('normalized tweet: ' + ' '.join(tweet))
            if tweet != "":
                tweet_tfidf = tfidf[self.dictionary.doc2bow(tweet)] #对每条推文进行tfidf化处理
                topicID = self.classify(tweet_tfidf, profiles_keywords)
                if topicID != -1: logging.info('Best Match to profile: '+ str(opicID))

    def classify(self, tweet, profiles):
        #logging.info('classify tweet_tfidf: ' + ' '.join(tweet))
        print 'classify tweet_tfidf: ', tweet
        bestMatch = -1; maxSimilarity = 0.0
        for index, profile in enumerate(profiles):
            similarity = self.calcSimilarity(tweet, profile)
            print 'similarity: ', similarity
            if similarity > maxSimilarity:
                maxSimilarity = similarity
                bestMatch = index
        if maxSimilarity > self.threshold:
            return bestMatch
        return -1

    def calcSimilarity(self, tweet, profile_keywords):
	'''利用Word2Vec模型计算每条推文的词和profile的每个关键词相似度，以推文词的tfidf值作为权值'''
        similarity = 0.0
        for token_id, weight in tweet:
            token = self.dictionary[token_id]
            for key, value in profile_keywords:
                profile_keyword = self.dictionary[key]
                try:
                    similarity += self.w2vModel.similarity(token, profile_keyword)*weight
                except Exception, e:
                    print e
                    pass
        return similarity

def preprocess():
    signal = threading.Event()
    classifier = MyThread("classify_thread", signal)
    classifier.loadWord2VecModel(sys.argv[1])
    classifier.start()
    #isReady = detectTweetStream(2015, 5, 24, 12, 54, 0, 0) # 开始时间
    isReady = True
    if isReady:
        signal.set()

def detectTweetStream(year, month, d, h, m, s, ms):
    start = datetime.datetime(year, month, d, h, m, s, ms)
    delta = (start - datetime.datetime.now()).seconds
    logging.info('waiting secondes: \n'+delta)
    time.sleep(delta)
    logging.info('tweet stream is ready\n')
    isReady = True
    return isReady

def loadProfiles(profileFilePath, corpus):
    profiles = []
    profileFiles = os.listdir(profileFilePath)
    for file in profileFiles:
        fr = codecs.open(os.path.join(profileFilePath, file), 'rb', 'utf-8')
        content = fr.read()
        tweets = extractTweets(content)
        #logging.info('extract profile: '+ tweets)
        tweets_normalized = normalizeTweet(tweets, corpus)
        profiles.append(tweets_normalized)
        logging.info('load profile: '+ ' '.join(tweets_normalized))
        fr.close()
    return profiles

def extrKeywords(profiles, tfidf, topK):
    '''从profile中抽取出代表话题的关键字（依据tfidf值）'''
    #print 'profiles bow: ', profiles
    profiles_tfidf = tfidf[profiles]
    profiles_keywords = []
    for profile_tfidf in profiles_tfidf:
        #logging.info('profile_tfidf: '+' '.join(profile_tfidf))
        #print 'profile_tfidf: ', profile_tfidf
        profileSorted = sorted(profile_tfidf, key = lambda item: -item[1])
        #logging.info('sorted profile_tfidf: '+' '.join(profileSorted))
        #print 'sorted profile_tfidf: ', profileSorted
        if len(profileSorted) > topK:
            profiles_keywords.append(profileSorted[:topK])
        else:
            profiles_keywords.append(profileSorted)
    return profiles_keywords

def loadCorpus():
    all_noun = [lemma.name() for synset in wn.all_synsets('n') for lemma in synset.lemmas()]
    #all_verb = [lemma.name for synset in wn.all_synsets('v') for lemma in synset.lemmas]
    corpus = all_noun
    #corpus.extend(all_verb)
    return corpus

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
re_pattern = r'''([A-Z]\.)+[A-Z]?|\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''

def normalizeTweet(line, corpus):
    text = re.sub(r'\$?\d+(\.\d+)?%?\w*|(@|https?:\/\/)[^ ]+|#|RT', '', line)
    words = [word.lower() for word in nltk.regexp_tokenize(text, re_pattern)]
    words_stemmed = [stemmer.stem(word) for word in words if word not in stopwords]
    words_filtered = [word for word in words_stemmed if word in corpus]
    return words_filtered

def extractTweets(content):
    content_json = json.loads(content)
    tweets_json = content_json['tweets']
    tweets = ''
    for i in range(len(tweets_json)):
        tweet = tweets_json[i]['tweet']
        #tweets += ' ' +  re.sub(r'(@|https?:\/\/)[^ ]+|#|RT', '', tweet)
        tweets += ' ' + tweet
    return tweets

if __name__ == '__main__':
    preprocess()
