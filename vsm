# -*-coding:utf-8-*-
'''
Created on May 24, 2015
@author Max
'''

from ldig import *
from utils import *
from math import sqrt
import gensim
import nltk
import threading
import time
import datetime
import os
import logging
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class MyThread(threading.Thread):
    """docstring for classify"""
    def __init__(self, thread_name, event):
        super(MyThread, self).__init__()
        self.name = thread_name
        self.threadEvent = event
        self.threshold = 0.5
    
    def preprocess(self, w2vModelName, profilesFile):
        self.loadWord2VecModel(w2vModelName)
        self.profiles, self.profiles_name = loadProfiles(profilesFile)

    def run(self):
        logging.info('%s is waiting...' % self.name)
        self.threadEvent.wait()
        logging.info('%s is running...' % self.name)
        self.process()

    def loadWord2VecModel(self, modelName):
        logging.info('loading model %s ...' % modelName)
        # time.sleep(5)
        self.w2vModel = gensim.models.Word2Vec.load_word2vec_format(modelName, binary=False)
        # self.w2vModel = "test"
        logging.info('model %s loaded' % modelName)

    def process(self):
        classify_result = [{} for i in range(len(self.profiles_name))]
        detector = ldig('model.latin')  # ldig models
        param, labels, trie = detector.load_params()
        stopwords = nltk.corpus.stopwords.words('english')
        
        data_file_path = sys.argv[1]
        files = os.listdir(data_file_path)
        files.sort()
        for f in files:
            filename = os.path.join(data_file_path, f)
            logging.info(filename)
            count = 0
            for line in gzip.open(filename, 'rb'):
                tweet_text,  timestamp, urls = extract_text(line)
                if tweet_text == "":
                    continue

                # tweet_text = url_expansion(urls)
                lang = predict_lang(param, labels, trie, 'en\t'+tweet_text)
                if lang != 'en':
                    continue

                tweet = normalizeTweet(tweet_text, stopwords)
                if len(tweet) <= 3:
                    continue

                count += 1
                if count % 1000 == 0:
                    logging.info('count: ' + str(count))    # print 'count: ', count

                topicID, similarity = self.classify(tweet, self.profiles)
                if topicID != -1:
                    classify_result[topicID][timestamp + '\t' + tweet_text] = similarity
                # logging.info('Best Match to profile: '+ str(topicID+1))
        dumpResult(classify_result, self.profiles_name)

    def classify(self, tweet, profiles):
        # logging.info('classify tweet_tfidf: ' + ' '.join(tweet))
        bestMatch = -1; maxSimilarity = 0.0
        for index, profile in enumerate(profiles):
            similarity = self.calcSim(tweet, profile)
            if similarity > maxSimilarity:
                maxSimilarity = similarity
                bestMatch = index
        # if maxSimilarity > self.threshold:
        return bestMatch, maxSimilarity

    def calcSim(self, tweet, profile_keywords):
        similarity = 0.0;sim_normalize = 0.0; cnt = 0
        for profile_keyword in profile_keywords:
            sim_values = []
            for token in tweet:
                try:
                    sim = self.w2vModel.similarity(token, profile_keyword)
                except Exception, e:
                    sim = 1 if (token == profile_keyword) else 0	
                
                if sim > 0.5: sim_values.append(sim)
            if len(sim_values) == 0:
                continue
            similarity += sum(sim_values); cnt += len(sim_values) ** 2
        if cnt != 0:
            length  = sqrt(len(profile_keywords)*cnt)
            sim_normalize = similarity/length
        return sim_normalize

def w2v_test():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    signal = threading.Event()
    classifier = MyThread("classify_thread", signal)
    classifier.preprocess('../wiki.en.text.vector', 'profiles')
    classifier.start()
    # isReady = detectTweetStream(2015, 6, 19, 11, 35, 0, 0)
    isReady = True
    if isReady:
        signal.set()

def detectTweetStream(year, month, d, h, m, s, ms):
    start = datetime.datetime(year, month, d, h, m, s, ms)
    delta = (start - datetime.datetime.now()).seconds
    logging.info('waiting secondes: ' + str(delta))
    time.sleep(delta)
    logging.info('tweet stream is ready')
    isReady = True
    return isReady

if __name__ == '__main__':
    '''
    parser = optparse.OptionParser()
    parser.add_option("-m", dest="model", help="word2vec model")
    parser.add_option("-p", dest="profiles", help="user profiles")

    (options, args) = parser.parse_args()
    if not options.model: parser.error("need word2vec models (-m)")
    '''
    w2v_test()
