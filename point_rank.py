# -*-coding:utf-8-*-
__author__ = 'Max-Zhu'

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.externals import joblib
import logging
import json
import nltk
import math
import re
import os

class PointRanker:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.pattern   = r'''([A-Z]\.)+[A-Z]?|\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''
        self.url_pattern = re.compile(r'(https?://)[^ ]+|#|RT|@')
    
    def extract_features(self, tweet):
        text = tweet['text']
        text = self.url_pattern.sub('', text)

        # Non_text feature
        has_hashtag = 1 if len(tweet['entities']['hashtags']) != 0 else 0
        has_url     = 1 if len(tweet['entities']['urls']) != 0 else 0
        has_user_mentions = 1 if len(tweet['entities']['user_mentions']) != 0 else 0
        retweet_count   = tweet['retweeted_status']['retweet_count'] if 'retweeted_status' in tweet else 0
        log_rtcount     = math.log(retweet_count, 2) if retweet_count>0 else 0
        rtcount_level   = len(str(retweet_count)) if retweet_count>0 else 0

        # Author's feature
        has_default_profile = 1 if tweet['user']['default_profile'] == '1' else 0
        followers_count = tweet['user']['followers_count']
        log_followers   = math.log(followers_count, 2) if followers_count>0 else 0
        statuses_count  = tweet['user']['statuses_count']
        log_statuses    = math.log(statuses_count, 2) if statuses_count>0 else 0

        # Text feature
        words           = nltk.regexp_tokenize(text, self.pattern)
        words_length    = len(words)
        stopwd_length   = len(filter(lambda w: w.lower() in self.stopwords, words))
        stopwd_ratio    = float(stopwd_length) / words_length
        captwd_length   = len(filter(lambda w: w[0].isupper(), words))
        captwd_ratio    = float(captwd_length) / words_length
        character_ratio = float(len(text)) / words_length
        features = [has_hashtag, has_url, has_user_mentions, has_default_profile,
                    words_length, stopwd_length, stopwd_ratio, captwd_length, captwd_ratio,
                    log_followers, log_statuses, character_ratio, retweet_count, rtcount_level,
                    log_rtcount]
        return features

    def feature_extraction(self):
        with open('result_test.txt', 'rU') as fr:
            with open('feature.txt', 'w') as fw:
                while True:
                    line = fr.readline()
                    if line == '':
                        break
                    tweet = json.loads(line)
                    label = tweet['importance']
                    features = self.extract_features(tweet)
                    features_format = [str(index+1) + ":" + str(feature) for index, feature in enumerate(features)]
                    fw.write(label + '\t' + ':'.join(features_format) + '\n')

    def train_lrmodel(self):
        logging.info('training logistic regression model...')
        if not os.path.exists('feature.txt'):
            self.feature_extraction()
        train_data, train_target = load_svmlight_file('feature.txt')
        lr = LogisticRegression()
        lr_model = lr.fit(train_data, train_target)
        scores = cross_validation.cross_val_score(lr, train_data, train_target, cv=5)
        print scores
        joblib.dump(lr_model, 'lr.model')
        logging.info('logistic regression model done and saved!')

    def predict(self, tweet):
        if not os.path.exists('lr.model'):
            self.train_lrmodel()
        logging.info('loading logistic regression model...')
        lr_model = joblib.load('lr.model')
        feature = self.extract_features(tweet)
        print feature
        result = lr_model.predict([feature])
        return result


def test():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    ranker = PointRanker()
    with open('result_test.txt', 'rU') as fr:
        line = fr.readline()
        tweet = json.loads(line)
        label = tweet['importance']
        predict = ranker.predict(tweet)
        print label, ' : ', predict


if __name__ == '__main__':
    test()


