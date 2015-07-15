# -*-coding:utf-8-*-
__author__ = 'Max-Zhu'

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np
import utils
import logging
import json
import nltk
import math
import re
import os


class PointRanker:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.token_pattern   = r'''([A-Z]\.)+[A-Z]?|\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''
        self.url_pattern = re.compile(r'(https?://)[^ ]+|#|RT|@')
        self.data_train, self.target_train, self.data_test, self.target_test = self.load_data()
        self.threshold = [0.5 for i in range(225)]    # 初始推送阈值，根据历史统计TopK的平均取值
        self.received_count = [0 for i in range(225)]  # 当前已接收相关推文数
        self.pushed_count = [0 for i in range(225)]   # 当前已推送推文数
        self.total_num_predicted = utils.pred_total_related()   # 预测当天总相关推文数

    def extract_features(self, tweet):
        text = tweet['text']
        text = self.url_pattern.sub('', text)

        # Non_text feature
        has_hashtag     = 1 if len(tweet['entities']['hashtags']) != 0 else 0
        has_url         = 1 if len(tweet['entities']['urls']) != 0 else 0
        has_user_mentions = 1 if len(tweet['entities']['user_mentions']) != 0 else 0
        has_pic         = 1 if 'media' in tweet['entities'] else 0

        retweet_count   = tweet['retweeted_status']['retweet_count'] if 'retweeted_status' in tweet else 0
        log_rtcount     = math.log(retweet_count, 2) if retweet_count > 0 else 0
        rtcount_level   = len(str(retweet_count)) if retweet_count > 0 else 0

        # Author's feature
        followers_count = tweet['user']['followers_count']
        log_followers   = math.log(followers_count, 2) if followers_count > 0 else 0
        statuses_count  = tweet['user']['statuses_count']
        log_statuses    = math.log(statuses_count, 2) if statuses_count > 0 else 0

        # Text feature
        words           = nltk.regexp_tokenize(text, self.token_pattern)
        words_length    = len(words)

        has_subjective  = 1 if len([sw for sw in ['i', 'my', 'me'] if sw in words]) != 0 else 0
        stopwd_length   = len(filter(lambda w: w.lower() in self.stopwords, words))
        stopwd_ratio    = float(stopwd_length) / words_length

        captwd_length   = len(filter(lambda w: w[0].isupper(), words))
        captwd_ratio    = float(captwd_length) / words_length

        character_ratio = float(len(text)) / words_length
        ellipsis        = 1 if re.search(r'\.\.+', text) else 0

        features = [has_hashtag, has_url, has_user_mentions, has_pic, has_subjective, ellipsis,
                    words_length, stopwd_length, stopwd_ratio, captwd_length, captwd_ratio,
                    log_followers, log_statuses, character_ratio, retweet_count, rtcount_level,
                    log_rtcount]
        return features

    def discret_feature(self, section):
        """对特征按区间进行离散化处理"""
        assert isinstance(section, (tuple, list))
        size = len(section)

        def divide(x):
            for i in range(size):
                if x <= section[i]:
                    return i
            return size
        return np.frompyfunc(divide, 1, 1)

    def feature_extraction(self, input_file_name, output_file_name):
        with open(input_file_name, 'rU') as fr:
            with open(output_file_name, 'w') as fw:
                while True:
                    line = fr.readline()
                    if line == '':
                        break
                    tweet = json.loads(line)
                    label = tweet['importance']
                    features = self.extract_features(tweet)
                    features_format = [str(index+1) + ":" + str(feature) for index, feature in enumerate(features)]
                    fw.write(str(label) + '\t' + '\t'.join(features_format) + '\n')

    def load_data(self):
        if not os.path.exists('features_train.txt'):
            self.feature_extraction('train.txt', 'features_train.txt')
        data_train, target_train = load_svmlight_file('features_train.txt')

        if not os.path.exists('features_test.txt'):
            self.feature_extraction('test.txt', 'features_test.txt')
        data_test, target_test = load_svmlight_file('features_test.txt')

        normalizer = Normalizer().fit(data_train)
        data_train = normalizer.transform(data_train)
        data_test = normalizer.transform(data_test)

        return data_train.toarray(), target_train, data_test.toarray(), target_test

    def train_model(self, classify_model):
        model_name = str(classify_model).split('(')[0]
        print '='*30
        if os.path.exists('models/' + model_name + '.model'):
            logging.info('loading model...' + model_name)
            clf = joblib.load('models/' + model_name + '.model')
        else:
            logging.info('training model...' + model_name)
            classifier = Pipeline([
                ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
                ('classification', classify_model)
            ])
            clf = classifier.fit(self.data_train, self.target_train)

            joblib.dump(clf, 'models/' + model_name + '.model')
            logging.info('model done and saved!')
        return clf

    def test_model(self, model):
        logging.info('testing model...')
        predict_point = model.predict(self.data_test)
        print '#'*30
        print metrics.confusion_matrix(self.target_test, predict_point)
        print metrics.classification_report(self.target_test, predict_point)

    def predict(self, model, tweet, profile_id):
        feature = self.extract_features(tweet)
        predict_point = model.predict([feature])
        if predict_point > 0.5:  # 判定为相关的推文，进一步决策是否推送
            self.received_count[profile_id] += 1
            if self.pushed_count != 10: # 未达到推送上限
                if predict_point > self.threshold[profile_id]:   # 超过当前设定阈值，则推送
                    self.push(tweet)
                    self.pushed_count[profile_id] += 1
                received_ratio = self.received_count[profile_id] / self.total_num_predicted[profile_id]
                if received_ratio > 0.618:  # 超过黄金分割线，未推满，需调整阈值
                    self.threshold[profile_id] *= 1.5-received_ratio  # 策略1：按接收比例调整
            utils.save_relate_tweet(profile_id, tweet)
        # return predict_point, ' : ', feature

    def push(self, tweet):
        """推送推文API"""
        print 'pushing...'


def test():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    ranker = PointRanker()
    lr = LogisticRegression()
    rc = RidgeClassifier()
    sc = SGDClassifier()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    rfc = RandomForestClassifier(n_estimators=100)
    etc = ExtraTreesClassifier(n_estimators=100)
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    svm = SVC()
    abc = AdaBoostClassifier(n_estimators=100)
    gbc = GradientBoostingClassifier(n_estimators=100)
    # classifiers = [(lr, 'logistic_regression'), (rfc, 'random_forest'), (etc, 'extra_tree')
    # (dt, 'decision_tree'), (knn, 'knn'), (svm, 'svm'), (abc, 'ada_boost'), (gbc, 'gradient_boost')]
    classifiers = [lr, rc, sc, mnb, bnb, rfc, etc, dt, knn, svm, abc, gbc]

    for classifier in classifiers:
        clf = ranker.train_model(classifier)
        ranker.test_model(clf)
        break

    # model = ranker.train_model(lr, 'lr')
    # ranker.test_model(model)
    #
    # with open('test.txt', 'rU') as fr:
    #     line = fr.readline()
    #     tweet = json.loads(line)
    #     label = tweet['importance']
    #     predict = ranker.predict(tweet)
    #     print label, ' : ', predict


def plot_data():
    """数据绘图"""
    # pr = PointRanker()


if __name__ == '__main__':
    test()
