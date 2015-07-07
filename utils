__author__ = 'Max-Zhu'

from bs4 import BeautifulSoup
import nltk
import json, re
import logging
import urllib2
import xml.sax


def loadProfiles(profileFile):
    logging.info('loading profiles...' + profileFile)
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = XMLHandler()
    parser.setContentHandler(handler)
    parser.parse(profileFile)
    result = handler.getDict()
    return result['title'], result['num']

class XMLHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.mapping = {'num':[], 'title':[]}
        self.content = ""
        self.tag = ""

    def startElement(self, tag, attributes):
        self.content = ""
        self.tag = tag

    def characters(self, content):
        if self.tag == 'num':
            self.content = content.strip()[-5:]
        elif self.tag == 'title':
            self.content = content.split()

    def endElement(self, tag):
        if tag in ['num', 'title']:
            self.mapping[tag].append(self.content)

    def getDict(self):
        return self.mapping

def dumpResult(results, names):
    for index, result in enumerate(results):
        fw = open('./result/profile_'+ names[index] + '.txt', 'w')
        fw.write('similarity' + '\t' + 'tweet text\n')
        fw.write('================================\n')
        result_sorted = sorted(result.iteritems(), key = lambda item: item[1], reverse = True)
        for key, value in result_sorted:
            fw.write(str(value) + "\t" + key + "\n")
        fw.close()

def normalizeTweet(line, stopwords):
	#logging.info('input tweet: ' + line)
    line = line.decode('utf-8')
    re_pattern = r'''([A-Z]\.)+[A-Z]?|\w+([-']\w+)*|\$?\d+(\.\d+)?%?\w*'''
    text = re.sub(r'\$?\d+(\.\d+)?%?\w*|(https?:\/\/)[^ ]+|#|RT|@', '', line)
    words = [word for word in nltk.regexp_tokenize(text.lower(), re_pattern) if word not in stopwords]
    #words = [wn.morphy(word) for word in words]
    #logging.info('normalized tweet: ' + ' '.join(words_filtered))
    return words

def url_expansion(urls):
    content = ""
    for url in urls:
        try:
            html = urllib2.urlopen(url).read()
            bs = BeautifulSoup(html)
            if bs:
                title = bs.title.string
                if title != "":
                    title = title.sub('\t+|\n+', ' ')
                    content += title
        except Exception, e:
            # print 'exception: ', e.message, type(e), 'url:', url, 'title: ', title
            pass
    return content

def extract_text(line):
    line = line.decode('utf-8')
    origin_text = ""; text = ""; urls = []
    timestamp = ""
    if line[0] == '{':
        tweet = json.loads(line)
        if tweet.has_key('created_at') and tweet['user']['lang'] == 'en':
            origin_text = re.sub(r'\t+|\n+', ' ', tweet['text'])
            urls  = [url_detail['expanded_url'] for url_detail in tweet['entities']['urls']]
            timestamp = tweet['created_at']
            # text = re_filtUrl.sub(' ', origin_text).strip()
    return origin_text, timestamp, urls
