# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from gsearch import *
import xml.sax

def expan_profile():
    google = GoogleAPI()
    profiles, numbers = load_profiles('./profiles')
    profiles_expanded = []
    stopwords = nltk.corpus.stopwords.words('english')

    for index, profile in enumerate(profiles):
        print profile
        contents_expanded = []
        results , has_error = google.search(profile)
        if has_error:
            break
        titles = ""; snippets = ""
        for r in results:
            titles += r.getTitle() + " "
            snippets += r.getContent() + " "
            contents_expanded.append(r.getTitle() + '\t\t' + r.getContent())

        content = titles + snippets
        content = re.sub('<br/>|<b>|</b>', '', content)
        words = re.findall('\w+', content.lower())
        words = [word for word in words if not word in stopwords if len(word) > 1]
        profiles_expanded.append(words)
        dump_contents('expanded/profile_'+numbers[index], contents_expanded)
    dump_profiles(profiles_expanded)

def expan_keywords():
    profiles_expanded = read_profiles()
    for content in profiles_expanded:
        term_freq = cal_tf(content)
        high_tf = [key for key, val in term_freq if val > 5]
        profile_expaned = ' '.join(high_tf)
        print profile_expaned
        profiles_expanded.append(profiles_expanded)

def cal_tf(words_list):
    tf = collections.defaultdict(int)
    for word in words_list:
        tf[word] += 1

    tf = sorted(tf.iteritems(), key=lambda term:term[1], reverse=True)
    return tf

def load_profiles(profile_file):
    # logging.info('loading profiles...' + profileFile)
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = XMLHandler()
    parser.setContentHandler(handler)
    parser.parse(profile_file)
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
            self.content = content

    def endElement(self, tag):  
        if tag in ['num', 'title']:
            self.mapping[tag].append(self.content)

    def getDict(self):  
        return self.mapping

def dump_contents(file_name, contents):
    with open(file_name, 'w') as fw:
        for content in contents:
            fw.write(content + '\n')

def dump_profiles(profiles):
    with open('profiles_expanded', 'w') as fw:
        for profile in profiles:
            content = ' '.join(profile)
            fw.write(content + '\n')

def read_profiles():
    profiles = []
    with open('profiles_expanded', 'r') as fr:
        line = fr.readline()
        profile = line.split(' ')
        profiles.append(profile)
    return profiles

if __name__ == '__main__':
    expan_profile()
