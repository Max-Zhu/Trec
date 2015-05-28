# coding=UTF-8
import nltk
import time
from nltk.corpus import brown
from nltk.stem.lancaster import LancasterStemmer
from cPickle import dump
from cPickle import load
# This is a fast and simple noun phrase extractor (based on NLTK)
# Feel free to use it, just keep a link back to this post
# http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
# Create by Shlomi Babluki
# May, 2013
 
 
# This is our fast Part of Speech tagger
start = time.clock()
#############################################################################
brown_train = brown.tagged_sents()     
regexp_tagger = nltk.RegexpTagger(
    [(r'^(NOT|Not)$','SW'),           #stopwords
     (r'(\w+)(\'s)$','SW'),
     (r'@[a-zA-Z]+','@'),
     (r'#[a-zA-Z]+','#'),
     (r'(http|https)://t\.co/[\w]{10}','SURL'),
                                        #短地址
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   #基数   (?
     (r'(-|:|;)$', ':'),                #冒号
     (r'\'*$', 'MD'),                   #任意个'
     (r'(The|the|A|a|An|an)$', 'AT'),   #the和a
     (r'.*able$', 'JJ'),                #able形容词
     (r'^[A-Z].*$', 'NNP'),             #大写开头的专有名词
     (r'.*ness$', 'NN'),                #ness名词
     (r'.*ly$', 'RB'),                  #ly副词
     (r'.*s$', 'NNS'),                  #名词复数
     (r'.*ing$', 'VBG'),                #进行时动词
     (r'.*ed$', 'VBD'),                 #完成时动词
     (r'.*', 'NN')                      #Default
])
unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)
#############################################################################
 
 
# This is our semi-CFG; Extend it according to your own needs
#############################################################################
cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"
cfg["DT+NN"] = "NNI"
cfg["AT+NN"] = "NNI"
#############################################################################
output = open('bigram_tagger.pkl','wb')
dump(bigram_tagger,output,-1)
output.close()
print 'Tagger training done'
end = time.clock()
print "%f s" % (end - start)

"""
test_tags = [tag for sent in brown.sents(categories='news')
             for (word,tag) in bigram_tagger.tag(sent)]
gold_tags = [tag for (word,tag) in brown.tagged_words(categories='news')]
print nltk.ConfusionMatrix(gold_tags,test_tags)
"""
