# coding=UTF-8
import nltk
import time
import json
from nltk.corpus import brown
from nltk.stem.lancaster import LancasterStemmer
from cPickle import load
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import urllib2
import sys

start = time.clock() 
reload(sys)
sys.setdefaultencoding('utf8')
# This is a fast and simple noun phrase extractor (based on NLTK)
# Feel free to use it, just keep a link back to this post
# http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
# Create by Shlomi Babluki
# May, 2013

""" 
# This is our fast Part of Speech tagger
#############################################################################
brown_train = brown.tagged_sents(categories='news')     
regexp_tagger = nltk.RegexpTagger(
    [(r'^(NOT|Not)$','SW'),             #stopwords
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
""" 
 
# This is our semi-CFG; Extend it according to your own needs
#############################################################################
cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"
cfg["DT+NN"] = "NNT"
cfg["AT+NN"] = "NNT"
#############################################################################
input = open('bigram_tagger.pkl','rb')
bigram_tagger = load(input)
input.close()
 
class NPExtractor(object):
 
    def __init__(self):
        super(NPExtractor,self).__init__()
 
    # Split the sentence into singlw words/tokens
    def tokenize_sentence(self, sentence):
        pattern = r'''(?x)  # 空格为\s & 多行 & 忽略注释
                    ([A-Z]\.)+  # e.g. U.S.A
                |   (http|https)://t\.co/[\w]{10}
                |   \w+(-\w+)*
                |   \$?\d+(\.\d+)?%?
                |   \.[\.]+
                |   @[a-zA-Z]+
                |   [\s]

                    '''
        #tokens = nltk.word_tokenize(sentence)
        tokens = nltk.regexp_tokenize(sentence,pattern)
        return tokens
 
    # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged
    
    def inner_similarity(self, tags, matches):
        simi = []
        for t1 in matches:
            try:
                t1sy = wn.synset(t1+'.n.01')
                for t2 in tags:
                    t2sy = wn.synset(t2+'.n.01')
                    p_simi = t2sy.path_similarity(t1sy)
                    # print p_simi
                    if p_simi> 0.2 and p_simi< 1.0 :
                        simi.append(t2)
            except nltk.corpus.reader.wordnet.WordNetError:
                pass
        matches.extend(simi)
        
        return matches


    # Extract the main topics from the sentence
    def extract(self, input_line):
        st=nltk.PorterStemmer()
         
        tokens = self.tokenize_sentence(input_line)
        #tokens = [st.stem(word) for word in tokens] 
        tags = self.normalize_tags(bigram_tagger.tag(tokens))
 
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break
 
        matches = []
        for t in tags:
            t[0].replace('  ','')
            t[0].replace('   ',' ')

            if t[0]==" ":
                

            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "SURL":
            if t[1] == "NNP" or t[1] == "NNI"  or t[1][:1] == "VB":
            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN" or t[1] == "SURL":
                matches.append(t[0])

            if t[1] == "NNT":
                last_space = t[0].find(" ")
                matches.append(t[0][last_space+1:])

            if t[1] == "#":
                matches.append(t[0][1:])

            if t[1] == "SURL":
                # print 'SURL Found'
                matches.append(t[0])
                #取h1部分,↑↓选一
                """s_url = t[0]
                f = urllib2.urlopen(s_url, timeout=5).read()
                soup= BeautifulSoup(f)
                # print '%s%s' % (soup.h1.string,'\ttitle')
                matches.append(soup.h1.string)"""
        
        return matches
        # return self.inner_similarity(tokens,matches)
 
 

def main():
    #test_text = "Jadakiss ft. Anthony Hamilton - Why? -&gt; http://t.co/RGsBMcs6L9 #NowPlayingOKCJamz"
    #test_ex = NPExtractor(test_text.strip())
    #print test_ex.tokenize_sentence(test_text)

    
    #sentence = "The dog can easily be upset by circumstances around them. This can result in poor behaviour and misery for their"
    f = open('text_en.txt','rU')
    pass_count = 0
    all_matches = []
    np_extractor = NPExtractor()
    for line in f:
        try:
            #print text_line.strip()
            
            result = np_extractor.extract(line.strip())
            all_matches.append(result)
            # output = open('np.txt','a')
            # output.write(str(result))
            # output.write('\n')
            # output.close()
        except (UnicodeDecodeError,ValueError):
            pass_count += 1
    f.close()
    output = open('np.txt','w')
    for l in all_matches:
        strl = str(l)
        strl = strl.replace('  ','')
        strl = strl.replace('   ',' ')
        output.write(strl)
        output.write('\n')
    output.close()
    print 'pass: %s'  % pass_count
    
if __name__ == '__main__':
    main()

end = time.clock()
print "%f s" % (end - start)
