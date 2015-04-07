#coding = 'utf-8'
 
import json, re, fileinput

PATTERN=re.compile('@\S+|http://t.co/\w{10}|\t+|\n+')

for line in fileinput.input():
    line = unicode(line, 'utf-8')
    if line == "":
        break
    if line[0] == '{':
	tweet = json.loads(line)
    if tweet.has_key('created_at') and tweet['user']['lang'] == 'en':
	print tweet['user']['lang'] + '\t' +PATTERN.sub(' ', tweet['text']).strip() + '\n'
