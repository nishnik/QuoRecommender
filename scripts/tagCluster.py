# uses chinese restraunt process

import nltk 
import csv, collections
#from nltk import word_tokenize, pos_tag, sent_tokenize, RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import enchant
import numpy
from numpy import genfromtxt
import sys
import json, gensim, logging

tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
# dictionary = enchant.request_dict("en_US")
tags = []
model = gensim.models.Word2Vec.load_word2vec_format('vec.txt', binary = False)

tags_clus = dict()
with open('xaa') as data_file:
    data = json.load(data_file)# type(data)=dict

for line, value in data.items():
    for x in value:
        tags.append(x)

tag_count = collections.Counter(tags)

ind = 0
for i, j in tag_count.most_common():
    tags_clus[ind] = [i, i] 
    ind += 1

def clean_ques(query):
    # here query is list of words that are present in the question
    query = query.lower()# converted to lowercase alphabet
    query = tokenizer.tokenize(query) # tokenized
    query = [q for q in query if q not in stop] # removed stop words
    return query

no_topic = 0
def cluster():
    ind = 0
    not_chk = []
    for i in range(1, len(tags_clus)):
        d = []
        b = clean_ques(tags_clus[i][0])
        for bb in b:
            try:
                model[bb]
            except:
                # print ("inner loop",sys.exc_info()[0], bb, b)
                continue
            d.append(bb)
        if len(d) == 0:
            not_chk.append(i)
            continue
        dist = -1
        try:
            for j in range(0, i):
                if j in not_chk:
                    continue
                try:
                    a = clean_ques(tags_clus[j][1])
                    c = []
                    for aa in a:
                        try:
                            model[aa]
                        except:
                            continue
                        c.append(aa)
                    if (len(c) == 0):
                        continue
                    n = model.n_similarity(c, d)
                except:
                    #print ("outer loop",sys.exc_info()[0], a, c, b, d)
                    continue
                # print (n)
                if (n > dist and n > 0.55):
                    dist = n
                    tags_clus[i][1] = tags_clus[j][1]
                    # print (tags_clus[i][0], " -> ", tags_clus[i][1], n)
        except:
            # print ("outer*2 loop",sys.exc_info()[0])
            continue
    for i in range(len(tags_clus)):
        print (tags_clus[i][0], " -> ", tags_clus[i][1])

print ('here')
cluster()