# For hierarichal topic using the data in hierarchy.json

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

def clean_query(query):
    # here query is list of words that are present in the question
    query = query.lower()# converted to lowercase alphabet
    query = tokenizer.tokenize(query) # tokenized
    query = [q for q in query if q not in stop] # removed stop words
    return query


def chk_in_model(b):
    d = []
    for bb in b:
        try:
            if bb in model:
                pass
        except:
            continue
        d.append(bb)
    return d

tags_clus = dict()
with open('xaa') as data_file:
    data = json.load(data_file)# type(data)=dict

with open('hierarchy.json') as data_file:
    hierarchy = json.load(data_file)

topics = hierarchy.keys()

topic_clean = {}

for j in topics:
    a = clean_query(j)
    c = chk_in_model(a)
    topic_clean[j] = c

for line, value in data.items():
    for x in value:
        tags.append(x)

tag_count = collections.Counter(tags)

clus_topic = {}


for i in data:
    d = []
    b = clean_query(i)
    d = chk_in_model(b)
    if len(d) == 0:
        continue
    dist = -1
    for j in topic_clean:
        try:
            if (len(topic_clean[j]) == 0):
                continue
            n = model.n_similarity(topic_clean[j], d)
        except:
            # print ("outer loop",sys.exc_info()[0], topic_clean[j], j, d)
            continue
        # print (n)
        n = abs(n)
        if (n > dist and n > 0.55):
            dist = n
            clus_topic[i] = j
    # print (i)
    # print (data[i])
    # if (i in clus_topic):
    #     print (clus_topic[i])
for i in data:
    try:
        print (i)
        print (data[i])
        if (i in clus_topic):
            clus_deep_topic = {}
            print ("Clus got: ", clus_topic[i], "\n")
            new_topics = hierarchy[clus_topic[i]]
            topic_new_clean = {}
            for j in new_topics:
                a = clean_query(j)
                c = chk_in_model(a)
                to_app = []
                to_chk = clean_query(clus_topic[i])
                for a in c:
                    if a not in to_chk:
                        to_app.append(a)
                topic_new_clean[j] = to_app
                topic_new_clean[j] = c
            d = []
            b = clean_query(i)
            d = chk_in_model(b)
            if len(d) == 0:
                continue
            dist = -1
            for j in topic_new_clean:
                try:
                    if (len(topic_new_clean[j]) == 0):
                        continue
                    n = model.n_similarity(topic_new_clean[j], d)
                except:
                    # print ("outer loop",sys.exc_info()[0], topic_clean[j], j, d)
                    continue
                # print (n)
                n = abs(n)
                if (n > dist and n > 0.55):
                    dist = n
                    clus_deep_topic[i] = j
            if (i in clus_deep_topic):
                print ("clus_deep_topic: ", clus_deep_topic[i], "\n")
    except:
        pass


print (i)
print (data[i])
if (i in clus_topic):
    clus_deep_topic = {}
    print ("Clus got: ", clus_topic[i], "\n")
    new_topics = hierarchy[clus_topic[i]]
    topic_new_clean = {}
    for j in new_topics:
        a = clean_query(j)
        c = chk_in_model(a)
        print (c)
        to_app = []
        to_chk = clean_query(clus_topic[i])
        for a in c:
            if a not in to_chk:
                to_app.append(a)
        topic_new_clean[j] = to_app
        print (to_app)
    d = []
    b = clean_query(i)
    d = chk_in_model(b)
    if len(d) == 0:
        pass
    dist = -1
    for j in topic_new_clean:
        try:
            if (len(topic_new_clean[j]) == 0):
                pass
            n = model.n_similarity(topic_new_clean[j], d)
        except:
            # print ("outer loop",sys.exc_info()[0], topic_clean[j], j, d)
            pass
        # print (n)
        n = abs(n)
        if (n > dist and n > 0.55):
            dist = n
            print ("clus_deep_topic", j)
            clus_deep_topic[i] = j
    if (i in clus_deep_topic):
        print ("clus_deep_topic: ", clus_deep_topic[i], "\n")