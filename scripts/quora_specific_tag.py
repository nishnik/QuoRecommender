import nltk 
import csv, collections
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy
from numpy import genfromtxt
import json, gensim, logging

tokenizer = RegexpTokenizer(r'\w+')
model = gensim.models.Word2Vec.load_word2vec_format('vec.txt', binary = False)

with open('xaa') as data_file:
    data = json.load(data_file)# type(data)=dict

def clean_ques_not_stop(query):
    # here query is list of words that are present in the question
    query = query.lower()# converted to lowercase alphabet
    query = tokenizer.tokenize(query) # tokenized
    return query


def chk_in_model(b):
    d = []
    for bb in b:
        if bb in model:
            pass
        else:
            continue
        d.append(bb)
    return d

ques_like = [\
    "Who",
    "What",
    "When",
    "Where",
    "Why",
    "How",
    "Is",
    "Does",
    "Will",
    "Which",
    "Would",
    "Has",
    "Could",
    "Am",
    ]

meta_data = {}
for i in data:
    d = []
    b = clean_ques_not_stop(i)
    d = chk_in_model(b)
    if len(d) == 0:
        continue
    dist = -1
    for q in ques_like:
        if q.lower() == d[0]:
            meta_data[i] = q
            break
        n = model.n_similarity(d, clean_ques_not_stop(q))
        if (n>0.8 and n>dist):
            dist = n
            meta_data[i] = q

with open('x_ques.json') as data_file:
    x_ques = json.load(data_file)


for i in data:
    print (i)
    print (data[i])
    if i in meta_data:
        topic_deep = {}
        print ("-----------------------------------------------", meta_data[i])
        new_topics = x_ques[meta_data[i] + ' "X" Questions']
        topic_new_clean = {}
        for j in new_topics:
            a = clean_ques_not_stop(j)
            c = chk_in_model(a)
            topic_new_clean[j] = c
        d = []
        b = clean_ques_not_stop(i)
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
            if (n > dist and n > 0.7):
                dist = n
                topic_deep[i] = j
                print ("topic_new: ", topic_deep[i], dist, "\n")
        
