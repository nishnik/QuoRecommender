import nltk 
import csv, collections
#from nltk import word_tokenize, pos_tag, sent_tokenize, RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import enchant
import numpy
from numpy import genfromtxt
import sys
import json, gensim, logging
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import random
from collections import Counter
from scipy import spatial

model = gensim.models.Word2Vec.load_word2vec_format('../data/vectors.bin', binary = True)
with open('../data/xaa') as data_file:
    data=json.load(data_file)# type(data)=dict

ques, tag = [], []
for key, value in data.items():
    ques.append(key)
    tag.append(data[key])

tags = []
with open('../data/out') as f:
    s = f.readlines()
    for line in s:
        tags.append(line)

questions_train, questions_test, tags_train, tags_test = train_test_split(ques, tag, test_size=0.1, random_state = random.randint(1, 100))
print (type(tags_test))
tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
dictionary = enchant.request_dict("en_US")
#stemmer = SnowballStemmer("english")


# def getcosine(v1, v2):
#   return 1 - spatial.distance.cosine(v1, v2)

def clean_ques(query):
    query = query.lower()# converted to lowercase alphabet
    query = tokenizer.tokenize(query) # tokenized
    query = [q for q in query if q not in stop] # removed stop words
    return query

def wordvec(word):
    try:
        return model[word]
    except KeyError:
        pass    
    return numpy.zeros(len(model["one"]))   

#Get the Word Centroid Distance
def wcd(sent1, sent2):
    # here sent1 & sent2 both are list of words
    if(len(sent1)>0 and len(sent2)>0):
        s1 = wordvec(sent1[0])
        s2 = wordvec(sent2[0])
    else:
        return 10000
    for i in range(1,len(sent1)):
        s1 = s1 + wordvec(sent1[i])
    for i in range(1,len(sent2)):
        s2 = s2 + wordvec(sent2[i])
    s1 = s1 / len(sent1)
    s2 = s2 / len(sent2)    
    return numpy.linalg.norm(s1 - s2) # returns the norm of the difference of the two vectors   
#print(word1)

#Get the Relaxed Word Mover Distance
def rwmd(sent1, sent2):
    s1, s2 = 0, 0
    dist1 , dist2 = 0, 0
    # dist1 is distance to move from sent1 to sent2
    if len(sent1) == 0 or len(sent2) == 0:
        return 0
    for i in range(len(sent1)):
        d = numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[0]))
        #d = getcosine(wordvec(sent1[i]) , wordvec(sent2[0]))
        val = 0
        for j in range(len(sent2) - 1):
            if (numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[j + 1])) < d): # calculating the minimum distance of sent1[i] with every sent2[j]
                d = numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[j + 1]))
                #d = getcosine(wordvec(sent1[i]) , wordvec(sent2[j + 1]))
                val = j + 1
        dist1 = dist1 + (1.0 / len(sent1)) * d  
    # dist2 is distance to move from sent2 to sent1 
    for i in range(len(sent2)):
        d = numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[0]))
        #d = getcosine(wordvec(sent2[i]) , wordvec(sent1[0]))
        val = 0
        for j in range(len(sent1) - 1):
            if (numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[0])) < d):
                d = numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[j + 1]))
                #d = getcosine(wordvec(sent2[i]) , wordvec(sent1[j + 1]))
                val = j + 1
        dist2 = dist2 + (1.0 / len(sent2)) * d  
    return max(dist1, dist2)            

#Get the one sided Relaxed Word Mover Distance
def rwmd_(sent1, sent2):
    s1, s2 = 0, 0
    dist1 , dist2 = 0, 0
    # dist1 is distance to move from sent1 to sent2
    if len(sent1) == 0 or len(sent2) == 0:
        return 0
    for i in range(len(sent1)):
        d = numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[0]))
        val = 0
        for j in range(len(sent2) - 1):
            if (numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[j + 1])) < d): # calculating the minimum distance of sent1[i] with every sent2[j]
                d = numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[j + 1]))
                val = j + 1
        dist1 = dist1 + (1.0 / len(sent1)) * d  
    return dist1

def getwcd(query, num):
    dic={}
    for i in range(len(ques)):
        if(len(ques[i])==0):
            continue
        ques1=clean_ques(ques[i])
        val = wcd(query,ques1)
        if(len(dic)<num):
            dic[ques[i]]=val
        else:
            m=max(dic,key=dic.get)
            if(dic[m]>val):
                del dic[m]
                dic[ques[i]]=val
    return list(dic.keys())

def getrwmd(query, kwcd, num):
    dic={}
    for i in range(len(kwcd)):
        ques1=clean_ques(kwcd[i])
        val=rwmd(query,ques1)
        #print (kwcd[i], val)
        if (len(dic)<num):
            dic[kwcd[i]]=val
        else:
            m=max(dic,key=dic.get)
            if(dic[m]>val):
                del dic[m]
                dic[kwcd[i]]=val
    return list(dic.keys())
        #create priority queue to store the dist
    #return top num values  

def getkNN(query, num):
    kwcd = getwcd(query, 10 * num)
    knn = getrwmd(query, kwcd, num)
    return knn

def rank_dic_ques(dic):
    m = max(dic.values())
    for i in dic:
        dic[i] = 1.0 - float(dic[i]) / (m * 1.0)
    return dic  

def rank_dic_tags(dic):
    m = max(dic.values())
    for i in dic:
        dic[i] = float(dic[i]) / (m * 1.0)
    return dic  
#get the top 20 tags by question similarity
def getTagsSimilarQues(query, k = 20):
    query = clean_ques(query)
    knn = getkNN(query, 50)
    #print(knn)
    #return tags of all 50 questions returned with count of occurrence
    tags=[]
    for i in knn:
        tags.extend(data[i])
    #tag1 = Counter(tags).most_common(k)
    dic = {}
    for w, c in Counter(tags).most_common(k):
        dic[w] = c
    return rank_dic_ques(dic)    

#get the top 20 tags by tag similarity to a question
def similar_tags(ques, num = 20):
    dic = {}
    for i in range(len(tags)):
        try:
            val=rwmd_(clean_ques(tags[i]), clean_ques(ques))
            if (len(dic)<num):
                dic[tags[i]]=val
            else:
                m = max(dic,key=dic.get)
                if(dic[m]>val):
                    del dic[m]
                    dic[tags[i]]=val
        except KeyError:
            pass                            
    return rank_dic_tags(dic)

pred = []
tt = []

def combine_linear(dic1, dic2):
    dic = {}
    for a in dic1:
        dic[a.strip()] = dic1[a]
    for a in dic2:
        if a.strip() in dic:
            if (dic[a.strip()] > dic2[a]):
                dic[a.strip()] = dic2[a]
        else:
            dic[a.strip()] = dic2[a]
    return dic

def dic_to_lis_sort(dic):
    lis = []
    for a in dic:
        lis.append([dic[a], a])
    lis.sort()
    to_ret = []
    for a,b in lis:
        to_ret.append(b)
    return (to_ret)

def recall_calculate(tt, lis):
    found = 0.0
    for a in tt:
        if a in lis:
            found = found + 1
            continue
    return float(found/(1.0 * len(tt)))

print len(questions_test)

recall_10 = {}
total_rec = 0.0
for i in range(len(questions_test)):
    print (i + 1)
    q = input("Enter a Question")
    #q = questions_test[i]
    dic1 = getTagsSimilarQues(q, 20)
    dic2 = similar_tags(q, 20)
    dic = combine_linear(dic1, dic2)
    lis = dic_to_lis_sort(dic)
    tt = data[questions_test[i]]
    lis = lis[:10]
    a = recall_calculate(tt, lis)
    total_rec = total_rec + a
    to_write = a, tt, lis
    recall_10[i] = to_write

print ("Total recall = ", float(total_rec/(1.0 * len(questions_test))))

with open('../data/recall.json','w') as outfile:
    json.dump(recall_10, outfile, indent=2, separators=(',', ': '))

#precision, recall, f1_score, support = metrics.precision_recall_fscore_support(tt, pred, average='micro')
# print("precision: %0.2f %%" % (precision * 100))
# print("recall: %0.2f %%" % (recall * 100))
# print("f score: %0.4f" % f1_score)