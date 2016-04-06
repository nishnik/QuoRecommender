import json
import nltk 
import csv, collections
import enchant
import numpy as np
import sys
from nltk import word_tokenize, pos_tag, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
from collections import Counter
from numpy import genfromtxt

questions = []
vocabQuesCount = Counter()

tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
dictionary = enchant.request_dict("en_US")
stemmer = PorterStemmer()

def clean_ques(ques):
	ques = ques.lower()
	ques = tokenizer.tokenize(ques)
	# for i in range(len(ques)):
	# 	if not enchant.dict_exists(ques[i]):
 	# 		ques[i] = dictionary.suggest(ques[i])[0]
	ques = [stemmer.stem(q) for q in ques]
	return ques

with open('tags.json') as data_file:    
    data = json.load(data_file)

for key in data:
	questions.append(key)

for ques in questions:
	vocab = clean_ques(ques)
	for word in vocab:
		vocabQuesCount[word] += 1

print(vocabQuesCount.most_common(10))
