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

tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
dictionary = enchant.request_dict("en_US")
tags = []
#model = gensim.models.Word2Vec.load_word2vec_format('vec.txt', binary = False)
model = gensim.models.Word2Vec.load_word2vec_format('vec.txt', binary = False)
print 'herebnklds'
tags_clus = dict()
with open('text.json') as data_file:
	data = json.load(data_file)# type(data)=dict

for line, value in data.items():
	for x in value:
		tags.append(x)

tag_count = collections.Counter(tags)

ind = 0
for i, j in tag_count.most_common():
	tags_clus[ind] = [i, i]	
	ind += 1
print 'cvjhvbfws'

def clean_ques(query):
	query = query.lower()
	query = query.replace('\n', ' ')
	query = tokenizer.tokenize(query)
	#query = [q for q in query if q not in stop]
	return query

def wordvec(word):
	return model[word]

def rwmd(sent1, sent2):
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

	# dist2 is distance to move from sent2 to sent1	
	for i in range(len(sent2)):
		d = numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[0]))
		val = 0
		for j in range(len(sent1) - 1):
			if (numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[j + 1])) < d):
				d = numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[j + 1]))
				val = j + 1
		dist2 = dist2 + (1.0 / len(sent2)) * d	

	return max(dist1, dist2)			

def cluster():
	ind = 0
	for i in range(100, len(tags_clus)):
		dist = 0
		try:# try:
			dist = rwmd(clean_ques(tags_clus[0][0]), clean_ques(tags_clus[i][0]))
		except KeyError:
			pass
		tags_clus[i][1] = tags_clus[0][1]
		for j in range(100):
			try:
		 		dist1 = rwmd(clean_ques(tags_clus[j][0]), clean_ques(tags_clus[i][0]))
			except KeyError:
				continue
				# 	dist1 = 1000
			if dist1 < dist:
				dist = dist1
				tags_clus[i][1] = tags_clus[j][1]
	print (ind)		
	# print 'here', len(tags_clus)				
	for i in range(len(tags_clus)):
		print (tags_clus[i][0], tags_clus[i][1])

print 'here'
print len(tags_clus)
cluster()