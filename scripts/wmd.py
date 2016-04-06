import numpy

#Get the Word Centroid Distance
def wcd(sent1, sent2):
	s1 = [0 for i in range(300)]
	s2 = [0 for i in range(300)]
	
	for i in range(len(sent1)):
		s1 = s1 + wordvec(sent1[i])
	for i in range(len(sent2)):
		s2 = s2 + wordvec(sent2[i])
	
	s1 = s1 / len(sent1)
	s2 = s2 / len(sent2)	
	return numpy.linalg.norm(s1 - s2)		

#Get the Relaxed Word Mover Distance
def rwmd(sent1, sent2):
	s1, s2 = 0, 0
	dist1 , dist2 = 0, 0
	# dist1 is distance to move from sent1 to sent2
	for i in range(len(sent1)):
		d = numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[0]))
		val = 0
		for j in range(len(sent2) - 1):
			if (numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[j + 1])) < d):
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




