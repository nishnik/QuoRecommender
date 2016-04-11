import json, collections

with open('clean.json') as data_file:
    a = json.load(data_file)# type(data)=dict

c = []
total = 0
for i in a:
	for j in a[i]:
		c.append(j)
	total = total + len(a[i])
counts = collections.Counter(c)



n = 5000
for word, count in counts.most_common(n):
    print("{0}".format(word))