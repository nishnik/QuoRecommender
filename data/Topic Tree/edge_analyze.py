# This tells you about the number of in-edge and out-edge
import json
with open("TopicHierarchyEdgeList.txt") as w:
	a = w.readlines()
first_e = {}
second_e = {}
for i in a:
	c = 0
	i = i.replace('\t',' ')
	c = i.find(' ')
	a = i[:c]
	b = i[c:]
	if a in first_e.keys():
		first_e[a] = first_e[a] + 1
	else:
		first_e[a] = 1

	if b in second_e.keys():
		second_e[a] = second_e[a] + 1
	else:
		second_e[a] = 1


with open('first_e.json','w') as outfile:
    json.dump(first_e, outfile, indent=2, separators=(',', ': '))


with open('second_e.json','w') as outfile:
    json.dump(second_e, outfile, indent=2, separators=(',', ': '))