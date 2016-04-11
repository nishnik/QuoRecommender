# This was used to clean the original json file which might have questions like:
# "Startups: What are some innovative business models for social good internet startups?"
# It removes the "Strartups: " from the question


import json

with open("text1.json") as o:
    a = json.load(o)

data ={}
for i in a:
    ii = ""
    flag = False
    for j in a[i]:
            if str(j+":") in i:
                    ii = i[i.find(j)+len(j)+1:].strip()
                    flag = True
                    break
    if flag == False:
        ii = i
    data[ii] = list(set(a[i]))

with open('clean.json','w') as outfile:
    json.dump(data, outfile, indent=2, separators=(',', ': '))