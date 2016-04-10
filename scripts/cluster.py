labels = []
with open("xaa") as f:
    data = json.load(f)    

for line, value in data.items():
    labels.append(value)
f.close()
print (len(labels))