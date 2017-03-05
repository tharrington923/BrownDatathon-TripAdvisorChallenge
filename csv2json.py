import csv
import json

with open('class1.csv', 'r') as csvfile:
    nextline = csv.reader(csvfile, delimiter=' ')
    class1 = []
    for x in nextline:
        yList = []
        for y in x[0].split(','):
            yList.append(float(y))
        class1.append(yList)

with open('class0.csv', 'r') as csvfile:
    nextline = csv.reader(csvfile, delimiter=' ')
    class0 = []
    for x in nextline:
        yList = []
        for y in x[0].split(','):
            yList.append(float(y))
        class0.append(yList)

class1out = []
for x in class1:
    out = {"f0": x[0],
        "f1": x[1],
        "f2": x[2],
        "f3": x[3],
        "f4": x[4],
        "f5": x[5],
        "f6": x[6],
        "f7": x[7],
        "f8": x[8],
        "f9": x[9],
        "f10": x[10],
        "f11": x[11],
        "f12": x[12]}
    class1out.append(out)

class0out = []
for x in class0:
    out = {"f0": x[0],
        "f1": x[1],
        "f2": x[2],
        "f3": x[3],
        "f4": x[4],
        "f5": x[5],
        "f6": x[6],
        "f7": x[7],
        "f8": x[8],
        "f9": x[9],
        "f10": x[10],
        "f11": x[11],
        "f12": x[12]}
    class0out.append(out)

with open('class1.js', 'w') as f:
    json.dump(class1out, f, indent = 4)

with open('class0.js', 'w') as f:
    json.dump(class0out, f, indent = 4)
