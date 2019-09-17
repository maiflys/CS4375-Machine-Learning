from DecisionTree import *
import pandas as pd
from sklearn.model_selection import *

import numpy as np
import matplotlib.pyplot as plt

header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)
nodes, leafNodes = getNodeList(t)
print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))


#print(nodes)
#print(leafNodes)
test = lst[0]
lab = classify(test, t)
test = lst[0:10]
print("Accuracy = " + str(computeAccuracy(test, t)))

#print(lab)
# print(t.question)
# random selection:

#trainDF = df.sample(frac=0.50, random_state=99)
#testDF = df.loc[~df.index.isin(trainDF.index), :]

#train = trainDF.values.tolist()
#test = testDF.values.tolist()
trainDF, testDF = train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()
print(train)


t = build_tree(train, header)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

