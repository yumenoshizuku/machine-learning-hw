import scan
import utils
import operator

# another version of decision tree, utilizing dictionaries instead of binary tree
# for better recursion performance, but both give identical results
# adapted from Machine Learning in Action, Peter Harrington

# see comment in decision_tree.py
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
     key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def train(dataSet, allAttr, level = 20):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1 or level == 0:
        return majorityCnt(classList)
    bestFeat = utils.chooseBestFeatureToSplit(dataSet, allAttr)
    myTree = {bestFeat:{}}
    featValues = [(bestFeat in example) for example in dataSet]
    uniqueVals = set(featValues)
    newAllAttr = [x for x in allAttr if x!= bestFeat]
    for value in uniqueVals:
        myTree[bestFeat][value] = train(utils.splitDataSet(dataSet, bestFeat, value), newAllAttr, level - 1)
    return myTree

# predict value from vector using the dictionary structure
def classify(inputTree, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    key = firstStr in testVec
    if type(secondDict[key]).__name__=='dict':
        classLabel = classify(secondDict[key], testVec)
    else:
        classLabel = secondDict[key]
    return classLabel

def test(tree, data):
    right = 0
    wrong = 0
    for i in data:
        if classify(tree, i) == i[-1]:
            right += 1
        else:
            wrong += 1
    return right/float(right+wrong)

