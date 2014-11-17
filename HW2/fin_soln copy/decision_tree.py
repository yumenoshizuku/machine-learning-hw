import utils
import operator

class DecisionTree:
    node_label = None  # takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
    node_attr = None
    left = None
    right = None
    
    def __init__(self, attr = None, label = None):
        self.node_attr = attr
        self.node_label = label

    def go(self, data):
        if self.node_label != None:
            return self.node_label
        if self.node_attr in data:
            return self.right.goo(data)
        else:
            return self.left.goo(data)


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
        return DecisionTree(None, classList[0])
    if len(dataSet[0]) == 1 or level == 0:
        return DecisionTree(None, majorityCnt(classList))
    bestFeat = utils.chooseBestFeatureToSplit(dataSet, allAttr)
    myTree = DecisionTree(bestFeat)
    newAllAttr = [x for x in allAttr if x!= bestFeat]
    myTree.right = train(utils.splitDataSet(dataSet, bestFeat, True), newAllAttr, level - 1)
    myTree.left = train(utils.splitDataSet(dataSet, bestFeat, False), newAllAttr, level - 1)
    return myTree

def test(tree, data):
    right = 0
    wrong = 0
    for i in data:
        if tree.go(i) == i[-1]:
            right += 1
        else:
            wrong += 1
    return right/float(right+wrong)

