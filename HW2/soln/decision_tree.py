import utils
import operator

class DecisionTree:
    node_label = None  # takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
    node_attr = None   # test on an attribute, none means leaf
    left = None        # if the attribute is not present, go left
    right = None       # if the attribute is present, go right
    
    # initialize a node with optional attribute and label
    def __init__(self, attr = None, label = None):
        self.node_attr = attr
        self.node_label = label

    # recursive function that gets the leaf based on presence of features
    def go(self, data):
        if self.node_label != None:
            return self.node_label
        if self.node_attr in data:
            return self.right.go(data)
        else:
            return self.left.go(data)

# count the majority of what's left, when there is only one feature left
# or approached the deepest allowed level
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
     key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# default maximum level of tree is 20
def train(dataSet, allAttr, level = 20):
    # base case 1, no more data
    if len(dataSet) == 0:
	return None
    classList = [example[-1] for example in dataSet]
    # base case 2, all vectors have the same value
    if classList.count(classList[0]) == len(classList):
        return DecisionTree(None, classList[0])
    # base case 3, only one feature is left, or at max level
    if len(dataSet[0]) == 1 or level == 0:
        return DecisionTree(None, majorityCnt(classList))
    # find best feature to split 
    bestFeat = utils.chooseBestFeatureToSplit(dataSet, allAttr)
    # make it the current node attribute
    myTree = DecisionTree(bestFeat)
    # reduce dimension of allowed attributes
    newAllAttr = [x for x in allAttr if x!= bestFeat]
    # populate left and right children of the node by recursive call
    myTree.right = train(utils.splitDataSet(dataSet, bestFeat, True), newAllAttr, level - 1)
    myTree.left = train(utils.splitDataSet(dataSet, bestFeat, False), newAllAttr, level - 1)
    # return the current node
    return myTree

# compute what percentage of predicted values are correct
def test(tree, data):
    right = 0
    wrong = 0
    for i in data:
        if tree.go(i) == i[-1]:
            right += 1
        else:
            wrong += 1
    return right/float(right+wrong)

