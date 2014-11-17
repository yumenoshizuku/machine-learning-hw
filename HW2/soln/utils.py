from math import log

# decision tree stuff

# helper function to find the option that gives highest information gain
def chooseBestFeatureToSplit(dataSet, allAttr):
    baseEntropy = entropy(dataSet)
    bestInfoGain = 0.0;
    bestFeature = ''
    for i in allAttr:
        infoGain = information_gain(dataSet, i, baseEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# compute information gain for binary option
def information_gain(dataSet, attr, baseEntropy):
    newEntropy = 0.0
    subDataSetHas = splitDataSet(dataSet, attr, True)
    subDataSetNo = splitDataSet(dataSet, attr, False)
    probHas = len(subDataSetHas)/float(len(dataSet))
    probNo = len(subDataSetNo)/float(len(dataSet))
    newEntropy = probHas * entropy(subDataSetHas) + probNo * entropy(subDataSetNo)
    return (baseEntropy - newEntropy)

# shannon entropy
def entropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# separate dataset based on presence of an attribute
def splitDataSet(dataSet, attr, value):
    retDataSet = []
    if value == True:
        for featVec in dataSet:
            if attr in featVec:
                reducedFeatVec = [v for v in featVec if v != attr]
                retDataSet.append(reducedFeatVec)
    else:
        for featVec in dataSet:
            if attr not in featVec:
                retDataSet.append(featVec)
    return retDataSet

# natural language processing stuff
def freq(lst):
    freq = {}
    length = len(lst)
    for ele in lst:
        if ele not in freq:
            freq[ele] = 0
        freq[ele] += 1
    return (freq, length)

def get_unigram(review):
    return freq(review.split())

def get_unigram_list(review):
    return get_unigram(review)[0].keys()
