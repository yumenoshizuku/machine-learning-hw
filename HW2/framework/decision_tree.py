import scan
import utils

class DecisionTree:
    node_label = None  # takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
    left = None
    right = None

    def decision(self, data):
        raise 'not defined'

    def go(self, data):
        if self.node_label != None:
            return self.node_label
        return self.go(self.decision(data))

# http://en.wikipedia.org/wiki/ID3_algorithm
def train(data):
    raise 'not implemented'

def test(data):
    raise 'not implemented'

