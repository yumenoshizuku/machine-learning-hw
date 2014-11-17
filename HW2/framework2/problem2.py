import decision_tree as dt
import operator
import scan
import utils
import os
import sys


def main():
    binary_label = True
    exclude_stopwords = True
    data_nosw, data_positive_nosw, data_negative_nosw = (scan.scan('fineshort.txt', exclude_stopwords, binary_label))
    top_all_nosw = open(os.path.join(sys.path[0], "top_all_nosw.txt"), "wb")
    top_positive_nosw = open(os.path.join(sys.path[0], "top_positive_nosw.txt"), "wb")
    top_negative_nosw = open(os.path.join(sys.path[0], "top_negative_nosw.txt"), "wb")
    
    
    all_review_nosw = ' '.join([row[0] for row in data_nosw])
    dict_all_nosw = utils.get_unigram(all_review_nosw)[0]
    top_all_nosw.write('\n'.join('%s %s' % x for x in sorted(dict_all_nosw.items(), key=operator.itemgetter(1), reverse = True)))

    positive_review_nosw = ' '.join([row[0] for row in data_positive_nosw])
    dict_positive_nosw = utils.get_unigram(positive_review_nosw)[0]
    top_positive_nosw.write('\n'.join('%s %s' % x for x in sorted(dict_positive_nosw.items(), key=operator.itemgetter(1), reverse = True)))

    negative_review_nosw = ' '.join([row[0] for row in data_negative_nosw])
    dict_negative_nosw = utils.get_unigram(negative_review_nosw)[0]
    top_negative_nosw.write('\n'.join('%s %s' % x for x in sorted(dict_negative_nosw.items(), key=operator.itemgetter(1), reverse = True)))

    top_all_nosw.close()
    top_positive_nosw.close()
    top_negative_nosw.close()
    
    data_sw, data_positive_sw, data_negative_sw = scan.scan('fineshort.txt', not exclude_stopwords, binary_label)
    top_all_sw = open(os.path.join(sys.path[0], "top_all_sw.txt"), "wb")
    top_positive_sw = open(os.path.join(sys.path[0], "top_positive_sw.txt"), "wb")
    top_negative_sw = open(os.path.join(sys.path[0], "top_negative_sw.txt"), "wb")
   
    all_review_sw = ' '.join([row[0] for row in data_sw])
    dict_all_sw = utils.get_unigram(all_review_sw)[0]
    top_all_sw.write('\n'.join('%s %s' % x for x in sorted(dict_all_sw.items(), key=operator.itemgetter(1), reverse = True)))

    positive_review_sw = ' '.join([row[0] for row in data_positive_sw])
    dict_positive_sw = utils.get_unigram(positive_review_sw)[0]
    top_positive_sw.write('\n'.join('%s %s' % x for x in sorted(dict_positive_sw.items(), key=operator.itemgetter(1), reverse = True)))

    negative_review_sw = ' '.join([row[0] for row in data_negative_sw])
    dict_negative_sw = utils.get_unigram(negative_review_sw)[0]
    top_negative_sw.write('\n'.join('%s %s' % x for x in sorted(dict_negative_sw.items(), key=operator.itemgetter(1), reverse = True)))

    top_all_sw.close()
    top_positive_sw.close()
    top_negative_sw.close()
    
    length = len(data_nosw)
    train_data = data_nosw[:int(length*.8)]
    test_data = data_nosw[int(length*.8):]

    #decision_tree = dt.train(train_data)
    #test_results = dt.test(decision_tree, test_data)

    #print test_results

if __name__ == '__main__':
    main()
