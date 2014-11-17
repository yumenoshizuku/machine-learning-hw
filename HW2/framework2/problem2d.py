import decision_tree as dt
import operator
import scan
import utils
import os
import sys


def main():
    binary_label = True
    exclude_stopwords = True
    data_nosw, data_positive_nosw, data_negative_nosw = (scan.scan('finemedium.txt', exclude_stopwords, binary_label))
    data = []
    for datum in data_nosw:
        new_datum = datum[0].split()
        new_datum.append(datum[1])
        data.append(new_datum)
    #print data
    positive_review_nosw = ' '.join([row[0] for row in data_positive_nosw])
    dict_positive_nosw = utils.get_unigram(positive_review_nosw)[0]
    positive_words = [x[0] for x in sorted(dict_positive_nosw.items(), key=operator.itemgetter(1), reverse = True)[1:501]]

    negative_review_nosw = ' '.join([row[0] for row in data_negative_nosw])
    dict_negative_nosw = utils.get_unigram(negative_review_nosw)[0]
    negative_words = [x[0] for x in sorted(dict_negative_nosw.items(), key=operator.itemgetter(1), reverse = True)[1:501]]
    
    all_words = positive_words
    all_words.extend(x for x in negative_words if x not in positive_words)

    length = len(data)
    train_data = data[:int(length*.8)]
    test_data = data[int(length*.8):]
    
    decision_tree = dt.train(train_data, all_words)
    test_results = dt.test(decision_tree, test_data)
    print test_results

    #decision_tree = dt.train(train_data)
    #test_results = dt.test(decision_tree, test_data)

    #print test_results

if __name__ == '__main__':
    main()
