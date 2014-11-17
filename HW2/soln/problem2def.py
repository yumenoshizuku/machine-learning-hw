import decision_tree as dt
import decision_tree_dict as dtd
import operator
import scan
import utils


def main():
    # default is binary outcome and no stopwords
    binary_label = True
    exclude_stopwords = True
    data_nosw, data_positive_nosw, data_negative_nosw = (scan.scan('finefoods.txt', exclude_stopwords, binary_label))
    
    # format data into 2 dimensional array
    data = []
    for datum in data_nosw:
        # first part of array is each review with words splitted
        new_datum = datum[0].split()
        # last item in each array is the label
        new_datum.append(datum[1])
        data.append(new_datum)
    
    # get a list of 500 most frequent positive words, ignoring the <br>
    positive_review_nosw = ' '.join([row[0] for row in data_positive_nosw])
    dict_positive_nosw = utils.get_unigram(positive_review_nosw)[0]
    positive_words = [x[0] for x in sorted(dict_positive_nosw.items(), key=operator.itemgetter(1), reverse = True)[1:501]]

    # get a list of 500 most frequent negative words, ignoring the <br>
    negative_review_nosw = ' '.join([row[0] for row in data_negative_nosw])
    dict_negative_nosw = utils.get_unigram(negative_review_nosw)[0]
    negative_words = [x[0] for x in sorted(dict_negative_nosw.items(), key=operator.itemgetter(1), reverse = True)[1:501]]
    
    # create non duplicate list of all frequent words from the two lists
    all_words = positive_words
    all_words.extend(x for x in negative_words if x not in positive_words)

    # split training and testing data
    length = len(data)
    train_data = data[:int(length*.8)]
    test_data = data[int(length*.8):]
    
    # using a dicision tree utilizing dictionaries
    decision_tree_dict = dtd.train(train_data, all_words)
    test_results_dict = dtd.test(decision_tree_dict, test_data)
    print test_results_dict

    # the same disicion tree utilizing binary tree
    decision_tree = dt.train(train_data, all_words)
    test_results = dt.test(decision_tree, test_data)
    print test_results
    
if __name__ == '__main__':
    main()
