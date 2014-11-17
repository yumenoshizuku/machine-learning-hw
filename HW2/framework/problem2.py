import decision_tree as dt
import operator
import scan
import utils


def main():
    binary_label = True
    exclude_stopwords = False
    data = scan.scan('fineshort.txt', exclude_stopwords, binary_label)
    length = len(data)
    all_review = ' '.join([row[0] for row in data])
    dict_all = utils.get_unigram(all_review)[0]
    print(sorted(dict_all.items(), key=operator.itemgetter(1)))
    train_data = data[:int(length*.8)]
    test_data = data[int(length*.8):]

    #decision_tree = dt.train(train_data)
    #test_results = dt.test(decision_tree, test_data)

    #print test_results

if __name__ == '__main__':
    main()
