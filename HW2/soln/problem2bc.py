import decision_tree as dt
import operator
import scan
import utils
import os
import sys

# write result of problem 2 (b) and (c) to files
def main():
    # defaults are binary outcome and exclude stopwords
    binary_label = True
    exclude_stopwords = True
    
    # a modified scan that returns three lists
    data_nosw, data_positive_nosw, data_negative_nosw = (scan.scan('finefoods.txt', exclude_stopwords, binary_label))
    
    # open files for writing
    top_all_nosw = open(os.path.join(sys.path[0], "top_all_nosw.txt"), "wb")
    top_positive_nosw = open(os.path.join(sys.path[0], "top_positive_nosw.txt"), "wb")
    top_negative_nosw = open(os.path.join(sys.path[0], "top_negative_nosw.txt"), "wb")
    
    # join all reviews for wordcount
    all_review_nosw = ' '.join([row[0] for row in data_nosw])
    dict_all_nosw = utils.get_unigram(all_review_nosw)[0]
    # sort by most frequent words and write to file
    top_all_nosw.write('\n'.join('%s %s' % x for x in sorted(dict_all_nosw.items(), key=operator.itemgetter(1), reverse = True)))

    # join all positive reviews for wordcount
    positive_review_nosw = ' '.join([row[0] for row in data_positive_nosw])
    dict_positive_nosw = utils.get_unigram(positive_review_nosw)[0]
    top_positive_nosw.write('\n'.join('%s %s' % x for x in sorted(dict_positive_nosw.items(), key=operator.itemgetter(1), reverse = True)))

    # join all negative reviews for wordcount
    negative_review_nosw = ' '.join([row[0] for row in data_negative_nosw])
    dict_negative_nosw = utils.get_unigram(negative_review_nosw)[0]
    top_negative_nosw.write('\n'.join('%s %s' % x for x in sorted(dict_negative_nosw.items(), key=operator.itemgetter(1), reverse = True)))

    # close files
    top_all_nosw.close()
    top_positive_nosw.close()
    top_negative_nosw.close()
    
    # same set of routines, but this time allowing stopwords
    data_sw, data_positive_sw, data_negative_sw = scan.scan('finefoods.txt', not exclude_stopwords, binary_label)
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

if __name__ == '__main__':
    main()
