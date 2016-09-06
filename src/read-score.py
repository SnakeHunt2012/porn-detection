#!/usr/bin/env python
# -*- coding: utf-8

from json import load, dump
from numpy import array, where
from argparse import ArgumentParser

def load_template_dict(template_file):

    with open(template_file, 'r') as fd:
        template_dict = load(fd)
    word_idf_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_idf_dict"].iteritems())
    word_index_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_index_dict"].iteritems())
    index_word_dict = dict((value, key.encode("utf-8")) for key, value in template_dict["word_index_dict"].iteritems())
    assert len(word_idf_dict) == len(word_index_dict)
    return word_idf_dict, word_index_dict, index_word_dict

def read_score(y_list, pred_list):
    
    # positive: pornographic
    # negative: normal
    true_positive_count = 0
    false_positive_count = 0
    true_negative_count = 0
    false_negative_count = 0

    positive_count = 0
    negative_count = 0
    assert len(y_list) == len(pred_list)
    for tag, pred in zip(y_list, pred_list):
        if tag == 1 and pred == 1:
            true_positive_count += 1
        elif tag == 0 and pred == 1:
            false_positive_count += 1
        elif tag == 0 and pred == 0:
            true_negative_count += 1
        elif tag == 1 and pred == 0:
            false_negative_count += 1
        else:
            assert False

    return true_positive_count, false_positive_count, true_negative_count, false_negative_count

def make_template(feature_list):

    word_idf_dict = dict((word, idf) for score, word, idf in feature_list)
    index_word_dict = dict((index, feature_list[index][1]) for index in xrange(len(feature_list)))
    word_index_dict = dict((feature_list[index][1], index) for index in xrange(len(feature_list)))
    with open("callback-template.json", 'w') as fd:
        dump({"word_idf_dict": word_idf_dict,
              "index_word_dict": index_word_dict,
              "word_index_dict": word_index_dict},
             fd, indent=4, ensure_ascii=False)

def main():

    parser = ArgumentParser()
    parser.add_argument("score_file", help = "score file in json format")
    parser.add_argument("template_file", help = "feature template file in json format")
    args = parser.parse_args()

    score_file = args.score_file
    template_file = args.template_file

    with open(score_file, 'r') as fd:
        score_dict = load(fd)

    word_idf_dict, word_index_dict, index_word_dict = load_template_dict(template_file)

    true_positive_count, false_positive_count, true_negative_count, false_negative_count = read_score(score_dict["y_validate"], score_dict["pred_validate"])
    precision_positive = float(true_positive_count) / (true_positive_count + false_positive_count)
    recall_positive = float(true_positive_count) / (true_positive_count + false_negative_count)
    precision_negative = float(true_negative_count) / (true_negative_count + false_negative_count)
    recall_negative = float(true_negative_count) / (true_negative_count + false_positive_count)
    url_validate = score_dict["url_validate"]
    y_validate = score_dict["y_validate"]
    pred_validate = score_dict["pred_validate"]
    assert len(url_validate) == len(y_validate) == len(pred_validate)
    for index in xrange(len(url_validate)):
        if y_validate[index] != pred_validate[index]:
            print url_validate[index], y_validate[index], pred_validate[index]
    print true_negative_count, false_negative_count, true_positive_count, false_positive_count
    print precision_positive, recall_positive, precision_negative, recall_negative

    feature_importance = array(score_dict["feature_importance"])
    sorted_list = []
    for feature_index in where(feature_importance)[0]:
        feature_score = feature_importance[feature_index]
        feature_name = index_word_dict[feature_index]
        feature_idf = word_idf_dict[feature_name]
        sorted_list.append((feature_score, feature_name, feature_idf))
    sorted_list.sort(reverse=True)
    #for item in sorted_list:
    #    feature_score, feature_name, feature_idf = item
    #    print feature_name, feature_score, feature_idf
    
    #make_template(sorted_list)

if __name__ == "__main__":

    main()

