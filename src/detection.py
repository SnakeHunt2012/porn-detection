#!/usr/bin/env python
# -*- coding: utf-8 -*-

from json import loads, dumps
from pickle import load, dump
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import OneClassSVM
from numpy import array

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
        elif tag == 0 and pred == -1:
            true_negative_count += 1
        elif tag == 1 and pred == -1:
            false_negative_count += 1
        else:
            print tag, pred
            assert False

    precision_positive = float(true_positive_count) / (true_positive_count + false_positive_count)
    recall_positive = float(true_positive_count) / (true_positive_count + false_negative_count)
    precision_negative = float(true_negative_count) / (true_negative_count + false_negative_count)
    recall_negative = float(true_negative_count) / (true_negative_count + false_positive_count)
    
    return precision_positive, recall_positive, precision_negative, recall_negative

def main():

    parser = ArgumentParser()
    parser.add_argument("train_file", help = "data file in pickle format {'url_list': [], 'label_list': [], 'feature_matrix': coo_matrix} (input)")
    parser.add_argument("validate_file", help = "data file in pickle format {'url_list': [], 'label_list': [], 'feature_matrix': coo_matrix} (input)")
    parser.add_argument("score_path", help = "path to dump score in json format (output)")
    parser.add_argument("model_path", help = "path to dump model in pickle format (output)")
    args = parser.parse_args()

    train_file = args.train_file
    validate_file = args.validate_file
    score_path = args.score_path
    model_path = args.model_path
    
    with open(train_file, "rb") as fd:
        data_dict_train = load(fd)
    url_train = data_dict_train["url_list"]
    y_train = data_dict_train["label_list"]
    X_train = data_dict_train["feature_matrix"]
    assert len(url_train) == len(y_train) == X_train.shape[0]

    with open(validate_file, "rb") as fd:
        data_dict_validate = load(fd)
    url_validate = data_dict_validate["url_list"]
    y_validate = data_dict_validate["label_list"]
    X_validate = data_dict_validate["feature_matrix"]
    assert len(url_validate) == len(y_validate) == X_validate.shape[0]

    classifier = OneClassSVM(kernel='rbf',
                             degree=3,
                             gamma='auto',
                             coef0=0.0,
                             tol=0.001,
                             nu=0.06,
                             shrinking=True,
                             cache_size=200,
                             verbose=False,
                             max_iter=-1,
                             random_state=None)
    
    print "training ..."
    classifier.fit(X_train)
    print "training done"

    pred_train = classifier.predict(X_train.toarray())
    pred_validate = classifier.predict(X_validate.toarray())

    acc_train = accuracy_score(y_train, pred_train)
    acc_validate = accuracy_score(y_validate, pred_validate)

    precision_positive, recall_positive, precision_negative, recall_negative = read_score(y_validate, pred_validate.tolist())

    score_dict = {
        "url_train": url_train,
        "y_train": y_train,
        "pred_train": pred_train.tolist(),
        "url_validate": url_validate,
        "y_validate": y_validate,
        "pred_validate": pred_validate.tolist(),
        "acc_train": acc_train,
        "acc_validate": acc_validate,
        "precision_positive": precision_positive,
        "recall_positive": recall_positive,
        "precision_negative": precision_negative,
        "recall_negative": recall_negative,
        #"feature_importance": classifier.feature_importances_.tolist()
    }
    print "dumping socre ..."
    with open(score_path, 'w') as fd:
        fd.write(dumps(score_dict, indent = 4))
    print "dumping socre done"

    print "dumping model ..."
    with open(model_path, "wb") as fd:
        dump(classifier, fd)
    print "dumping model done"


if __name__ == "__main__":

    main()
