#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from codecs import open
from random import shuffle
from argparse import ArgumentParser

recursive_layer_dim = 150
recursive_layer_num = 2
max_sequence_size = 100
batch_size = 100

#def load_data():
#
#    data_X = []
#    data_Y = []
#    
#    matrix_a = np.random.random(size=(5000, 1000, 50))
#    matrix_b = np.random.random(size=(5000, 1000, 50))
#    matrix_c = np.random.random(size=(5000, 1000, 50))
#    matrix_d = matrix_a + matrix_b + matrix_c
#    data_X.extend(matrix_d.tolist())
#    data_Y.extend([[1, 0]] * 5000)
#
#    matrix_a = np.random.random(size=(5000, 1000, 50))
#    matrix_a *= 2
#    data_X.extend(matrix_a.tolist())
#    data_Y.extend([[0, 1]] * 5000)
#
#    data_X = np.array(data_X)
#    data_Y = np.array(data_Y)
#    
#    print(data_X.shape, data_Y.shape)
#
#    index_list = range(10000)
#    shuffle(index_list)
#
#    train_index_array = np.array(index_list[:5000])
#    validate_index_array = np.array(index_list[5000:])
#
#    train_X = data_X[train_index_array]
#    train_Y = data_Y[train_index_array]
#    validate_X = data_X[validate_index_array]
#    validate_Y = data_Y[validate_index_array]
#
#    return train_X, train_Y, validate_X, validate_Y

#def load_data(word_index_dict, pos_data_file, neg_data_file):
#
#    pos_data_list = []
#    with open(pos_data_file, 'r') as fd:
#        for line in fd:
#            splited_line = line.strip().split()
#            word_index_list = []
#            for word in splited_line:
#                if word in word_index_dict:
#                    word_index_list.append(word_index_dict[word])
#                if len(word_index_list) >= max_sequence_size:
#                    break
#            if len(word_index_list) < max_sequence_size:
#                word_index_list.extend([0] * (max_sequence_size - len(word_index_list)))
#            assert len(word_index_list) == max_sequence_size
#            pos_data_list.append(word_index_list)
#    pos_target_list = [[0.0, 1.0]] * len(pos_data_list)
#        
#    neg_data_list = []
#    with open(neg_data_file, 'r') as fd:
#        for line in fd:
#            splited_line = line.strip().split()
#            word_index_list = []
#            for word in splited_line:
#                if word in word_index_dict:
#                    word_index_list.append(word_index_dict[word])
#                if len(word_index_list) >= max_sequence_size:
#                    break
#            if len(word_index_list) < max_sequence_size:
#                word_index_list.extend([0] * (max_sequence_size - len(word_index_list)))
#            assert len(word_index_list) == max_sequence_size
#            neg_data_list.append(word_index_list)
#    neg_target_list = [[1.0, 0.0]] * len(neg_data_list)
#
#    data_array = np.array(pos_data_list + neg_data_list, dtype = "int")
#    target_array = np.array(pos_target_list + neg_target_list, dtype = "float32")
#
#    index_list = range(len(pos_data_list) + len(neg_data_list))
#    shuffle(index_list)
#    train_index_array= np.array(index_list[:-500])
#    validate_index_array = np.array(index_list[-500:])
#    return (data_array[train_index_array, :], target_array[train_index_array, :],
#            data_array[validate_index_array, :], target_array[validate_index_array, :])

def shuffle_data(x_array, y_array):
    
    assert x_array.shape[0] == y_array.shape[0]
    index_list = range(x_array.shape[0])
    shuffle(index_list)
    index_array = np.array(index_list)
    x_array = x_array[index_array, :]
    y_array = y_array[index_array, :]
    return x_array, y_array
    

def load_data(word_index_dict, pos_data_file, neg_data_file):

    pos_data_list = []
    with open(pos_data_file, 'r') as fd:
        for line in fd:
            splited_line = line.strip().split()
            word_index_list = []
            for word in splited_line:
                if word in word_index_dict:
                    word_index_list.append(word_index_dict[word])
                if len(word_index_list) >= max_sequence_size:
                    break
            if len(word_index_list) < max_sequence_size:
                word_index_list.extend([0] * (max_sequence_size - len(word_index_list)))
            assert len(word_index_list) == max_sequence_size
            pos_data_list.append(word_index_list)
    pos_target_list = [[0.0, 1.0]] * len(pos_data_list)
    pos_data_array = np.array(pos_data_list, dtype = "int")
    pos_target_array = np.array(pos_target_list, dtype = "float32")
    pos_data_array, pos_target_array = shuffle_data(pos_data_array, pos_target_array)
    split_index = int(pos_data_array.shape[0] * 0.2)
    train_pos_data_array, validate_pos_data_array = pos_data_array[:-split_index, :], pos_data_array[-split_index:, :]
    train_pos_target_array, validate_pos_target_array = pos_target_array[:-split_index, :], pos_target_array[-split_index:, :]

    # expand pos data in training set multi times
    train_pos_data_array = np.vstack([train_pos_data_array] * 5)
    train_pos_target_array = np.vstack([train_pos_target_array] * 5)
        
    neg_data_list = []
    with open(neg_data_file, 'r') as fd:
        for line in fd:
            splited_line = line.strip().split()
            word_index_list = []
            for word in splited_line:
                if word in word_index_dict:
                    word_index_list.append(word_index_dict[word])
                if len(word_index_list) >= max_sequence_size:
                    break
            if len(word_index_list) < max_sequence_size:
                word_index_list.extend([0] * (max_sequence_size - len(word_index_list)))
            assert len(word_index_list) == max_sequence_size
            neg_data_list.append(word_index_list)
    neg_target_list = [[1.0, 0.0]] * len(neg_data_list)
    neg_data_array = np.array(neg_data_list, dtype = "int")
    neg_target_array = np.array(neg_target_list, dtype = "float32")
    neg_data_array, neg_target_array = shuffle_data(neg_data_array, neg_target_array)
    split_index = int(neg_data_array.shape[0] * 0.2)
    train_neg_data_array, validate_neg_data_array = neg_data_array[:-split_index, :], neg_data_array[-split_index:, :]
    train_neg_target_array, validate_neg_target_array = neg_target_array[:-split_index, :], neg_target_array[-split_index:, :]

    # expand pos data in training set multi times
    validate_pos_data_array = np.vstack([validate_pos_data_array] * 5)
    validate_pos_target_array = np.vstack([validate_pos_target_array] * 5)
        
    train_data_array = np.vstack((train_pos_data_array, train_neg_data_array))
    train_target_array = np.vstack((train_pos_target_array, train_neg_target_array))
    train_data_array, train_target_array = shuffle_data(train_data_array, train_target_array)
    validate_data_array = np.vstack((validate_pos_data_array, validate_neg_data_array))
    validate_target_array = np.vstack((validate_pos_target_array, validate_neg_target_array))
    validate_data_array, validate_target_array = shuffle_data(validate_data_array, validate_target_array)

    print train_data_array.shape, train_target_array.shape, validate_data_array.shape, validate_target_array.shape
    return train_data_array, train_target_array, validate_data_array, validate_target_array

def load_word_dict(tsv_file):

    word_index_dict = {}
    word_vector_list = []
    with open(tsv_file, "r") as fd:
        index = 0
        for line in fd:
            splited_line = line.strip().split("\t")
            if len(splited_line) != 2:
                continue
            word, vector = splited_line
            if word not in word_index_dict:
                word_index_dict[word] = index
                word_vector_list.append([float(value) for value in vector.split()])
                index += 1
        assert len(word_vector_list) == len(word_index_dict) == index
        word_vector_list[0] = [0.0] * 200
    return word_index_dict, np.array(word_vector_list, dtype = "float32")

def main():

    parser = ArgumentParser()
    parser.add_argument("word_dict_file", help = "word2vec file in tsv format")
    parser.add_argument("pos_data_file", help = "pos_data_file in tsv format")
    parser.add_argument("neg_data_file", help = "neg_data_file in tsv format")
    args = parser.parse_args()
    
    word_dict_file = args.word_dict_file
    pos_data_file = args.pos_data_file
    neg_data_file = args.neg_data_file

    word_index_dict, word_vector_array = load_word_dict(word_dict_file)
    train_X, train_Y, validate_X, validate_Y = load_data(word_index_dict, pos_data_file, neg_data_file)

    embedding_matrix_tsr = tf.constant(word_vector_array, dtype = tf.float32)

    input_plr = tf.placeholder(tf.int32, [None, max_sequence_size])
    target_plr = tf.placeholder(tf.float32, [None, 2])

    embedded_input_tsr = tf.nn.embedding_lookup(embedding_matrix_tsr, input_plr)
    dropout_tsr = tf.placeholder(tf.float32)
    learning_rate_tsr = tf.placeholder(tf.float32)

    cell = tf.nn.rnn_cell.GRUCell(recursive_layer_dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = dropout_tsr)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * recursive_layer_num)

    output_tsr, _ = tf.nn.dynamic_rnn(cell, embedded_input_tsr, dtype = tf.float32)
    output_tsr = tf.transpose(output_tsr, [1, 0, 2])
    output_tsr = tf.gather(output_tsr, int(output_tsr.get_shape()[0]) - 1)

    weight_var = tf.Variable(tf.truncated_normal([recursive_layer_dim, 2], stddev = 0.1))
    bias_var = tf.Variable(tf.constant(0.1, shape = [2]))

    proba_tsr = tf.nn.softmax(tf.matmul(output_tsr, weight_var) + bias_var)
    pred_tsr = tf.argmax(proba_tsr, 1)
    loss_tsr = -tf.reduce_sum(target_plr * tf.log(proba_tsr))
    error_tsr = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(target_plr, 1), pred_tsr), tf.float32))
    target_tsr = tf.argmax(target_plr, 1)

    #train_op = tf.train.RMSPropOptimizer(learning_rate_tsr).minimize(loss_tsr)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate_tsr).minimize(loss_tsr)
    train_op = tf.train.AdamOptimizer().minimize(loss_tsr)

    # train model
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    for epoch in xrange(50):
        partition_begin = 0
        partition_end = 0
        #learning_rate_arr = 0.0005
        while partition_end != len(train_X):
            partition_begin = partition_end
            if partition_end + batch_size > len(train_X):
                partition_end = len(train_X)
            else:
                partition_end += batch_size
            error_before, pred_before, target_arr = session.run([error_tsr, pred_tsr, target_tsr], feed_dict = {input_plr: train_X[partition_begin:partition_end], target_plr: train_Y[partition_begin:partition_end], dropout_tsr: 1.0})
            _, error_after, pred_after, target_arr = session.run([train_op, error_tsr, pred_tsr, target_tsr], feed_dict = {input_plr: train_X[partition_begin:partition_end], target_plr: train_Y[partition_begin:partition_end], dropout_tsr: 0.7})
            #print pred_before, pred_after, train_Y[partition_begin:partition_end], train_X[partition_begin:partition_end][1]
            #for index in train_X[partition_begin:partition_end][1]:
            #    for word in word_index_dict:
            #        if word_index_dict[word] == index:
            #            print word,
            #print
            #print error_before, error_after, session.run(error_tsr, feed_dict = {input_plr: validate_X, target_plr: validate_Y, dropout_tsr: 1.0}), pred_before[:10], pred_after[:10], train_Y[partition_begin:partition_end][1:10]
            print error_before, error_after, session.run(error_tsr, feed_dict = {input_plr: validate_X, target_plr: validate_Y, dropout_tsr: 1.0}), pred_before[:10], pred_after[:10], target_arr[:10]
        print "epoch:", epoch
    

if __name__ == "__main__":

    main()
