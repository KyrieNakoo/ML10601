import sys
import numpy as np
import csv
import time
from numpy import *


def load_data(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for record in tsv_reader:
            tsv_data.append(record[1])
        # tsv_data = tsv_data
        return tsv_data


def load_label(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for record in tsv_reader:
            tsv_data.append(int(record[0]))
        # tsv_data = tsv_data
        return tsv_data


def load_dict(dic):
    feature_word = {}
    with open(dic, 'r') as f:
        reader = f.readlines()
        for line in reader:
            each = line.split()
            feature_word[each[0]] = each[1]
        # tsv_data = tsv_data
        return feature_word


def load_dict2(dic2):
    word_vec = {}
    with open(dic2, 'r') as f:
        reader = f.readlines()
        for line in reader:
            each = line.split()
            word_vec[each[0]] = list(map(float, each[1:]))
            length = len(word_vec[each[0]])
        return word_vec, length


def split_word(data):
    word = []
    for i in range(len(data)):
        word.append(data[i].split())
    return word


def model_one(review, dictionary):
    feature = np.zeros([len(review), len(dictionary)])
    for i in range(len(review)):
        for word in review[i]:
            if word in dictionary:
                feature[i, int(dictionary[word])] = 1
    return feature


def model_two(review, wordvec, length):
    feature = [[0]*length for i in range(len(review))]
    for i in range(len(review)):
        a = 0
        for word in review[i]:
            if word in wordvec:
                a += 1
                feature[i] = np.sum([feature[i], wordvec[word]], axis=0)
        feature[i] = [x/a for x in feature[i]]
    feature_matrix = mat(feature)
    return feature_matrix


def outFile(data, f_name):
    filename = str(f_name)
    np.savetxt(filename, data, fmt='%d', delimiter='\t')


def outFile_float(data, f_name):
    filename = str(f_name)
    np.savetxt(filename, data, fmt='%1.6f', delimiter='\t')


def output(inputdata, flag, dict1, dict2, f_name):
    flag = int(flag)
    data_input = load_data(inputdata)
    data_label = load_label(inputdata)
    word = split_word(data_input)
    # print(word[1])
    if flag == 1:
        dic1 = load_dict(dict1)
        # print(dict1[1])
        feature1 = model_one(word, dic1)
        # print(feature1[1])
        feature_vector1 = np.c_[data_label, feature1]
        outFile(feature_vector1, f_name)
    elif flag == 2:
        dic2, length = load_dict2(dict2)
        # print(dic2['film'])
        feature2 = model_two(word, dic2, length)
        # print(feature2[1])
        feature_vector2 = np.c_[data_label, feature2]
        outFile_float(feature_vector2, f_name)


if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict1_input = sys.argv[4]
    f_train_out = sys.argv[5]
    f_validation_out = sys.argv[6]
    f_test_out = sys.argv[7]
    feature_flag = sys.argv[8]
    dict2_input = sys.argv[9]

    st = time.time()
    # data input
    # data_train = load_data(train_input)
    # data_valid = load_data(validation_input)
    # data_test = load_data(test_input)
    #
    # data_train_label = load_label(train_input)  # label of review
    # train_word = split_word(data_train)   # separated word list
    # valid_word = split_word(data_valid)
    # test_word = split_word(data_test)
    #
    # if feature_flag == 1:
    #     dict_one = load_dict(dict1_input)
    #     feature_one_train = model_one(train_word, dict_one)
    # #     feature_one_valid = model_one(valid_word, dict_one)
    # #     feature_one_test = model_one(test_word, dict_one)
    #     feature_vector_one = np.c_[data_train_label, feature_one_train]
    #     outFile(feature_vector_one, f_train_out)
    # #
    # #
    # if feature_flag == 2:
    # dict_two, vector_len = load_dict2(dict2_input)
    # feature_two = model_two(train_word, dict_two, vector_len)
    # print(feature_two.shape, feature_two[1:3, 1:3])
    # feature_vector_two = np.c_[data_train_label, feature_two]
    # print(feature_vector_two.shape, feature_vector_two[0:3, 1:5])
    #     outFile_float(feature_vector_two, f_train_out)



    # outFile(feature_two, f_train_out)
    output(train_input, feature_flag, dict1_input, dict2_input, f_train_out)
    output(validation_input, feature_flag, dict1_input, dict2_input, f_validation_out)
    output(test_input, feature_flag, dict1_input, dict2_input, f_test_out)
    print(time.time()-st)


