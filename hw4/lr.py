import sys
import numpy as np
import math
import csv
import time
import scipy.linalg as la
from numpy import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def load_data(data):
    tsv_data = []
    with open(data, 'r') as f:
        reader = f.readlines()
        for line in reader:
            each = line.split()
            record = list(map(float, each[1:]))
            tsv_data.append(record)
        word_data = mat(tsv_data)
    return word_data


def load_label(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for record in tsv_reader:
            tsv_data.append(float(record[0]))
        label_data = mat(tsv_data)
        return label_data


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


def SGD(alpha, theta, label, word_matrix, entry_index):
    a = float(alpha)
    # print(a)
    xi = word_matrix[:, entry_index]
    N = len(word_matrix.T)
    # print(theta.T@xi)
    theta = a * xi/N*(label[entry_index] - math.exp(dot(theta.T, xi))/(1+math.exp(dot(theta.T, xi))))
    # print(float(llh))
    return theta


def likehood(theta, label, word_matrix):
    llh = 0
    for i in range(len(word_matrix.T)):
        llh += (- label[i]*dot(theta.T, word_matrix[:, i])+math.log(1+math.exp(dot(theta.T, word_matrix[:, i]))))
    return float(llh)


def update(epoch, word_matrix_train, word_matrix_valid, parameter, label_train, label_valid):
    N_train = len(word_matrix_train.T)
    N_valid = len(word_matrix_valid.T)
    ave_lh_train = np.zeros(epoch)
    # ave_lh_train_1 = np.zeros(epoch)
    # ave_lh_train_001 = np.zeros(epoch)
    ave_lh_valid = np.zeros(epoch)
    for i in range(epoch):
        print(i)
        for j in range(len(word_matrix_train.T)):
            parameter += SGD(0.01, parameter, label_train, word_matrix_train, j)
            # parameter_1 += SGD(0.1, parameter_1, label_train, word_matrix_train, j)
            # parameter_001 += SGD(0.001, parameter_001, label_train, word_matrix_train, j)
        ave_lh_train[i] = likehood(parameter, label_train, word_matrix_train)/N_train
        # ave_lh_train_1[i] = likehood(parameter_1, label_train, word_matrix_train)/ N_train
        # ave_lh_train_001[i] = likehood(parameter_001, label_train, word_matrix_train)/ N_train
        ave_lh_valid[i] = likehood(parameter, label_valid, word_matrix_valid)/N_valid

    return parameter, ave_lh_train, ave_lh_valid


def predict(theta, word_matrix):
    pre_prob = []
    pre_label = []
    for i in range(len(word_matrix.T)):
        b = math.exp(dot(theta.T, word_matrix[:, i]))
        pre_prob.append(b/(1+b))
    for j in range(len(pre_prob)):
        if pre_prob[j] >= 0.5:
            pre_label.append(1)
        else:
            pre_label.append(0)
    return mat(pre_label).T


def fileOut(pre_label_train, pre_label_test, true_label_train, true_label_test, f_train, f_test, f_error):
    train_rate = (len(true_label_train) - sum(pre_label_train == true_label_train))/len(true_label_train)
    test_rate = (len(true_label_test) - sum(pre_label_test == true_label_test))/len(true_label_test)

    f_error = str(f_error)
    f = open(f_error, "w")
    f.write("error(train): ")
    f.write(str(train_rate))
    f.write('\n')
    f.write("error(test): ")
    f.write(str(test_rate))
    f.write('\n')
    f.close()

    f_train = str(f_train)
    f1 = open(f_train, "w")
    for i in range(len(pre_label_train)):
        f1.write(str(pre_label_train[i, 0]))
        f1.write('\n')
    f1.close()

    f_test = str(f_test)
    f1 = open(f_test, "w")
    for j in range(len(pre_label_test)):
        f1.write(str(pre_label_test[j, 0]))
        f1.write('\n')
    f1.close()


def filelike(data, name):
    f_data = str(name)
    f1 = open(f_data, "w")
    for i in range(len(data)):
        f1.write(str(data[i]))
        f1.write('\n')
    f1.close()

# def getTheta(input_train, epoch):
#     data = load_data(input_train)
#     label_input = load_label(input_train).T
#
#     dat = np.insert(data, 0, values=np.ones(len(data)), axis=1)
#     dat_bia = dat.T
#     ini_theta = np.zeros(len(dat_bia)).reshape(len(dat_bia), 1)
#     num_epoch = int(epoch)
#     parameter = update(num_epoch, dat_bia, ini_theta, label_input)
#     pre_label = predict(parameter, dat_bia)
#     return pre_label, label_input





if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    f_train_out = sys.argv[5]
    f_test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = sys.argv[8]

    st = time.time()
    # data_train = load_data(train_input)       # data without label
    # data_test = load_data(test_input)
    # data_valid = load_data(validation_input)
    # # print(data_train.shape, data_test.shape)   #(1200, 14164) (400, 14164)
    #
    # label_train = load_label(train_input).T                     # label_ture
    # label_test = load_label(test_input).T
    # label_valid = load_label(validation_input).T
    # # print(label_train.shape, label_test.shape)   #(1200, 1) (400, 1)
    #
    # dat_train = np.insert(data_train, 0, values=np.ones(len(data_train)), axis=1)
    # dat_test = np.insert(data_test, 0, values=np.ones(len(data_test)), axis=1)
    # dat_valid = np.insert(data_valid, 0, values=np.ones(len(data_valid)), axis=1)
    # dat_bia_train = dat_train.T
    # dat_bia_test = dat_test.T
    # dat_bia_valid = dat_valid.T
    #
    # ini_theta1 = np.zeros(len(dat_bia_train)).reshape(len(dat_bia_train), 1)
    # # ini_theta2 = np.zeros(len(dat_bia_train)).reshape(len(dat_bia_train), 1)
    # # ini_theta3 = np.zeros(len(dat_bia_train)).reshape(len(dat_bia_train), 1)
    num_epoch = int(num_epoch)
    # parameter, lh_train, lh_valid = update(num_epoch, dat_bia_train, dat_bia_valid,
    #            ini_theta1, label_train, label_valid)
    #
    # filelike(lh_train, 'likehood_trian_2.tsv')
    # # filelike(lh_1, 'likehood_trian_a1.tsv')
    # # filelike(lh_001, 'likehood_trian_a001.tsv')
    # # filelike(lh_valid, 'likehood_valid.tsv')
    # filelike(lh_valid, 'likehood_valid_2.tsv')

    data_lh_train = load_label('likehood_trian_2.tsv')
    data_lh_valid = load_label('likehood_valid_2.tsv')

    # data_lh_valid = load_label('likehood_valid_2.tsv')
    # SGD(0.001, ini_theta, label_train, dat_bia_train, 2)

    # data_lh1_train = load_label('likehood_trian_a1.tsv')
    # data_lh001_train = load_label('likehood_trian_a001.tsv')

    plt.figure(figsize=(10, 6))
    x = np.arange(0, 5000, 1)
    xs = np.linspace(0, num_epoch, 5000)
    y1 = data_lh_train.T.tolist()
    # y2 = data_lh_valid.T.tolist()
    y2 = data_lh_valid.T.tolist()
    # y3 = data_lh001_train.T.tolist()
    model1 = make_interp_spline(x, y1)
    model2 = make_interp_spline(x, y2)
    # model3 = make_interp_spline(x, y3)

    ys1 = model1(xs)
    ys2 = model2(xs)
    # ys3 = model3(xs)
    plt.plot(ys1, color = 'red', label = 'Train')
    plt.plot(ys2, color = 'steelblue', label = 'Valid')
    # plt.plot(ys3, color='seagreen', label='a=0.001')

    plt.xlabel('epoch')
    plt.ylabel('average negative likehood')
    plt.legend()
    plt.show()
    # plt.xlim()

    # pre_label_train = predict(parameter, dat_bia_train)
    # pre_label_test = predict(parameter, dat_bia_test)
    # pre_label_valid = predict(parameter, dat_bia_valid)



    # fileOut(pre_label_train, pre_label_test, label_train, label_test, f_train_out, f_test_out, metrics_out)

    print(time.time() - st)




