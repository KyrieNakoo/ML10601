import numpy as np
import sys
import csv
import math
# from numpy import *
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def load_label(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter=',')
        for record in tsv_reader:
            tsv_data.append(float(record[0]))
        label = np.mat(tsv_data)
        return label


def load_data(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter=',')
        for each in tsv_reader:
            record = list(map(float, each[1:]))
            tsv_data.append(record)
        pix_value = np.mat(tsv_data).T
    return pix_value


def ini_par(feature_data, label, flag, D, K):
    num = feature_data.shape[0]
    mode = int(flag)
    D = int(D)
    K - int(K)
    a = np.insert(np.zeros((D, num)), 0, values=np.zeros(D), axis=1)
    b = np.insert(np.zeros((K, D)), 0, values=np.zeros(K), axis=1)
    fwb = np.insert(feature_data, 0, values=np.ones(feature_data.shape[1]), axis=0)
    y = np.zeros((K, feature_data.shape[1]))
    for i in range(label.shape[1]):
        hot = int(label[0, i])
        y[hot, i] = 1
    y = np.mat(y)
    # if mode == 2
    #     alpha = np.zeros(num).reshape(num, 1)
    #     beta = np.zeros(num).reshape(num, 1)
    if mode == 1:
        a = np.insert((2*np.random.random((D, num))-1)/10, 0, values=np.zeros(D), axis=1)
        b = np.insert((2*np.random.random((K, D))-1)/10, 0, values=np.zeros(K), axis=1)
    return fwb, y, a, b


def forward(alpha_temp, beta_temp, sample):
    a_j = np.dot(alpha_temp, sample)
    z_j = 1 / (1 + np.exp(-a_j))
    # # z_j_bia D+1 x N
    z_j_bia = np.row_stack((np.ones(sample.shape[1]), z_j))
    # # b_k K x N
    b_k = np.dot(beta_temp, z_j_bia)
    # print(b_k.shape)
    b_k_sum = np.dot(np.ones(b_k.shape[0]), np.exp(b_k))
    # # y_hat_k 4 x N
    y_hat_k = np.exp(b_k) / b_k_sum
    return y_hat_k


def update_par(alpha_i, beta_i, pixel_tr, pixel_val, y_hot_tr, y_hot_val, epoch, rate):
    s_t_alpha = np.zeros([alpha_i.shape[0], alpha_i.shape[1]])
    s_t_beta = np.zeros([beta_i.shape[0], beta_i.shape[1]])
    alpha = alpha_i
    beta = beta_i
    entropy_epoch = np.empty(shape=[int(epoch), 2])
    y_hat_tr = np.empty(shape=[4, 0])
    y_hat_val = np.empty(shape=[4, 0])
    eps = 0.00001
    loss = 0
    for i in range(int(epoch)):
        # y_hat_loop = np.empty(shape=[4, 0])
        # print(y_hat_loop)
        for j in range(len(pixel_tr.T)):
            # # a_j D x 1
            a_j = np.dot(alpha, pixel_tr[:, j])
            z_j = 1 / (1 + np.exp(-a_j))
            # # z_j_bia D+1 x 1
            z_j_bia = np.row_stack((np.ones(1), z_j))
            # # b_k K x 1
            b_k = np.dot(beta, z_j_bia)
            b_k_sum = np.dot(np.ones(b_k.shape[0]), np.exp(b_k))
            # # y_hat_k 4 x 1
            y_hat_k = np.exp(b_k) / b_k_sum
            # y_hat_loop = np.c_[y_hat_loop, y_hat_k]
            # print(y_hat_loop)
            cor_entro_loss = -sum(np.multiply(y_hot_tr[:, j], np.log(y_hat_k)))
            # print(cor_entro_loss)
            # # der_l_b 4 x 1
            der_l_b = y_hat_k - y_hot_tr[:, j]
            # # der_l_beta 4 x D+1
            der_l_beta = np.dot(der_l_b, z_j_bia.T)
            # # der_l_z D x 1
            der_l_z = np.dot(der_l_b.T, beta[:, 1:]).T
            # # der_l_a D x 1
            der_l_a = np.multiply(der_l_z, np.multiply(z_j, 1 - z_j))
            # # der_l_alpha Dx M+1
            der_l_alpha = np.dot(der_l_a, pixel_tr[:, j].T)
            s_t_alpha += np.square(der_l_alpha)
            alpha -= np.multiply(float(rate)/np.sqrt(s_t_alpha + eps), der_l_alpha)
            s_t_beta += np.square(der_l_beta)
            beta -= np.multiply(float(rate)/np.sqrt(s_t_beta + eps), der_l_beta)
        # a = np.dot(alpha, pixel)
        # z = 1 / (1 + np.exp(-a))
        # # # z_bia D+1 x N
        # z_bia = np.row_stack((np.ones(pixel.shape[1]), z))
        # # # b K x N
        # b = np.dot(beta, z_bia)
        # b_sum = np.dot(np.ones(b.shape[0]), np.exp(b))
        # # # y_hat 4 x N
        # y_hat_loop = np.exp(b) / b_sum
        # entropy_epoch[i+1] = -np.sum(np.multiply(y_hot, np.log(y_hat_loop)))/(pixel.shape[1])
        y_hat_loop_tr = forward(alpha, beta, pixel_tr)
        y_hat_loop_val = forward(alpha, beta, pixel_val)
        # print(y_hat_loop.shape, alpha.shape, beta.shape)
        entropy_epoch[i, 0] = -np.sum(np.multiply(y_hot_tr, np.log(y_hat_loop_tr))) / (pixel_tr.shape[1])
        entropy_epoch[i, 1] = -np.sum(np.multiply(y_hot_val, np.log(y_hat_loop_val))) / (pixel_val.shape[1])
        y_hat_tr = y_hat_loop_tr
        y_hat_val = y_hat_loop_val
        # print(entropy_epoch[i+1])
    # return alpha, beta, y_hat, entropy_epoch
    # print(y_hat)
    # return alpha, beta, y_hat_tr, y_hat_val, entropy_epoch
    print(alpha, '\n', beta)


def plotEm(alpha_i, beta_i, pixel_tr, y_hot_tr, epoch, rate):
    s_t_alpha = np.zeros([alpha_i.shape[0], alpha_i.shape[1]])
    s_t_beta = np.zeros([beta_i.shape[0], beta_i.shape[1]])
    alpha = alpha_i
    beta = beta_i
    entropy_tr_epoch = {}
    # y_hat_tr = np.empty(shape=[4, 0])
    # y_hat_val = np.empty(shape=[4, 0])
    eps = 0.00001
    loss = 0
    for i in range(int(epoch)):
        # y_hat_loop = np.empty(shape=[4, 0])
        # print(y_hat_loop)
        for j in range(len(pixel_tr.T)):
            # # a_j D x 1
            a_j = np.dot(alpha, pixel_tr[:, j])
            z_j = 1 / (1 + np.exp(-a_j))
            # # z_j_bia D+1 x 1
            z_j_bia = np.row_stack((np.ones(1), z_j))
            # # b_k K x 1
            b_k = np.dot(beta, z_j_bia)
            b_k_sum = np.dot(np.ones(b_k.shape[0]), np.exp(b_k))
            # # y_hat_k 4 x 1
            y_hat_k = np.exp(b_k) / b_k_sum
            # y_hat_loop = np.c_[y_hat_loop, y_hat_k]
            # print(y_hat_loop)
            cor_entro_loss = -sum(np.multiply(y_hot_tr[:, j], np.log(y_hat_k)))
            # print(cor_entro_loss)
            # # der_l_b 4 x 1
            der_l_b = y_hat_k - y_hot_tr[:, j]
            # # der_l_beta 4 x D+1
            der_l_beta = np.dot(der_l_b, z_j_bia.T)
            # # der_l_z D x 1
            der_l_z = np.dot(der_l_b.T, beta[:, 1:]).T
            # # der_l_a D x 1
            der_l_a = np.multiply(der_l_z, np.multiply(z_j, 1 - z_j))
            # # der_l_alpha Dx M+1
            der_l_alpha = np.dot(der_l_a, pixel_tr[:, j].T)
            s_t_alpha += np.square(der_l_alpha)
            alpha -= np.multiply(float(rate)/np.sqrt(s_t_alpha + eps), der_l_alpha)
            s_t_beta += np.square(der_l_beta)
            beta -= np.multiply(float(rate)/np.sqrt(s_t_beta + eps), der_l_beta)
        y_hat_loop_tr = forward(alpha, beta, pixel_tr)
        # y_hat_loop_val = forward(alpha, beta, pixel_val)
        # print(y_hat_loop.shape, alpha.shape, beta.shape)
        entropy_tr_epoch[i + 1] = \
            -np.sum(np.multiply(y_hot_tr, np.log(y_hat_loop_tr))) / (pixel_tr.shape[1])
        # y_hat_tr = y_hat_loop_tr
        # y_hat_val = y_hat_loop_val
        # print(entropy_epoch[i+1])
    # return alpha, beta, y_hat, entropy_epoch
    # print(y_hat)
    return entropy_tr_epoch


def loss_0ut(loss_value, f_name, col):
    f_loss = str(f_name)
    f = open(f_loss, "w")
    for key in loss_value:
        f.write(str((loss_value[key][int(col)])))
        f.write('\n')
    f.close()


def err_rate(y_hat_train, label_train, y_hat_val, label_val, entropy, train_label_n, val_label_n, error_n):
    y_train_predict = np.argmax(y_hat_train, axis=0)
    y_val_predict = np.argmax(y_hat_val, axis=0)
    # print(y_predict.shape)
    # print(label_true.shape)
    pre_train_right = np.sum(y_train_predict == label_train)
    pre_val_right = np.sum(y_val_predict == label_val)
    err_num_train = float(y_hat_train.shape[1]-pre_train_right) / y_hat_train.shape[1]
    err_num_val = float(y_hat_val.shape[1]-pre_val_right) / y_hat_val.shape[1]

    f_train = str(train_label_n)
    f1 = open(f_train, "w")
    for i in range(y_train_predict.shape[1]):
        f1.write(str(y_train_predict[0, i]))
        f1.write('\n')
    f1.close()

    f_val = str(val_label_n)
    f2 = open(f_val, "w")
    for i in range(y_val_predict.shape[1]):
        f2.write(str(y_val_predict[0, i]))
        f2.write('\n')
    f2.close()

    f_error = str(error_n)
    f = open(f_error, "w")
    for key in entropy:
        f.write('epoch=%d' % key + " " + "crossentropy(train):" + " " + str((entropy[key][0])))
        f.write('\n')
        f.write('epoch=%d' % key + " " + "crossentropy(validation):" + " " + str((entropy[key][1])))
        f.write('\n')
    f.write("error(train): "+'%f' % err_num_train)
    f.write('\n')
    f.write("error(validation): " + '%f' % err_num_val)
    f.close()


if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = sys.argv[6]
    hidden_units = sys.argv[7]
    init_fig = sys.argv[8]
    learning_rate = sys.argv[9]
    # loss_tr_out = sys.argv[3]
    # loss_val_input = sys.argv[4]
    # loss_sgd_input = sys.argv[5]

    # Matrix, row = example, col = feature

    # # 1 x N
    label_train = load_label(train_input)
    label_valid = load_label(validation_input)
    # # print(label)
    #
    # # # M x N
    feature_train = load_data(train_input)
    feature_valid = load_data(validation_input)
    # print(feature)

    # # feature_bia M+1 x N;
    # # y_hot 4 x N;
    # # alpha D x M+1;
    # # beta K x D+1;
    feature_bia_train, y_hot_train, alpha_input, beta_input = \
        ini_par(feature_train, label_train, init_fig, int(hidden_units), 4)

    feature_bia_valid, y_hot_valid, nouse_alpha, nouse_beta = \
        ini_par(feature_valid, label_valid, init_fig, int(hidden_units), 4)

    update_par(alpha_input, beta_input, feature_bia_train, feature_bia_valid, y_hot_train, y_hot_valid, num_epoch, learning_rate)

    # # 4 x 100
    # y_hat_valid = forward(alpha_trained, beta_trained, feature_bia_valid)
    # print(entro_loop)
    # err_rate(y_hat_trained, label_train, y_hat_valid, label_valid, entro_loop, train_out, validation_out, metrics_out)
    # err_rate(y_hat_trained, label_train)

    # x = np.array([5, 20, 50, 100, 200])
    # y_t = np.empty(shape=[1, 5])
    # y_v = np.empty(shape=[1, 5])
    # for i in range(len(x)):
    #     hid_unit = x[i]
    #     feature_bia_train, y_hot_train, alpha_input, beta_input = ini_par(feature_train, label_train, 1, int(hid_unit), 4)
    #     feature_bia_valid, y_hot_valid, nouse_alpha, nouse_beta = ini_par(feature_valid, label_valid, 1, int(hid_unit), 4)
    #     y_t[0, i] = update_par(alpha_input, beta_input, feature_bia_train, feature_bia_valid, y_hot_train, y_hot_valid, 100, 0.01)[0]
    #     y_v[0, i] = update_par(alpha_input, beta_input, feature_bia_train, feature_bia_valid, y_hot_train, y_hot_valid, 100, 0.01)[1]
    # print(y_t, y_v)

    # plt.figure(figsize=(10, 6))
    # y1 = [1.03333348, 0.83818286, 0.67413324, 0.48285836, 0.31126149]
    # y2 = [1.23357091, 1.21791211, 1.18411412, 1.21851386, 1.34812285]
    #
    # plt.plot(x, y1, color='red', label='Train')
    # plt.plot(x, y2, color='blue', label='Validation')
    # for i in range(len(x)):
    #     plt.scatter(x[i], y1[i], s=20, color='black')
    #     plt.scatter(x[i], y2[i], s=20, color='black')
    # plt.xlabel('epoch')
    # plt.ylabel('Avg.Train and Validation Cross-Entropy Loss')
    # plt.legend()
    # plt.show()

    # feature_bia_train, y_hot_train, alpha_input, beta_input = ini_par(feature_train, label_train, 1, 50, 4)
    # feature_bia_valid, y_hot_valid, nouse_alpha, nouse_beta = ini_par(feature_valid, label_valid, 1, 50, 4)
    # loss = update_par(alpha_input, beta_input, feature_bia_train, feature_bia_valid, y_hot_train, y_hot_valid, 100, 0.01)
    #
    # loss_0ut(loss, loss_out)
    #
    # los_sgd = load_label(loss_sgd_input)
    # los_ad = load_label(loss_ad_input)
    #
    # plt.figure(figsize=(10, 6))
    # x = np.arange(1, 101, 1)
    # y1 = np.asarray(los_sgd).flatten()
    # y2 = np.asarray(los_ad).flatten()
    #
    # plt.plot(x, y1, color='red', label='SGD')
    # plt.plot(x, y2, color='blue', label='SGD with Adagrad')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('Validation cross-entropy loss')
    # plt.show()

    # feature_bia_train, y_hot_train, alpha_input, beta_input = ini_par(feature_train, label_train, 1, 50, 4)
    # feature_bia_valid, y_hot_valid, nouse_alpha, nouse_beta = ini_par(feature_valid, label_valid, 1, 50, 4)
    # entropy = update_par(alpha_input, beta_input, feature_bia_train, feature_bia_valid, y_hot_train, y_hot_valid, 100,
    #                     0.1)
    # entropy_01 = update_par(alpha_input, beta_input, feature_bia_train, feature_bia_valid, y_hot_train, y_hot_valid, 100,
    #                      0.01)
    # entropy_001 = update_par(alpha_input, beta_input, feature_bia_train, feature_bia_valid, y_hot_train, y_hot_valid, 100,
    #                      0.001)
    # # loss_out(entropy, 0)
    #
    # lr1_t = entropy[:, 0]
    # lr1_v = entropy[:, 1]

    # lr01_t = entropy_01[:, 0]
    # lr01_v = entropy_01[:, 1]

    # lr001_t = entropy_001[:, 0]
    # lr001_v = entropy_001[:, 1]
    # plt.figure(figsize=(10, 6))
    # x = np.arange(1, 101, 1)
    # y1 = np.asarray(lr001_t).flatten()
    # y2 = np.asarray(lr001_v).flatten()
    #
    # plt.plot(x, y1, color='red', label='Train')
    # plt.plot(x, y2, color='blue', label='Validation')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('Avg. cross-entropy loss')
    # plt.title('LR = 0.001', fontsize=25)
    # plt.show()






