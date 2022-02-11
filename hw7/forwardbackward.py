import numpy as np
import sys
import csv
import learnhmm as lh
import math


def load_word(data, dic_word, dic_tag):
    index_pair = []
    with open(data, 'r') as f:
        seq_6 = []
        reader = f.readlines()
        for i in range(len(reader)):
            each = reader[i].split()
            if reader[i] != '\n':
                seq = [(dic_word[each[0]]), dic_tag[each[1]]]
                seq_6.append(seq)
            elif reader[i] == '\n':
                index_pair.append(seq_6)
                seq_6 = []
        index_pair.append(seq_6)
        return index_pair


def beta(pi, B, A, index_pair, word_dict, tag_dict):
    rev_tag_dic = {v: k for k, v in tag_dict.items()}
    rev_word_dic = {v: k for k, v in word_dict.items()}
    pre_word_tag = []
    lls = float()
    correct = 0
    length = 0
    for seq in index_pair:
        word = [rev_word_dic[seq[0][0]]]
        a = np.log(A[:, seq[0][0]]) + np.log(pi)
        b = np.mat(np.zeros((len(tag_dict), 1)))
        for i in range(1, len(seq)):
            word.append(rev_word_dic[seq[i][0]])
            sub = np.zeros((len(B), 1))
            sub1 = np.zeros((len(B), 1))
            for j in range(len(B)):
                x = a[:, i - 1] + np.log(B[:, j])
                y = np.log(A[:, seq[::-1][i-1][0]]) + b[:, 0] + np.log(B[j, :].T)
                part = lse(x)
                part1 = lse(y)
                sub[j, 0] = np.log(A[j, seq[i][0]]) + part
                sub1[j, 0] = part1
            a = np.column_stack((a, sub))
            b = np.column_stack((sub1, b))
        p = a+b
        lls += lse(a[:, -1])
        pre_index = np.argmax(p, axis=0)
        pre_index_list = list(np.array(pre_index))[0]
        pre_tag = [rev_tag_dic[pre_index_list[i]] for i in range(len(pre_index_list))]
        pair = list(zip(word, pre_tag))
        pre_word_tag.append(pair)

        correct += np.sum(np.mat(seq)[:, 1].T == np.mat(pre_index))
        length += len(seq)

    ave_log = lls/len(index_pair)
    ave_log = float(ave_log)
    ratio = correct/length
    return pre_word_tag, ave_log, ratio


def lse(data):
    m = np.max(data)
    tem = 0
    for i in range(len(data)):
        tem += np.exp(data[i] - m)
    return m + np.log(tem)


def file_out(predict, ave_log, ratio, pre_name, matrix_name):
    f_predict = str(pre_name)
    f1 = open(f_predict, "w")
    for pair in predict:
        for word, tag in pair:
            f1.write(word)
            f1.write('\t')
            f1.write(tag)
            f1.write('\n')
        f1.write('\n')
    f1.close()

    f_matrix = str(matrix_name)
    f2 = open(f_matrix, "w")
    f2.write("Average Log-Likelihood: " + str(ave_log))
    f2.write('\n')
    f2.write("Accuracy: " + str(ratio))
    f2.close()


if __name__ == '__main__':
    validation_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    # tag_dic: [tag string] to index int
    tag_dic = lh.load_dict(index_to_tag)
    # word_dic: [word string] to index int
    word_dic = lh.load_dict(index_to_word)

    valid_index_pair = load_word(validation_input, word_dic, tag_dic)

    ini = np.mat(np.loadtxt(str(hmminit))).T
    # print(ini)
    trans = np.mat(np.loadtxt(str(hmmtrans)))
    # print(trans)
    emit = np.mat(np.loadtxt(str(hmmemit)))

    # print(valid_word_index)
    # predict_out, average_log, ratio = beta(ini, trans, emit, valid_index_pair, word_dic, tag_dic)
    predict, average, ratio = beta(ini, trans, emit, valid_index_pair, word_dic, tag_dic)
    file_out(predict, average, ratio, predicted_file, metric_file)
