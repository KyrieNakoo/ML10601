import numpy as np
import sys
import csv


def load_dict(dic):
    word_index = {}
    with open(dic, 'r') as f:
        reader = f.readlines()
        for i in range(len(reader)):
            each = reader[i].split()
            word_index[each[0]] = i
        # tsv_data = tsv_data
    return word_index


def load_word(data, dic_word, dic_tag):
    tsv_tag = []
    index_tag = []
    word_tag = []
    index_pair = []
    with open(data, 'r') as f:
        seq_2 = []
        seq_4 = []
        seq_5 = []
        seq_6 = []
        reader = f.readlines()
        for i in range(len(reader)):
            each = reader[i].split()
            if reader[i] != '\n':
                # seq_1.append(each[0])
                seq_2.append(each[1])
                # seq_3.append(dic_word[each[0]])
                seq_4.append(dic_tag[each[1]])
                seq_5.append(each)
                seq = [(dic_word[each[0]]), dic_tag[each[1]]]
                seq_6.append(seq)
            elif reader[i] == '\n':
                # tsv_word.append(seq_1)
                tsv_tag.append(seq_2)
                # index_seq.append(seq_3)
                index_tag.append(seq_4)
                word_tag.append(seq_5)
                index_pair.append(seq_6)
                # seq_1 = []
                seq_2 = []
                # seq_3 = []
                seq_4 = []
                seq_5 = []
                seq_6 = []
        # tsv_word.append(seq_1)
        tsv_tag.append(seq_2)
        # index_seq.append(seq_3)
        index_tag.append(seq_4)
        word_tag.append(seq_5)
        index_pair.append(seq_6)
        return tsv_tag, index_tag, word_tag, index_pair


def load_all(data, dic_word, dic_tag):
    index_pair = []
    with open(data, 'r') as f:
        seq = []
        pi = np.ones((len(dic_tag), 1))
        A = np.ones((len(dic_tag), len(dic_word)))
        B = np.ones((len(dic_tag), len(dic_tag)))
        reader = f.readlines()
        for i in range(len(reader)):
            each = reader[i].split()
            if reader[i] != '\n':
                one = [(dic_word[each[0]]), dic_tag[each[1]]]
                seq.append(one)
                # print(seq)
                A[dic_tag[each[1]], dic_word[each[0]]] += 1
                # print(A)
            elif reader[i] == '\n':
                index_pair.append(seq)
                pi[seq[0][1], 0] += 1
                # print(pi)
                # print(seq)
                for j, k in list(zip(np.array(seq)[:-1, 1], np.array(seq)[1:, 1])):
                    B[j, k] += 1
                    # print(np.array(seq)[:-1, 1], np.array(seq)[1:, 1])
                    # print(j,k)
                seq = []
        index_pair.append(seq)
        # A[dic_tag[each[1]], dic_word[each[0]]] += 1
        pi[seq[0][1], 0] += 1
        for j, k in list(zip(np.array(seq)[:-1, 1], np.array(seq)[1:, 1])):
            B[j, k] += 1
        ini = pi/sum(pi)
        # print(ini.dtype)
        trans = B / np.mat(np.sum(B, axis=1)).T
        emit = A / np.mat(np.sum(A, axis=1)).T
        return ini, trans, emit, index_pair


def prob_out(pi, B, A, i_name, t_name, e_name):
    np.savetxt(str(i_name), pi, delimiter='\t')
    np.savetxt(str(t_name), B, delimiter='\t')
    np.savetxt(str(e_name), A, delimiter='\t')


if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    # tag_dic: [tag string] to index int
    tag_dic = load_dict(index_to_tag)
    # word_dic: [word string] to index int
    word_dic = load_dict(index_to_word)
    # word, tag: list of sequence word/tag list
    # word_index, tag_index: list of sequence word_index/tag_index list
    # tag, tag_index, word_tag_pair, index_pair = load_word(train_input, word_dic, tag_dic)
    pi, B, A, index_pair = load_all(train_input, word_dic, tag_dic)
    prob_out(pi, B, A, hmminit, hmmtrans, hmmemit)
    # trans_B(B)