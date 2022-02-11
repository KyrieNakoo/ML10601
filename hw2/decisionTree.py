import sys
import numpy as np
import csv
import math
from math import log
from scipy.stats import entropy


def load_data(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for record in tsv_reader:
            tsv_data.append(record)
        tsv_data = tsv_data[1:]
        return tsv_data


def entropyLabel(data):
    label_list = [i[-1] for i in data]
    type_label = list(set(label_list))
    dict_labelCount = {}
    entropy = 0
    for key in label_list:
        dict_labelCount[key] = dict_labelCount.get(key, 0)+1

    for key in dict_labelCount:
        # prob = dict_labelCount[key]/len(label_list)
        entropy -= (dict_labelCount[key]/len(label_list))*log((dict_labelCount[key]/len(label_list)), 2)
    return entropy                    # entropy of the label i.e H(Y)


def entropyCal(data, id):
    id = int(id)
    attribute_list = [i[id] for i in data]
    label_list = [i[-1] for i in data]
    type_attribute = list(set(attribute_list))
    type_label = list(set(label_list))
    # print(attribute_list)

    entro = 0
    for key in type_attribute:
        part_label = []
        for i in range(len(attribute_list)):
            if attribute_list[i] == key:
                part_label.append(data[i])
        # print(part_label)
        # part_entropy = entropyLabel(part_label)
        # print(entropyLabel(part_label))
        entro += (len(part_label)/len(label_list))*(entropyLabel(part_label))
    return entro


def inf_Gain(data, id):
    entropy = entropyLabel(data)
    con_entropy = entropyCal(data, id)
    infGain = entropy - con_entropy
    return infGain


def featureToSplit(data):
    data = np.array(data)
    attribute_count = data.shape[1]-1
    label_col = [i[-1] for i in data]

    infGain = []
    for j in range(attribute_count):
        # attribute_col = [i[j] for i in data]
        infGain.append(inf_Gain(data, j))
    # print(infGain)

    if max(infGain) > 0:
        node = infGain.index(max(infGain))
        col = data[:, node]
        return node        # get the feature with max IG.

    return None


def splitDataset(data, id, cat):
    # data = np.array(data)
    id = int(id)
    attribute_list = [i[id] for i in data]
    type_attribute = list(set(attribute_list))
    # print(attribute_list)

    subData = [data[j] for j in range(len(data)) if data[j][id] == cat]
    return subData


def createTree(data, depth, max_depth):
    label = [i[-1] for i in data]
    depth = int(depth)
    max_depth = int(max_depth)
    if label.count([label[0]]) == len(label):
        return label[0]

    if featureToSplit(data) == None:
        return max(sorted(label, reverse=True), key=label.count)

    if depth > max_depth -1:
        return max(sorted(label, reverse=True), key=label.count)

    if depth == len(data[0])-1:
        return max(sorted(label, reverse=True), key=label.count)
    else:
        divAttribute = int()
        divAttribute = featureToSplit(data)
        depth += 1

        mytree = {divAttribute: {}}

        attributeCol = [i[divAttribute] for i in data]
        attributeValue = list(set(attributeCol))

        for cat in attributeValue:
            mytree[divAttribute][cat] = createTree(splitDataset(data, divAttribute, cat), depth, max_depth)
    return mytree


def inference(data, tree):
    predictLabel = []
    for sample in data:
        node = tree
        while isinstance(node, dict):
            key = sample[list(node.keys())[0]]
            node = list(node.values())[0][key]
        predictLabel.append(node)
    return predictLabel



def outFile(dat_train, dat_test, f_train, f_test, f_error):
    train_label = [i[-1] for i in dat_train]
    test_label = [i[-1] for i in dat_test]
    er_trainCount = 0
    er_testCount = 0

    train_preLabel = inference(dat_train, tree)
    test_preLabel = inference(dat_test, tree)

    for i in range(len(train_label)):
        if train_label[i] != train_preLabel[i]:
            er_trainCount += 1

    for i in range(len(test_label)):
        if test_label[i] != test_preLabel[i]:
            er_testCount += 1

    trainErr = er_trainCount/len(dat_train)
    testErr = er_testCount/len(dat_test)
    # print(er_trainCount, er_testCount)
    # print(len(dat_test), len(dat_train))


    f_error = str(f_error)
    f = open(f_error, "w")
    f.write("error(train): ")
    f.write(str(trainErr))
    f.write('\n')
    f.write("error(test): ")
    f.write(str(testErr))
    f.write('\n')
    f.close()

    f_train = str(f_train)
    f1 = open(f_train, "w")
    for lab in train_preLabel:
        f1.write(lab)
        f1.write('\n')
    f1.close()

    f_test = str(f_test)
    f1 = open(f_test, "w")
    for lab in test_preLabel:
        f1.write(lab)
        f1.write('\n')
    f1.close()







if __name__ == '__main__':
    train_input = sys.argv[1]
    input_test = sys.argv[2]
    max_depth = sys.argv[3]
    out_train = sys.argv[4]
    out_test = sys.argv[5]
    metrix_name = sys.argv[6]

    data_test = load_data(input_test)

    data_input = load_data(input_train)

    # print(entropyCal(data_input, 2))
    # print(entropyCal(data_input,0))
    # id = featureToSplit(data_input)
    # print(id)
    # print(splitDataset(data_input, id, 'notA'))
    tree = createTree(data_input, 0, max_depth)
    # print(tree)
    # print(inference(data_input, tree))
    outFile(data_input, data_test, out_train, out_test, metrix_name)
