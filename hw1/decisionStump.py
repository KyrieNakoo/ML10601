import sys
import numpy as np
import csv

def load_data(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for record in tsv_reader:
            tsv_data.append(record)
        tsv_data = tsv_data[1:]
        return tsv_data


def classify(data_train, data_test, id, f_train, f_test, error):
    id = int(id)
    hint_train = [i[id] for i in data_train]
    col_train = [i[-1] for i in data_train]
    val = np.zeros((2,2))
    x = list(set(hint_train))
    y = list(set(col_train))

    label_train = []
    label_test = []
    err_test = 0
    err_train = 0

    for i in range(len(data_train)):
        if hint_train[i] == x[0] and col_train[i] == y[0]:
            val[0][0]+=1
        if hint_train[i] == x[0] and col_train[i] == y[1]:
            val[0][1]+=1
        if hint_train[i] == x[1] and col_train[i] == y[0]:
            val[1][0]+=1
        if hint_train[i] == x[1] and col_train[i] == y[1]:
            val[1][1]+=1
    # return(val)

    max = np.max(val)
    index = np.unravel_index(val.argmax(), val.shape)
    # return(index, err_train)

    det_col = index[0]
    det_party = index[1]
    s = int(index[1])
    det_partyOpp = -s+1
    det = [x[det_col], y[det_party]]
    det_Opp = [x[det_col], y[det_partyOpp]]

    # return(det)   # the determining label

    for j in range(len(data_train)):
        if hint_train[j] == det[0]:
            label_train.append(det[1])
        else:
            label_train.append(det_Opp[1])

    for k in range(len(data_test)):
        hint_test = [i[id] for i in data_test]
        if hint_test[k] == det[0]:
            label_test.append(det[1])
        else:
            label_test.append(det_Opp[1])

    col_test = [i[-1] for i in data_test]
    for l in range(len(data_test)):
        if col_test[l] != label_test[l]:
            err_test+=1

    for m in range(len(data_train)):
        if col_train[m] != label_train[m]:
            err_train+=1

    err_train = err_train/(len(data_train))
    err_test = err_test/(len(data_test))
    error_matrix = [err_train, err_test]
    # print(error_matrix)

    f_train = str(f_train)
    f_test = str(f_test)
    f_error = str(error)

    f1 = open(f_train, "w")
    for lab in label_train:
        f1.write(lab)
        f1.write('\n')
    f1.close()

    f2 = open(f_test, "w")
    for lab in label_test:
        f2.write(lab)
        f2.write('\n')
    f2.close()

    f3 = open(f_error, "w")
    f3.write("error(train): ")
    f3.write(str(error_matrix[0]))
    f3.write('\n')
    f3.write("error(test): ")
    f3.write(str(error_matrix[1]))
    f3.write('\n')
    f3.close()

    print(f1,f2,f3)



















if __name__ == '__main__':
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    id = sys.argv[3]
    f_trainName = sys.argv[4]
    f_testName = sys.argv[5]
    error = sys.argv[6]
    train_read = (load_data(train_data))
    test_read = (load_data(test_data))
    # print(dat)
    classify(train_read, test_read, id, f_trainName, f_testName, error)

