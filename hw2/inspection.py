import sys
import numpy as np
import csv
import math

def load_data(data):
    tsv_data = []
    with open(data, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for record in tsv_reader:
            tsv_data.append(record)
        tsv_data = tsv_data[1:]
        return tsv_data



def inspect(data_input, f_out):
    label = [i[-1] for i in data_input]
    val = np.zeros(2)
    x = list(set(label))
    rate = np.zeros(2)
    entropy_root = int()

    for i in range(len(data_input)):
        if label[i] == x[0]:
            val[0]+=1
        if label[i] == x[1]:
            val[1]+=1

    for j in range(len(x)):
        rate[j] = val[j]/len(data_input)
        entropy_root -= rate[j]*(math.log(rate[j],2))

    max = np.max(val)
    index = val.argmax()
    max_label = x[index]
    error = 1-max/len(data_input)


    matrix = [entropy_root, error]
    f_out = str(f_out)

    f = open(f_out, "w")
    f.write("entropy: ")
    f.write(str(matrix[0]))
    f.write('\n')
    f.write("error: ")
    f.write(str(matrix[1]))
    f.write('\n')
    f.close()

    print(f)













if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]

    data_input = load_data(input)
    inspect(data_input, output)