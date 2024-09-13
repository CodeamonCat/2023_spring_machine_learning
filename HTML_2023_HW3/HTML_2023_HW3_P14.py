import numpy as np
import random
from tqdm import trange

def calculate_E_in_sqr(x, y, w):
    return np.mean([(np.dot(w, x[i]) - y[i])**2 for i in range(len(x))])

def read_file(args, filename):
    data = np.loadtxt(filename, dtype=float)
    x = data[:,:-1]
    y = data[:,-1]
    args['features'] = len(x[0])
    args['size'] = len(x)
    return x, y

def SGD(args, x, y):
    error_list = list()
    w_list = list()
    for i in trange(args['repeat_time']):
        random.seed()
        w = np.zeros(args['feature'])
        for j in range(args['iteration']):
            pick = random.randint(0, args['size']-1)
            w = w + args['learning_rate'] * 2 * (y[pick] - np.dot(w, x[pick])) * x[pick]
        error_list.append(calculate_E_in_sqr(x, y, w))
        w_list.append(w)
    return w_list, error_list

def main(args):
    x_train, y_train = read_file(args, args['filename_train'])
    x_train = np.c_[np.ones(len(x_train)), x_train] # x_{n, 0}=1
    args['feature'] = len(x_train[0])
    
    weight_list, E_in_list = SGD(args, x_train, y_train)
    E_in_average = np.mean(E_in_list)
    print("E_in_average: ", E_in_average)

if __name__ == '__main__':
    args = {
        'filename_test': "hw3_test.dat",
        'filename_train': "hw3_train.dat",
        'feature': 11,  # without include y
        'iteration': 800,
        'learning_rate': 0.001,
        'repeat_time': 1000,
        'size': 0
    }

    main(args)