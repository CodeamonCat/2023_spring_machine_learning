import numpy as np
import random
from itertools import combinations_with_replacement
from liblinear.liblinearutil import *

def train_test_split(args, x, y):
    rng = np.random.default_rng()
    idx = np.arange(len(x))
    rng.shuffle(idx)
    x_train = x[idx[:args['split']]]
    y_train = y[idx[:args['split']]]
    x_test = x[idx[args['split']:]]
    y_test = y[idx[args['split']:]]
    return x_train, y_train, x_test, y_test

def transformation(args, x):
    x_origin = x[:, 1:] # remove x_{n=0}=1
    transformed_x = x.copy()
    combination_tuples = list()
    iterable = list(i for i in range(1, args['dimension']))
    for r in range(1, args['Q']):
        combination_tuples.extend(list(combinations_with_replacement(iterable, r+1)))
    
    for combination in range(len(combination_tuples)):
        x_temp = 1
        for j in combination_tuples[combination]:
            x_temp = x_origin[:,j-1].reshape(-1, 1) * x_temp
        transformed_x = np.hstack((
            transformed_x, x_temp
            ))
    return transformed_x

def read_file(args, filename):
    data = np.loadtxt(filename, dtype=float)
    x = data[:,:-1]
    x = np.c_[np.ones(len(x)), x] # x_{n=0}=1
    y = data[:,-1]
    args['dimension'] = x.shape[1]
    return x, y

def main(args):
    x_train, y_train = read_file(args, args['filename_train'])
    x_train = transformation(args, x_train)
    x_test, y_test = read_file(args, args['filename_test'])
    x_test = transformation(args, x_test)
    lamb_list = list()
    E_val_list = list()
    E_out_list = list()
    for i in range(args['repeat_time']):
        x_train_split, y_train_split, x_test_split, y_test_split = train_test_split(args, x_train, y_train)
        args['size'] = x_train_split.shape[0]

        model_list = list()
        E_val = list()
        for lamb in args['lambda']:
            C = 1/(2*lamb)
            prob = problem(y_train_split, x_train_split)
            param = parameter('-s 0 -c {} -e 0.000001 -q'.format(C))
            m = train(prob, param)
            p_label, p_acc, p_val = predict(y_test_split, x_test_split, m)
            E_val.append(round(np.mean(y_test_split != p_label), 6))
            model_list.append(m)

        min_E_val = min(E_val)
        min_E_val_index = [i for i, v in enumerate(E_val) if v == min_E_val]
        lamb_list.append(max(min_E_val_index))
        E_val_list.append(min_E_val)
        # predict test dataset
        p_label, p_acc, p_val = predict(y_test, x_test, model_list[max(min_E_val_index)])
        E_out_list.append(round(np.mean(y_test != p_label), 6))

    best_lambda = args['lambda'][max(set(lamb_list), key = lamb_list.count)]
    print("best_lambda: ", best_lambda)
    print("E_out: ", np.mean(E_out_list))
    
if __name__ == '__main__':
    args = {
        'dimension': 0,
        'filename_test': "hw4_test.dat",
        'filename_train': "hw4_train.dat",
        'lambda': [10**(-6), 10**(-3), 10**(0), 10**3, 10**6],
        'Q': 4,
        'repeat_time': 256,
        'seed': 1126,
        'size': 0,
        'split': 120
    }

    main(args)