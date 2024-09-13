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

def KFold(args, x, y):
    rng = np.random.default_rng()
    idx = np.arange(len(x))
    rng.shuffle(idx)
    x_split = np.array([x[idx[i*args['fold_size']:(i+1)*args['fold_size']]] for i in range(args['fold'])])
    y_split = np.array([y[idx[i*args['fold_size']:(i+1)*args['fold_size']]] for i in range(args['fold'])])
    return x_split, y_split

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
    args['size'] = x_train.shape[0]
    args['fold_size'] = int(args['size'] / args['fold'])

    E_cv_list = list()
    for i in range(args['repeat_time']):
        E_cv = list()
        x_split, y_split = KFold(args, x_train, y_train)

        for lamb in args['lambda']:
            C = 1/(2*lamb)
            E_val = list()
            for fold in range(args['fold']):
                x_valid_fold = x_split[fold]
                y_valid_fold = y_split[fold]
                x_train_fold = np.vstack(x_split[j] for j in range(args['fold']) if j != fold)
                y_train_fold = np.hstack(y_split[j] for j in range(args['fold']) if j != fold)

                prob = problem(y_train_fold, x_train_fold)
                param = parameter('-s 0 -c {} -e 0.000001 -q'.format(C))
                m = train(prob, param)
                p_label, p_acc, p_val = predict(y_valid_fold, x_valid_fold, m)
                E_val.append(round(np.mean(y_valid_fold != p_label), 6))
            E_cv.append(np.mean(E_val))
        E_cv_list.append(min(E_cv))

    print("E_cv: ", np.mean(E_cv_list))

if __name__ == '__main__':
    args = {
        'dimension': 0,
        'filename_test': "hw4_test.dat",
        'filename_train': "hw4_train.dat",
        'fold': 5,
        'fold_size': 0, # size of each fold
        'lambda': [10**(-6), 10**(-3), 10**(0), 10**3, 10**6],
        'Q': 4,
        'repeat_time': 256,
        'seed': 1126,
        'size': 0,
        'split': 120
    }

    main(args)