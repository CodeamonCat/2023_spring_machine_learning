import numpy as np
from itertools import combinations_with_replacement
from liblinear.liblinearutil import *

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
    x_test, y_test = read_file(args, args['filename_test'])
    x_train = transformation(args, x_train)
    x_test = transformation(args, x_test)

    E_out = list()
    for lamb in args['lambda']:
        C = 1/(lamb)
        prob = problem(y_train, x_train)
        param = parameter('-s 6 -c {} -e 0.000001 -q'.format(C))
        m = train(prob, param)
        p_label, p_acc, p_val = predict(y_test, x_test, m)
        E_out.append(round(np.mean(y_test != p_label), 6))

    min_E_out = min(E_out)
    min_E_out_index = [i for i, v in enumerate(E_out) if v == min_E_out]
    print("E_out: ", E_out)
    print("min E_out: ", min(E_out))
    print("Choice: ", chr(97+max(min_E_out_index)))
    
if __name__ == '__main__':
    args = {
        'dimension': 0,
        'filename_test': "hw4_test.dat",
        'filename_train': "hw4_train.dat",
        'lambda': [10**(-6), 10**(-3), 10**(0), 10**3, 10**6],
        'Q': 4,
    }

    main(args)