import math
import numpy as np
from libsvm.svmutil import *
from tqdm import trange

def adaboost(args, x, y):
    u = np.ones((args['size'], )) / args['size']   # weights
    E_in_u, alpha = np.zeros((args['T'],)), np.zeros((args['T'],))
    s, i, theta = np.zeros((args['T'],)), np.zeros((args['T'],), dtype=int), np.zeros((args['T'],))
    for t in trange(args['T']):
        # get optimal parameters
        E_in_u[t], s[t], i[t], theta[t] = decision_stump_multi(args, x, y, u)
        diamond = np.sqrt((1-E_in_u[t])/E_in_u[t])
        # update u, which are weights
        for idx in range(len(y)):
            if y[idx] != s[t] * sign(x[idx][i[t]] - theta[t]):
                u[idx] *= diamond
            else:
                u[idx] /= diamond
        alpha[t] = np.log(diamond)
    return E_in_u, s, i, theta, alpha

def decision_stump(args, x, y, u):
    # h(x) = s * sign (x - theta)
    s = 1; theta = float('-inf'); error = float('inf')
    sorted_x = np.array(sorted(x))
    theta_list = np.array([float('-inf')] + [((sorted_x[i]+sorted_x[i+1])/2) for i in range(0, x.shape[0]-1)])

    for theta_hypothesis in theta_list:
        y_pos = np.where(x >= theta_hypothesis, 1, -1)
        y_neg = np.where(x < theta_hypothesis, 1, -1)
        error_pos = np.sum(u[np.where(y_pos != y)])/np.sum(u)
        error_neg = np.sum(u[np.where(y_neg != y)])/np.sum(u)
        
        if error_pos > error_neg:
            if error_neg < error:
                error = error_neg
                theta = theta_hypothesis
                s = -1
        else:
            if error_pos < error:
                error = error_pos
                theta = theta_hypothesis
                s = 1
    return error, s, theta

def decision_stump_multi(args, x, y, u):
    s = np.zeros((args['dimension'],))
    theta = np.zeros((args['dimension'],))
    error = np.zeros((args['dimension'],))

    for dim in range(args['dimension']):
        error[dim], s[dim], theta[dim] = decision_stump(args, x[:,dim], y, u)
    i = np.argmin(error)
    return error[i], s[i], i, theta[i]

def get_E_in_list(args, x, y, s, i, theta):
    error_list = list()
    for t in range(args['T']):
        error = 0
        for n in range(len(y)):
            if(y[n] != (s[t] * np.sign(x[n][i[t]] - theta[t]))):
                error += 1
        error_list.append(error/len(y))
    return error_list

def predict_G(args, x, y, s, i, theta, alpha):
    error = 0
    for n in range(len(x)):
        temp = 0
        for alpha_n in range(len(alpha)):
            temp += alpha[alpha_n] * (s[alpha_n] * np.sign(x[n][i[alpha_n]] - theta[alpha_n]))
        guess = np.sign(temp)
        if(y[n] != guess):
            error += 1
    return error/len(y)

def sign(value):
    if value >= 0:
        return 1
    else:
        return -1

def transformation(args, y, x):
    transformed_x, transformed_y = list(), list()
    # idx:value => value only
    for n in range(len(y)):
        if(y[n] == args['label_pos'] or y[n] == args['label_neg']):
            transformed_y.append(y[n])
            row = list()
            for index, value in x[n].items():
                row.append(value)
            transformed_x.append(row)
    # one-versus-one: label "label_pos" versus label "label_neg"
    for label in range(len(transformed_y)):
        transformed_y[label] = 1 if transformed_y[label] == args['label_pos'] else -1
    return np.array(transformed_y), np.array(transformed_x)

def main(args):
    y_train, x_train = transformation(args, *svm_read_problem(args['adaboost_train']))
    y_test, x_test = transformation(args, *svm_read_problem(args['adaboost_test']))
    
    # hyper-parameters
    args['dimension'] = x_train.shape[1]
    args['size'] = len(x_train)
    # adaboost and decision_stump
    E_in_u, s, i, theta, alpha = adaboost(args, x_train, y_train)   # only return one value
    # P17, P18
    E_in_list = get_E_in_list(args, x_train, y_train, s, i, theta)
    print("min E_in(g_t): ", min(E_in_list))
    print("max E_in(g_t): ", max(E_in_list))
    # P19, P20
    E_in_G = predict_G(args, x_train, y_train, s, i, theta, alpha)
    E_out_G = predict_G(args, x_test, y_test, s, i, theta, alpha)
    print("E_in(G): ", E_in_G)
    print("E_out(G): ", E_out_G)
    print("===Fuck you HTML===")

if __name__ == '__main__':
    args = {
    'adaboost_train': "adaboost_letter.scale.tr",
    'adaboost_test': "adaboost_letter.scale.t",
    'dimension': 16,
    'label_pos': 11,
    'label_neg': 26,
    'size': 0,
    'T': 1000,
    'type': 1,  # 1: best, 0: worst
    }

    main(args)