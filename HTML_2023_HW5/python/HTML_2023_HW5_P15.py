import numpy as np
from libsvm.svmutil import *

def main(args):
    choices = [0.1, 1, 10, 100, 1000]
    y_train, x_train = svm_read_problem(args['svm_train'])
    y_test, x_test = svm_read_problem(args['svm_test'])

    for type in range(len(y_train)):
        y_train[type] = 1 if y_train[type] == args['classifier'] else 0
    for type in range(len(y_test)):
        y_test[type] = 1 if y_test[type] == args['classifier'] else 0

    for choice in choices:
        print(f'=== choice: {choice}')
        prob = svm_problem(y_train, x_train)
        param = svm_parameter('-s 0 -t 2 -g {gamma} -c {C} -q'.format(C=args['C'], gamma=choice))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y_test, x_test, m)

if __name__ == '__main__':
    args = {
    'C': 0.1,
    'classifier': 7,
    'svm_train': "svm_letter.scale.tr",
    'svm_test': "svm_letter.scale.t",
    }

    main(args)