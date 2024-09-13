import numpy as np
from libsvm.svmutil import *

def main(args):
    for classifier in range(2,7):
        print(f'=== "{classifier}" versus "not {classifier}"')
        y_train, x_train = svm_read_problem(args['svm_train'])
        for type in range(len(y_train)):
            y_train[type] = 1 if y_train[type] == classifier else 0
        
        prob = svm_problem(y_train, x_train)
        param = svm_parameter('-s 0 -t 1 -d {Q} -g 1 -r 1 -c {C} -q'.format(C=args['C'], Q=args['Q']))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
        print("sVM: ", len(m.get_SV()))

if __name__ == '__main__':
    args = {
    'adaboost_train': "adaboost_letter.scale.tr",
    'adaboost_test': "adaboost_letter.scale.t",
    'C': 1,
    'Q': 2,
    'svm_train': "svm_letter.scale.tr",
    'svm_test': "svm_letter.scale.t",
    }

    main(args)