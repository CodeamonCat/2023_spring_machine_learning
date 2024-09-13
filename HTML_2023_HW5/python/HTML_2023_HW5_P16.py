import numpy as np
import random
from libsvm.svmutil import *
from tqdm import trange

def main(args):
    choices = [0.1, 1, 10, 100, 1000]
    y_train, x_train = svm_read_problem(args['svm_train'])

    for type in range(len(y_train)):
        y_train[type] = 1 if y_train[type] == args['classifier'] else 0

    gamma_count = [0, 0, 0, 0, 0]
    for i in trange(args['repeat_time']):
        sampling = random.sample(range(len(y_train)), args['sampling'])
        training_id = [i for i in range((len(y_train))) if i not in sampling]
        validation_id = [i for i in range((len(y_train))) if i in sampling]

        min = 1
        min_gamma = 0
        for i, g in enumerate(choices):
            prob = svm_problem([y_train[idx] for idx in training_id], [x_train[idx] for idx in training_id])
            param = svm_parameter('-s 0 -t 2 -g {gamma} -c {C} -q'.format(C=args['C'], gamma=g))
            m = svm_train(prob, param)
            p_label, p_acc, p_val = svm_predict([y_train[idx] for idx in validation_id], [x_train[idx] for idx in validation_id], m)
            print(p_acc)
            if p_acc[1] < min:
                min_gamma = i
                min = p_acc[1]
        gamma_count[min_gamma] += 1
        print("gamma_count:", gamma_count)
    print("final gamma_count:", gamma_count)

if __name__ == '__main__':
    args = {
    'C': 0.1,
    'classifier': 7,
    'sampling': 200,
    'svm_train': "svm_letter.scale.tr",
    'svm_test': "svm_letter.scale.t",
    'repeat_time': 500,
    }

    main(args)