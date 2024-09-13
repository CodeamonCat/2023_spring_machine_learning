import numpy as np
from libsvm.svmutil import *

def main(args):
    y_train, x_train = svm_read_problem(args['svm_train'])
    target = 1

    for i in range(len(y_train)):
        y_train[i] = 1 if y_train[i] == target else 0
    
    prob = svm_problem(y_train, x_train)
    param = svm_parameter('-s 0 -t 0 -c {} -q'.format(args['C']))
    m = svm_train(prob, param)
    support_vector_coefficients = m.get_sv_coef()
    support_vectors = m.get_SV()

    height = len(support_vector_coefficients)
    width = 0
    primal = list()

    for i in range(len(support_vectors)):
        width = max(max(support_vectors[i].keys()), width)
    
    for i in range(width):
        accuracy = 0
        for j in range(height):
            if(i+1) in support_vectors[i]:
                accuracy += support_vectors[j][i+1] * support_vector_coefficients[j][0]
        primal.append(accuracy)
    primal = np.array(primal)
    
    print("||w||: ", sum(primal * primal) ** 0.5)

if __name__ == '__main__':
    args = {
    'C': 1,
    'svm_train': "svm_letter.scale.tr",
    'svm_test': "svm_letter.scale.t",
    }

    main(args)