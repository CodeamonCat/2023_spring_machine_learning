import numpy as np

def decision_stump(args, x, y):
    # h(x) = s * sign (x - theta)
    s = 1; theta = 0
    theta_list = np.array([float('-inf')] + [((x[i]+x[i+1])/2) for i in range(0, x.shape[0]-1)])
    error = float('inf')
    for theta_hypothesis in theta_list:
        y_pos = np.where(x > theta_hypothesis, 1, -1)
        y_neg = np.where(x <= theta_hypothesis, 1, -1)
        error_pos = sum(y_pos != y)
        error_neg = sum(y_neg != y)
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

    if theta == float('-inf'): theta = -0.5
    return s, theta, float(error/x.shape[0])

def decision_stump_multi(args, x, y):
    s = np.zeros((args['size'],))
    theta = np.zeros((args['size'],))
    error = np.zeros((args['size'],))

    for i in range(args['size']):
        s[i], theta[i], error[i] = decision_stump(args, x[:, i], y)
    dimension = np.argmin(error)
    if args['type'] == 0: dimension = np.argmax(error)

    return dimension, s[dimension], theta[dimension], error[dimension]

def predict(dimension, s, theta, x_test):
    predict = s * np.sign(x_test[:, dimension]-theta)
    predict[predict == 0] = -1
    return predict

def read_file(filename):
    data = np.loadtxt(filename, dtype=float)
    x = data[:,:-1]
    y = data[:,-1]
    return x, y

def main(args):
    x_train, y_train = read_file(args['filename_train'])
    x_test, y_test = read_file(args['filename_test'])
    
    # change args parameter
    args['size'] = x_train.shape[1]

    dimension_best, s_best, theta_best, E_in_best = decision_stump_multi(args, x_train, y_train)
    args['type'] = 0
    dimension_worst, s_worst, theta_worst, E_in_worst = decision_stump_multi(args, x_train, y_train)
    
    # predict
    y_pred_best = predict(dimension_best, s_best, theta_best, x_test)
    y_pred_worst = predict(dimension_worst, s_worst, theta_worst, x_test)

    # E_out
    E_out_best = np.sum(y_pred_best != y_test)/len(y_test)
    E_out_worst = np.sum(y_pred_worst != y_test)/len(y_test)

    print("E_in_best: ", E_in_best)
    print("E_out_best: ", E_out_best)
    print("delta E_in: ", E_in_worst-E_in_best)
    print("delta E_out: ", E_out_worst-E_out_best)

if __name__ == '__main__':
    args = {
        'filename_test': "hw2_test.dat",
        'filename_train': "hw2_train.dat",
        'size': 1126,
        'type': 1,  # 1: best, 0: worst
        'tau': 0
    }

    main(args)