import numpy as np
from tqdm import trange

def calculate_E_in_01(x, y, w):
    return np.mean(np.dot(x, w) * y <= 0)

def calculate_w_lin(x, y):
    # pseudo inverse: np.linalg.pinv(np.array([values]))
    return np.dot(np.linalg.pinv(x), y)

def read_file(args, filename):
    data = np.loadtxt(filename, dtype=float)
    x = data[:,:-1]
    y = data[:,-1]
    return x, y

def transformation(x, Q):
    # x_0 is included
    x_origin = x[:, 1:]
    transformed_x = x.copy()
    for q in range(2, Q+1):
        transformed_x = np.hstack((transformed_x, x_origin**q))
    return transformed_x

def main(args):
    x_train, y_train = read_file(args, args['filename_train'])
    x_train = np.c_[np.ones(len(x_train)), x_train] # x_{n, 0}=1
    x_test, y_test = read_file(args, args['filename_test'])
    x_test = np.c_[np.ones(len(x_test)), x_test] # x_{n, 0}=1
    x_train = transformation(x_train, args['Q'])
    x_test = transformation(x_test, args['Q'])

    weight = calculate_w_lin(x_train, y_train)
    E_in = calculate_E_in_01(x_train, y_train, weight)
    E_out = calculate_E_in_01(x_test, y_test, weight)

    print("E_in: ", E_in)
    print("E_out: ", E_out)
    print("|E_in-E_out|: ", np.abs(E_in - E_out))

if __name__ == '__main__':
    args = {
        'filename_test': "hw3_test.dat",
        'filename_train': "hw3_train.dat",
        'Q': 8,
    }

    main(args)