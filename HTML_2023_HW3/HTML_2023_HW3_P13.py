import numpy as np

def calculate_E_in_sqr(x, y, w):
    return np.mean([(np.dot(w, x[i]) - y[i])**2 for i in range(len(x))])

def calculate_w_lin(x, y):
    # pseudo inverse: np.linalg.pinv(np.array([values]))
    return np.dot(np.linalg.pinv(x), y)

def read_file(filename):
    data = np.loadtxt(filename, dtype=float)
    x = data[:,:-1]
    y = data[:,-1]
    return x, y

def main(args):
    x_train, y_train = read_file(args['filename_train'])
    x_train = np.c_[np.ones(len(x_train)), x_train] # x_{n, 0}=1

    weight = calculate_w_lin(x_train, y_train)
    E_in = calculate_E_in_sqr(x_train, y_train, weight)
    print("E_in: ", E_in)

if __name__ == '__main__':
    args = {
        'filename_test': "hw3_test.dat",
        'filename_train': "hw3_train.dat"
    }

    main(args)