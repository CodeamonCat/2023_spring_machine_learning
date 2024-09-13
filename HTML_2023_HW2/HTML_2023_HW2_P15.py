import math
import numpy as np
from tqdm import trange

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

def generate_data(args):
    # np.random.seed(args['seed'])  # fixed seed may causes error
    x = np.sort(np.random.uniform(args['range_neg'], args['range_pos'], args['size']))
    y = np.sign(x)
    y[y == 0] = -1
    probability = np.random.uniform(0, 1, args['size'])
    noise_ratio = 1 - args['tau']
    y[probability > noise_ratio] *= -1
    return x, y

def main(args):
    total_E_in = 0; total_E_out = 0

    for i in trange(args['repeat_time']):
        x, y = generate_data(args)
        s, theta, E_in = decision_stump(args, x, y)
        E_out = min(math.fabs(theta), args['range_pos'])*(1-2*args['tau'])+args['tau']
        total_E_in += E_in
        total_E_out += E_out
    
    print("Average total_E_in:", total_E_in/args['repeat_time'])
    print("Average total_E_out:", total_E_out/args['repeat_time'])
    print("mean of E_out(g,tau)-E_in(g):", (total_E_out-total_E_in)/args['repeat_time'])

if __name__ == '__main__':
    args = {
        'range_neg': -0.5,
        'range_pos': 0.5,
        'repeat_time': 100000,
        'seed': 1126,
        'size': 2,
        'tau': 0.2
    }

    main(args)