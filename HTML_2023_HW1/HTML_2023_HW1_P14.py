import random
from tqdm import trange

def average(lst):
    return sum(lst) / len(lst)

def cdot(args, w_t, x_list):
    sum = 0
    for i in range(args['feature']):
        sum += w_t[i]*x_list[i]
    return sum

def E_in(args, w, x_train, y_train):
    sum = 0
    for i in range(args['size']):
        wx = cdot(args, w, x_train[i])
        if((wx <= 0 and y_train[i] > 0) or (wx > 0 and y_train[i] < 0)):
            sum += 1
    return sum/args['size']

def PLA(args, seed, x_train, y_train):
    random.seed(seed)
    time = 0
    M = 4 * args['size']
    w_t = [0]*args['feature']

    while(time < M):
        time += 1
        pick = random.randint(0, args['size']-1)
        wx = cdot(args, w_t, x_train[pick])
        if((wx <= 0 and y_train[pick] > 0) or (wx > 0 and y_train[pick] < 0)):
            time = 0
            w_t = [w_t[i] + x_train[pick][i] * y_train[pick] for i in range(args['feature'])]
    return w_t

def read_file(args):
    x_train = list()
    y_train = list()
    with open (args['filename'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_data = line.split()
            line_data = [float(i) for i in line_data]
            x_train.append([1]+line_data[:-1])  # set x_0 = 1 to every X_n
            y_train.append(line_data[-1])
    return x_train, y_train

def main(args):
    x_train, y_train = read_file(args)

    error_list = list()
    for i in trange(args['repeat_time']):
        w_pla = PLA(args, args['seed']+i, x_train, y_train)
        error = E_in(args, w_pla, x_train, y_train)
        error_list.append(error)

    error_avg = average(error_list)
    print("average E_in(w_pla):", error_avg)

if __name__ == '__main__':
    args = {
    'feature': 11,                 # include y
    'filename': 'hw1_train.dat',
    'seed': 1126,
    'size': 256,                    # size = N
    'repeat_time': 1000,
    }

    main(args)