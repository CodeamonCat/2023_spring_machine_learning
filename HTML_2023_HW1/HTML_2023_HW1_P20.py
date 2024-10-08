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

def median(lst):
    lst = sorted(lst)
    mid = len(lst) // 2
    res = (lst[mid] + lst[~mid]) / 2
    return res

def PLA(args, seed, x_train, y_train):
    random.seed(seed)
    time = 0
    M = 4 * args['size']
    w_t = [0]*args['feature']
    update = 0

    while(time < M):
        time += 1
        pick = random.randint(0, args['size']-1)
        wx = cdot(args, w_t, x_train[pick])
        if((wx <= 0 and y_train[pick] > 0) or (wx > 0 and y_train[pick] < 0)):
            time = 0
            update += 1
            w_t = [w_t[i] + x_train[pick][i] * y_train[pick] for i in range(args['feature'])]
    return w_t, update

def read_file(args):
    x_train = list()
    y_train = list()
    with open (args['filename'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_data = line.split()
            line_data = [float(i) for i in line_data]
            x_train.append([0.1126]+line_data[:-1])  # set x_0 = 1 to every X_n
            y_train.append(line_data[-1])
    return x_train, y_train

def main(args):
    x_train, y_train = read_file(args)
    
    error_list = list()
    update_list = list()
    w_0_list = list()
    w_0x_0_list = list()
    for i in trange(args['repeat_time']):
        w_pla, update = PLA(args, args['seed']+i, x_train, y_train)
        error = E_in(args, w_pla, x_train, y_train)
        error_list.append(error)
        update_list.append(update)
        w_0_list.append(w_pla[0])
        w_0x_0_list.append(w_pla[0]*x_train[0][0])

    error_avg = average(error_list)
    update_median = median(update_list)
    w_0_median = median(w_0_list)
    w_0x_0_median = median(w_0x_0_list)
    print("average E_in(w_pla):", error_avg)
    print("median number of updates:", update_median)
    print("median of all w_0:", w_0_median)
    print("median of all w_0x_0:", w_0x_0_median)

if __name__ == '__main__':
    args = {
    'feature': 11,                  # include y
    'filename': 'hw1_train.dat',
    'seed': 1126,
    'size': 256,                    # size = N
    'repeat_time': 1000,
    }

    main(args)