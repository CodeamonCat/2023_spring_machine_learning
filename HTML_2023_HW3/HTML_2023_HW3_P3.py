import numpy as np

def phi(x):
    return [1, x[0], x[1], x[0]*x[0], x[0]*x[1], x[1]*x[1]]

def main(dataset, weight_vector):
    result = 0
    for i in range(len(weight_vector)):
        print("weight_vector: ", weight_vector[i])
        counter = 0
        for j in range(len(dataset)):
            hypothesis = int(np.sign(np.dot(weight_vector[i], phi(dataset[j]))))
            if hypothesis == dataset[j][2]:
                print("O:", "(", hypothesis, ",", dataset[j][2], ")")
                counter += 1
            else:
                print("X:", "(", hypothesis, ",", dataset[j][2], ")")
        if counter == len(dataset):
            print("===separates the dataset correct===")
            result += 1
        else:
            print("===separates the dataset incorrect===")
    print("There are", result, "weight vectors that can seperate dataset.")

if __name__ == '__main__':
    dataset = np.array([
        # x_1,x_2, y
        [0, 0, -1],
        [4, 0, 1],
        [-4, 0, 1],
        [0, 2, -1],
        [0, -2, -1],
    ])

    weight_vector = np.array([
        [-1, 0, 0, 0.5, 0, -0.5],
        [-1, 0, 0, -0.5, 0, 0.5],
        [-2, 0, 0, 1, 0, 1],
        [-1, 0, 0, 0.2, 0, 0.1],
    ])

    main(dataset, weight_vector)