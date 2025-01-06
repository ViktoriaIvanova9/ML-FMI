import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
eps = 0.001

def sigmoid(result):
    return (1 / (1 + np.exp(-result)))

def create_dataset_NAND():
    return [((0, 0), 1),((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

def initialize_weight():
    random_number = np.random.uniform(-1, 1)
    return random_number

def initialize_bias():
    random_bias = np.random.uniform(-1, 1)
    return random_bias

def model(params, x, bias):
    dot_product = np.dot(params, x)
    return sigmoid(dot_product + bias)

def calculate_loss(model, dataset, params_w, bias):
    loss = []
    for target, result in dataset:
        loss.append((result - model(params_w, target, bias)) ** 2)

    return np.mean(loss)

def finite_differences_method(dataset, w, bias):
    w_new = np.copy(w)
    w_new += eps
    loss_before = calculate_loss(model, dataset, w, bias)
    loss_after = calculate_loss(model, dataset, w_new, bias)

    L = (loss_after - loss_before) / eps
    return L

def finite_differences_method_bias(dataset, w, bias):
    loss_before = calculate_loss(model, dataset, w, bias)
    loss_after = calculate_loss(model, dataset, w, bias + eps)

    L_bias = (loss_after - loss_before) / eps
    return L_bias

def train_model_NAND(epochs, w, bias):
    dataset_NAND = create_dataset_NAND()

    for _ in range(epochs):
        L = finite_differences_method(dataset_NAND, w, bias)
        L_bias = finite_differences_method_bias(dataset_NAND, w, bias)
        w -= learning_rate * L
        bias -= learning_rate * L_bias

        loss = calculate_loss(model, dataset_NAND, w, bias)

    for input, _ in dataset_NAND:
        print(f'NAND - Input: {input}, Parameters: {w}, Loss: {loss}, Predicted value {model(w, input, bias)}')

def main():
    w1 = initialize_weight()
    w2 = initialize_weight()
    w = [w1, w2]
    np_w = np.array(w)

    bias = initialize_bias()

    epochs = 100000
    train_model_NAND(epochs, np_w, bias)


if __name__ == '__main__':
    main()