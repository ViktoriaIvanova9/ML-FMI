import numpy as np

learning_rate = 0.001
eps = 0.001

def sigmoid(result):
    return (1 / (1 + np.exp(-result)))

def create_dataset_AND():
    return [((0, 0), 0),((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]

def create_dataset_OR():
    return [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]

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

def finite_differences_method(dataset, w, bias, i):
    w_new = np.copy(w)
    w_new[i] += eps
    loss_before = calculate_loss(model, dataset, w, bias)
    loss_after = calculate_loss(model, dataset, w_new, bias)

    L = (loss_after - loss_before) / eps
    return L

def finite_differences_method_bias(dataset, w, bias):
    loss_before = calculate_loss(model, dataset, w, bias)
    loss_after = calculate_loss(model, dataset, w, bias + eps)

    L_bias = (loss_after - loss_before) / eps
    return L_bias

def train_model_AND(epochs, w, bias):
    dataset_AND = create_dataset_AND()

    for _ in range(epochs):
        for i in range(len(w)):
            L = finite_differences_method(dataset_AND, w, bias, i)
            L_bias = finite_differences_method_bias(dataset_AND, w, bias)
            w[i] -= learning_rate * L
            bias -= learning_rate * L_bias

            loss = calculate_loss(model, dataset_AND, w, bias)

    for input, _ in dataset_AND:
        print(f'AND - Input: {input}, Parameters: {w}, Loss: {loss}, Predicted value {model(w, input, bias)}')

def train_model_OR(epochs, w, bias):
    dataset_OR = create_dataset_OR()

    for _ in range(epochs):
        for i in range(len(w)):
            L = finite_differences_method(dataset_OR, w, bias, i)
            L_bias = finite_differences_method_bias(dataset_OR, w, bias)
            w[i] -= learning_rate * L
            bias -= learning_rate * L_bias

            loss = calculate_loss(model, dataset_OR, w, bias)

    for input, _ in dataset_OR:
        print(f'OR - Input: {input}, Parameters: {w}, Loss: {loss}, Predicted value {model(w, input, bias)}')

def main():
    w1 = initialize_weight()
    w2 = initialize_weight()
    w = [w1, w2]
    np_w = np.array(w)

    bias = initialize_bias()

    epochs = 100000
    train_model_AND(epochs, np_w, bias)
    train_model_OR(epochs, np_w, bias)

if __name__ == '__main__':
    main()

    # What has changed in comparison to task 5? - when you give 2 zeros, it becomes closer to 0