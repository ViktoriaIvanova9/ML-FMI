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

def model(params, x):
    dot_product = np.dot(params, x)
    return sigmoid(dot_product)

def calculate_loss(model, dataset, params_w):
    loss = []
    for target, result in dataset:
        loss.append((result - model(params_w, target)) ** 2)

    return np.mean(loss)

def finite_differences_method(dataset, w, i):
    w_new = np.copy(w)
    w_new[i] += eps
    loss_before = calculate_loss(model, dataset, w)
    loss_after = calculate_loss(model, dataset, w_new)

    L = (loss_after - loss_before) / eps
    return L

def train_model_AND(epochs, w):
    dataset_AND = create_dataset_AND()

    for _ in range(epochs):
        for i in range(len(w)):
            L = finite_differences_method(dataset_AND, w, i)
            w[i] -= learning_rate * L

            loss = calculate_loss(model, dataset_AND, w)

    for input, _ in dataset_AND:
        print(f'AND - Input: {input}, Parameters: {w}, Loss: {loss}, Predicted value: {model(w, input)}')

def train_model_OR(epochs, w):
    dataset_OR = create_dataset_OR()

    for _ in range(epochs):
        for i in range(len(w)):
            L = finite_differences_method(dataset_OR, w, i)
            w[i] -= learning_rate * L

        loss = calculate_loss(model, dataset_OR, w)

    for input, _ in dataset_OR:
        print(f'OR - Input: {input}, Parameters: {w}, Loss: {loss}, Predicted value {model(w, input)}')

def main():
    w1 = initialize_weight()
    w2 = initialize_weight()
    w = [w1, w2]
    np_w = np.array(w)

    epochs = 100000
    train_model_AND(epochs, np_w)
    train_model_OR(epochs, np_w)

if __name__ == '__main__':
    main()

    # What do you notice about the confidence the model has in them? - it is quessing them pretty good,
    # when I expect 1 it is above 0.7 and when I expect 0 it is below 0.2. Only for (0, 0) it is 0.5