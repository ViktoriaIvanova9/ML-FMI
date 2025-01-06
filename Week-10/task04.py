import numpy as np

def create_dataset(n):
    dataset_results = []
    for i in range(0, n):
        dataset_results.append((i, i*2))

    np_ds_result = np.array(dataset_results)
    return np_ds_result

def initialize_weight(x, y):
    random_uf = np.random.uniform(x, y)
    return random_uf

learning_rate = 0.001
eps = 0.001
dataset = create_dataset(6)

def model(param, x):
    return x * param

def calculate_loss(model, dataset, param_w):
    loss = []
    for x, y in dataset:
        loss.append((y - model(param_w, x)) ** 2)

    return np.mean(loss)

def finite_differences_method(w):
    loss_before = calculate_loss(model, dataset, w)
    loss_after = calculate_loss(model, dataset, w + eps)

    L = (loss_after - loss_before) / eps
    return L

def train_model(epochs, w):
    for _ in range(epochs):
        L = finite_differences_method(w)
        w -= learning_rate * L
        loss = calculate_loss(model, dataset, w)

        print(f'Parameter value: {w}')
        print(f'Claculated loss: {loss}')

def main():
    w = initialize_weight(0, 10)
    train_model(500, w)

if __name__ == '__main__':
    main()