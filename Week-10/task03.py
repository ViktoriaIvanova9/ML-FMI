import numpy as np

def create_dataset(n):
    dataset_results = []
    for i in range(0, n):
        dataset_results.append((i, i*2))

    np_ds_result = np.array(dataset_results)
    return np_ds_result

def initialize_weight(x, y):
    np.random.seed(42)
    random_uf = np.random.uniform(x, y)
    return random_uf

np.random.seed(42)
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

def finite_differences_method():
    w = initialize_weight(0, 10)
    loss_before = calculate_loss(model, dataset, w)
    loss_after = calculate_loss(model, dataset, w + eps)

    L = (loss_after - loss_before) / eps
    return L

def main():
    w = initialize_weight(0, 10)
    for epoch in range(10):
        loss_before = calculate_loss(model, dataset, w)
        L = finite_differences_method()
        w -= learning_rate * L
        loss_after = calculate_loss(model, dataset, w)

        print(f'Epoch {epoch}: ')
        print(f'Loss before updating the parameter: {loss_before}')
        print(f'Loss after updating the parameter: {loss_after}')


if __name__ == '__main__':
    main()