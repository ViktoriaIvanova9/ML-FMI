import numpy as np

np.random.seed(42)

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

def model(param, x):
    return x * param

w = initialize_weight(0, 10)

def calculate_loss(model, dataset):
    loss = []
    for x, y in dataset:
        loss.append((y - model(w, x)) ** 2)

    return np.mean(loss)

def main():
    dataset = create_dataset(6)
    loss = calculate_loss(model, dataset)
    print(f'MSE: {loss}') # MSE: 27.92556532998047

if __name__ == '__main__':
    main()

    model_parameters = [w + 0.001 * 2, w + 0.001, w - 0.001, w - 0.001 * 2]
    # what happens to loss function when you pass w + 0.001 * 2, w + 0.001, w - 0.001 and w - 0.001 * 2? -
    # the loss is 
    # MSE: 27.92556532998047 - w
    # MSE: 27.989600040224502 - w + 0.001 * 2
    # MSE: 27.957573518435822 - w + 0.001
    # MSE: 27.893575474858455 - w - 0.001
    # MSE: 27.86160395306978 - w - 0.001 * 2
    # We see that for bigger parameter the loss is higher


    