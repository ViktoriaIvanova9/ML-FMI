import numpy as np

def create_dataset(n):
    dataset_results = []
    for i in range(0, n):
        dataset_results.append((i, i*2))

    np_ds_result = np.array(dataset_results)
    return np_ds_result

def initialize_weights(x, y):
    random_uf = np.random.uniform(x, y)
    return random_uf

def main():
    print(create_dataset(4))          # [(0, 0), (1, 2), (2, 4), (3, 6)]
    print(initialize_weights(0, 100)) # 95.07143064099162
    print(initialize_weights(0, 10))  # 3.745401188473625

if __name__ == '__main__':
    main()