import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
eps = 0.001

def sigmoid(result):
    return (1 / (1 + np.exp(-result)))

def main():
    x = np.linspace(-10, 10)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.title('Sigmoid function representation')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()