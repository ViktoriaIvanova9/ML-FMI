import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

def log_loss(raw_model_output):
    return np.log(1 + np.exp(-raw_model_output))

def hinge_loss(raw_model_output):
    return np.maximum(0, 1 - raw_model_output)

def main():
    raw_model_output = np.linspace(-2, 2, 1000)

    logloss = log_loss(raw_model_output)
    hingeloss = hinge_loss(raw_model_output)

    plt.plot(raw_model_output, logloss)
    plt.plot(raw_model_output, hingeloss) # don't understand the idea of hinge loss

    plt.legend(['logistic', 'hinge'])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()