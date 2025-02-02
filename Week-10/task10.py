import numpy as np

learning_rate = 0.001
eps = 0.001
epochs = 100000

class Xor:
    def __init__(self, dataset):
        self.dataset = dataset

        self.weights_hidden = np.random.uniform(-1, 1, 4)
        self.biases_hidden = np.random.uniform(-1, 1, 2)
        self.bias = np.random.uniform(-1, 1)

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def model(self, x):
        hidden_layer_dot_product = np.dot(self.weights_hidden, x)
        hidden_output = self.sigmoid(hidden_layer_dot_product + self.biases_hidden)

        dot_product = np.dot(self.weights, hidden_output)
        return self.sigmoid(dot_product + self.bias)

    def calculate_loss(self, model, current_weights, current_bias):
        loss_hidden = []
        for target, result in self.dataset:
            loss_hidden.append((result - model(current_weights, target, current_bias)) ** 2)

        return np.mean(loss_hidden)

    def finite_differences_method(self, dataset, w, bias):
        w_new = np.copy(w)
        w_new += eps
        loss_before = self.calculate_loss(self.model, dataset, w, bias)
        loss_after = self.calculate_loss(self.model, dataset, w_new, bias)

        L = (loss_after - loss_before) / eps
        return L

    def finite_differences_method_bias(self, dataset, w, bias):
        loss_before = self.calculate_loss(self.model, dataset, w, bias)
        loss_after = self.calculate_loss(self.model, dataset, w, bias + eps)

        L_bias = (loss_after - loss_before) / eps
        return L_bias

    def result_from_OR(self, dataset_OR, weights_to_use, bias_to_use):
        for _ in range(epochs):
            L = self.finite_differences_method(dataset_OR, weights_to_use, bias_to_use)
            L_bias = self.finite_differences_method_bias(dataset_OR, weights_to_use, bias_to_use)
            weights_to_use -= learning_rate * L
            bias_to_use -= learning_rate * L_bias

            loss = self.calculate_loss(self.model, dataset_OR, weights_to_use, bias_to_use)

        for input, _ in dataset_OR:
            prediction = self.model(weights_to_use, input, bias_to_use)

        return prediction

    def result_from_AND(self, dataset_AND, weights_to_use, bias_to_use):
        for _ in range(epochs):
            L = self.finite_differences_method(dataset_AND, weights_to_use, bias_to_use)
            L_bias = self.finite_differences_method_bias(dataset_AND, weights_to_use, bias_to_use)
            weights_to_use -= learning_rate * L
            bias_to_use -= learning_rate * L_bias

            loss = self.calculate_loss(self.model, dataset_AND, weights_to_use, bias_to_use)

        for input, _ in dataset_AND:
            prediction = self.model(weights_to_use, input, bias_to_use)

        return prediction
        
    def result_from_NAND(self, dataset_NAND, weights_to_use, bias_to_use):
        for _ in range(epochs):
            L = self.finite_differences_method(dataset_NAND, weights_to_use, bias_to_use)
            L_bias = self.finite_differences_method_bias(dataset_NAND, weights_to_use, bias_to_use)
            weights_to_use -= learning_rate * L
            bias_to_use -= learning_rate * L_bias

            loss = self.calculate_loss(self.model, dataset_NAND, weights_to_use, bias_to_use)

        for input, _ in dataset_NAND:
            prediction = self.model(weights_to_use, input, bias_to_use)

        return prediction

    def train_model_XOR(self):
        pass

    def forward(self, model, input_one, input_two):
        dataset_OR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
        dataset_NAND = [((0, 0), 1),((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
        dataset_AND = [((0, 0), 0),((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]

        output_OR = self.result_from_OR(dataset_OR, )
        output_NAND = self.result_from_NAND(dataset_NAND, )


        result_from_XOR = self.result_from_AND(dataset_AND, output_OR, output_NAND, self.bias)
        return result_from_XOR

if __name__ == '__main__':
    xor_dataset = [((0, 0), 0),((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

    xor_nn = Xor(xor_dataset)
    xor_nn.train_model_XOR()

    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for curr_input in inputs:
        result = xor_nn.forward(curr_input)
        print(f"Input: {curr_input}, Predicted Output: {result}")
