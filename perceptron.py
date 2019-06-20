import random


class Perceptron:
    def __init__(self, input_number, step_size=0.1):
        self._ins = input_number
        # random weights
        self._w = [random.random() for _ in range(input_number)]
        self._eta = step_size  # convergence rate

    def predict(self, inputs):
        # Dot product of inputs and weights
        weighted_average = sum(w * elm for w, elm in zip(self._w, inputs))
        if weighted_average > 0:
            return 1
        return 0

    def train(self, inputs, expected_output):
        output = self.predict(inputs)
        error = expected_output - output
        if error != 0:
            self._w = [w + self._eta * error * x for w, x in
                       zip(self._w, inputs)]
        return error
