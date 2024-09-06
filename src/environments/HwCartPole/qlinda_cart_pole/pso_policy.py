"""Evaluate PSO policy."""
import pickle

import numpy as np


class PSOPolicy:
    """PSO policy."""

    def __init__(
        self,
        initial_weights,
        layers,
        neurons,
        input_size,
        output_size,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.neurons = neurons

        shapes = []
        shapes.append((input_size, neurons))
        shapes.append((neurons,))
        for _ in range(1, layers):
            shapes.append((neurons, neurons))
            shapes.append((neurons,))
        shapes.append((neurons, output_size))
        shapes.append((output_size,))

        self.weights = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            weight = initial_weights[idx : idx + size].reshape(shape)
            self.weights.append(weight)
            idx += size

    def get_action(self, state, return_nn_io=False):
        """Evaluate PSO policy."""
        dict_action = {}
        if len(state) > 0:
            stacked_state = np.zeros(shape=(len(state), self.input_size))
            for i, one_state in enumerate(state.items()):
                stacked_state[i, :] = one_state[1]
            nn_input = stacked_state
            next_input = nn_input
            for i in range(int(len(self.weights) / 2)):
                w_i = self.weights[i * 2]
                b_i = self.weights[i * 2 + 1]
                summed = np.zeros(shape=(len(next_input), len(b_i)))
                for j, bias in enumerate(b_i):
                    w_i_j = np.repeat(np.array([w_i[:, j]]), len(next_input), axis=0)
                    b_i_j = np.repeat(np.array([bias]), len(next_input), axis=0)
                    summed[:, j] = (
                        np.sum(np.multiply(w_i_j, next_input), axis=1) + b_i_j
                    )
                if i < int(len(self.weights) / 2) - 1:
                    next_input = np.maximum(summed, 0)
                else:
                    next_input = summed
            nn_output = next_input
            nn_output = np.clip(a=nn_output, a_min=-1.0, a_max=1.0)

            for i, one_state in enumerate(state.items()):
                dict_action[one_state[0]] = nn_output[i, :]

        if not return_nn_io:
            return dict_action
        return dict_action, nn_input, nn_output
