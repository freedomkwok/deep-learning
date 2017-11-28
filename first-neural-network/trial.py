from my_answers import NeuralNetwork
import numpy as np

def MSE(y, Y):
    return np.mean((y-Y)**2)

network = NeuralNetwork(3, 2, 1, 0.5)
inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

network.weights_input_to_hidden = test_w_i_h.copy()
network.weights_hidden_to_output = test_w_h_o.copy()


network.train(inputs, targets)

print(network.weights_input_to_hidden, np.allclose(network.weights_input_to_hidden,
                                                   np.array([[ 0.10562014, -0.20185996],
                                                             [0.39775194, 0.50074398],
                                                             [-0.29887597, 0.19962801]])))

print(network.weights_hidden_to_output)
print(np.array([[0.37275328],
                [-0.03172939]]))

# final_output = network.run(inputs)
# print(final_output, 0.09998924)

