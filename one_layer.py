import numpy as np
import matplotlib.pyplot as plt

class ComplexLayer:
    def __init__(self, units):
        self.units = units
        # Initialize weights for each unit (a, b, c, d)
        self.a = np.random.randn()
        self.b = np.random.randn()
        self.c = np.random.randn()
        self.d = np.random.randn()
        # For storing intermediate values for backpropagation
        self.cache = {}

    def complex_tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def forward_pass(self, x):
        self.cache['x'] = x
        u = np.exp(self.c * x) * np.cos(self.d * x) 
        v = np.exp(self.c * x) * np.sin(self.d * x)  
        output = (self.a + self.b * 1j) * (u + v * 1j)
        self.cache['u'] = u
        self.cache['v'] = v
        return output

    def compute_gradients(self, grad_output):
        x, u, v = self.cache['x'], self.cache['u'], self.cache['v']
        epsilon = grad_output
        grad_a = 2 * np.real(epsilon * np.conj(u + v * 1j))
        grad_b = 2 * np.real(epsilon * np.conj(1j * (u + v * 1j)))
        grad_c = 2 * np.real(epsilon * np.conj((self.a + self.b * 1j) * x * (u + v * 1j)))
        grad_d = 2 * np.real(epsilon * np.conj((self.a + self.b * 1j) * 1j * x * (u + v * 1j)))
        grad_input = 2 * np.real(epsilon * np.conj((self.a + self.b * 1j) * (self.c - self.d * 1j)))
        return grad_input, grad_a, grad_b, grad_c, grad_d

    def update_weights(self, grads, learning_rate):
        grad_a, grad_b, grad_c, grad_d = grads
        self.a -= learning_rate * grad_a
        self.b -= learning_rate * grad_b
        self.c -= learning_rate * grad_c
        self.d -= learning_rate * grad_d

class ComplexNeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, units):
        self.layers.append(ComplexLayer(units))

    def forward_pass(self, x):
        activations = [x]
        for layer in self.layers:
            x = layer.forward_pass(x)
            activations.append(x)
        return activations

    def backward_pass(self, activations, y, learning_rate):
        # Start from the output layer
        grad_output = 2 * (activations[-1] - y)  # Initial gradient (dL/dy)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            grad_output, grad_a, grad_b, grad_c, grad_d = layer.compute_gradients(grad_output)
            layer.update_weights((grad_a, grad_b, grad_c, grad_d), learning_rate)
            # grad_output is now the gradient for the next layer's output

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            Loss = 0
            for x, y in zip(X, Y):
                activations = self.forward_pass(x)
                y_hat = activations[-1]
                loss = np.abs(y_hat - y) ** 2
                Loss += loss
                self.backward_pass(activations, y, learning_rate)
            print(f"Epoch {epoch + 1}, Loss: {Loss / len(X)}")

    def inference(self, X):
        predictions = [self.forward_pass(x)[-1] for x in X]
        return predictions

# Data generation for training
X_train = np.linspace(0, 2 * np.pi, 100)
Y_train = (.1)*np.exp((.1+1j) * X_train)

# Create and train the network
network = ComplexNeuralNetwork()
network.add_layer(1)  # Add a single unit layer
network.train(X_train, Y_train)

# Validation dataset
X_validation = np.linspace(0, 20 * np.pi, 1000)
Y_validation = (.1)*np.exp((.1+1j) * X_validation)

# Inference on the validation dataset
predictions = network.inference(X_validation)

#Print the weights
print("Weights of the network:")
for i, layer in enumerate(network.layers):
    print(f"Layer {i + 1}: a={layer.a}, b={layer.b}, c={layer.c}, d={layer.d}")

# Plotting results for validation
plt.figure(figsize=(8, 8))
plt.plot(np.real(Y_validation), np.imag(Y_validation), 'ro', label='Actual')
plt.plot(np.real(predictions), np.imag(predictions), 'bx', label='Predicted')
plt.title('Validation: Actual vs Predicted Values on the Unit Circle')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()