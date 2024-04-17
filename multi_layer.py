import numpy as np
import matplotlib.pyplot as plt

class ComplexLayer:
    def __init__(self, units):
        self.units = units
        # Initialize complex weights more conservatively to ensure stability
        self.a = np.random.randn() #* 0.1
        self.b = np.random.randn() #* 0.1
        self.c = np.random.randn() #* 0.1
        self.d = np.random.randn() #* 0.1

        self.cache = {}

    def complex_tanh(self,z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def complex_tanh_derivative(self,z):
        return 1 - self.complex_tanh(z) ** 2

    def forward_pass(self, x):
        self.cache = {}  # Ensure cache is cleared and reset for each forward pass
        self.cache['x'] = x  # Cache the input x for use in the backward pass
        z = (self.c * x) + (self.d * 1j * x)
        activated_z = self.complex_tanh(z)
        self.cache['z'] = z  # Cache z for use in the derivative computation
        self.cache['activated_z'] = activated_z  # Optional: cache this if needed for more complex derivatives
        return (self.a + self.b * 1j) * activated_z

    def compute_gradients(self, grad_output):
        x = self.cache['x']
        z = self.cache['z']
        activated_z = self.cache['activated_z']
        # Compute the derivative of complex_tanh
        d_activated_z = self.complex_tanh_derivative(z) * grad_output
        
        # Compute gradients for each weight
        grad_a = np.real(d_activated_z * np.conj(activated_z))
        grad_b = np.real(d_activated_z * np.conj(1j * activated_z))
        grad_c = np.real(d_activated_z * np.conj((self.a + self.b * 1j) * x))
        grad_d = np.real(d_activated_z * np.conj((self.a + self.b * 1j) * 1j * x))

        # Compute gradient with respect to the input of the layer
        grad_input = np.real(d_activated_z * np.conj((self.a + self.b * 1j) * (self.c - self.d * 1j)))
        
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
        # The initial gradient based on the output error
        grad_output = 2 * (activations[-1] - y)
        
        # Loop through the layers in reverse to apply backpropagation
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # Receive gradients for input and weights
            grad_output, grad_a, grad_b, grad_c, grad_d = layer.compute_gradients(grad_output)
            # Update layer weights
            layer.update_weights((grad_a, grad_b, grad_c, grad_d), learning_rate)


    def train(self, X, Y, epochs=1000, learning_rate=0.01):
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
Y_train = np.exp(1j * X_train)

# Create and train the network
network = ComplexNeuralNetwork()
network.add_layer(1)  # Add a single unit layer
network.train(X_train, Y_train)

# Validation dataset
X_validation = np.linspace(0, 2 * np.pi, 100)
Y_validation = np.exp(1j * X_validation)

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