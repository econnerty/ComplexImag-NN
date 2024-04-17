import numpy as np
import matplotlib.pyplot as plt

def initialize_weights(input_size, hidden_size, output_size):
    # Initialize weights as vectors (complex components)
    a = np.random.randn(hidden_size,output_size) * 0.01
    b = np.random.randn(hidden_size,output_size) * 0.01
    c = np.random.randn(input_size,hidden_size) * 0.01
    d = np.random.randn(input_size,hidden_size) * 0.01
    return a, b, c, d

def forward_pass(x, a, b, c, d):
    # c and d form the exponents; a and b scale the output
    u = np.exp(c * x) * np.cos(d * x)
    v = np.exp(c * x) * np.sin(d * x)
    y_hat = (a + b * 1j) * (u + v * 1j)
    return y_hat, u, v

def compute_gradients(x, y, y_hat, u, v, a, b, c, d):
    epsilon = y_hat - y
    grad_a = 2 * np.real(epsilon * np.conj(u + v * 1j))
    grad_b = 2 * np.real(epsilon * np.conj(1j * (u + v * 1j)))
    grad_c = 2 * np.real(epsilon * np.conj((a + b * 1j) * x * (u + v * 1j)))
    grad_d = 2 * np.real(epsilon * np.conj((a + b * 1j) * 1j * x * (u + v * 1j)))

    return np.array([grad_a, grad_b, grad_c, grad_d])

def update_weights(a, b, c, d, grads, learning_rate):
    a -= learning_rate * grads[0].sum(axis=1, keepdims=True)
    b -= learning_rate * grads[1].sum(axis=1, keepdims=True)
    c -= learning_rate * grads[2].sum(axis=0, keepdims=True)
    d -= learning_rate * grads[3].sum(axis=0, keepdims=True)
    return a, b, c, d

def train_network(X, Y, input_size, hidden_size, output_size, epochs=100, learning_rate=0.01):
    a, b, c, d = initialize_weights(input_size, hidden_size, output_size)
    #print(f'Initial weights: a={a}, b={b}, c={c}, d={d}')
    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(X, Y):
            y_hat, u, v = forward_pass(x, a, b, c, d)
            loss = np.sum(np.abs(y_hat - y)**2)
            total_loss += loss
            grads = compute_gradients(x, y, y_hat, u, v, a, b, c, d)
            a, b, c, d = update_weights(a, b, c, d, grads, learning_rate)
            #print(f'weights: a={a}, b={b}, c={c}, d={d}')
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(X)}")
    return a, b, c, d

# Data generation for training
X_train = np.linspace(0, 2 * np.pi, 100)
Y_train = np.exp((.1+1j) * X_train)

# Training the network
input_size = 1
hidden_size = 3  # this reflects each component as a vector/matrix
output_size = 1
epochs = 100
learning_rate = 0.001


a, b, c, d = train_network(X_train, Y_train, input_size, hidden_size, output_size, epochs, learning_rate)
#print(f'Final weights: a={a}, b={b}, c={c}, d={d}')

#Validation dataset
X_validation = np.linspace(0, 10 * np.pi, 100)
Y_validation = np.exp((.1+1j) * X_validation)

# Validation
predictions = [forward_pass(x, a, b, c, d)[0] for x in X_validation]

predictions = np.array(predictions).flatten()

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
