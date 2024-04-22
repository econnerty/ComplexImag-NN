import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax
import imageio
import os
from scipy.integrate import solve_ivp

def initialize_weights(layer_sizes, key):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)
        # Xavier/Glorot initialization for weights
        std_dev = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
        weights.append(random.normal(subkey, (layer_sizes[i], layer_sizes[i+1])) * std_dev)
        biases.append(np.zeros((layer_sizes[i+1],)))  # Initialize biases to zero
    return weights, biases

def relu(x):
    return jnp.maximum(0, x)

@jit
def forward_pass(x, weights, biases):
    for w, b in zip(weights[:-1], biases[:-1]):
        x = jnp.dot(x, w) + b
        x = relu(x)  # Activation function: tanh
    return jnp.dot(x, weights[-1]) + biases[-1]

@jit
def loss(params, x, y):
    weights, biases = params
    y_hat = forward_pass(x, weights, biases)
    return jnp.mean((y_hat - y) ** 2)

def update(params, opt_state, x, y, optimizer):
    grads = grad(lambda params: loss(params, x, y))(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def create_video(filenames, output_file='real_lotka_training.mp4'):
    with imageio.get_writer(output_file, fps=8) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        for _ in range(10):  # Repeat last frame
            writer.append_data(image)

def train_network(X, Y, layer_sizes, epochs=100, learning_rate=0.0001):
    key = random.PRNGKey(0)
    weights, biases = initialize_weights(layer_sizes, key)
    params = (weights, biases)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    os.makedirs('./real', exist_ok=True)
    filenames = []

    for epoch in range(epochs):
        for x, y in zip(X, Y):
            params, opt_state = update(params, opt_state, x, y, optimizer)
            current_loss = loss(params, x, y).item()
        
        # Plot training progress
        weights, biases = params
        plt.figure()
        predictions = vmap(lambda x: forward_pass(x, weights, biases))(X)
        plt.plot(Y, label='Actual Data')
        plt.plot(predictions, label='Predicted Data')
        plt.title(f'Epoch {epoch+1} Loss: {current_loss}')
        plt.xlabel('Data Point')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        filename = f'./real/plot_epoch_{epoch+1}.png'
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)
        
        print(f"Epoch {epoch+1}, Loss: {current_loss}")

    create_video(filenames)
    return params

# Define the Lotka-Volterra model
def lotka_volterra(t, z, alpha, beta, gamma, delta):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Simulate the Lotka-Volterra model
# Parameters for the Lotka-Volterra model
alpha, beta, gamma, delta = 1.3, 0.02, 0.3, 0.01
initial_conditions = [50, 20]
t_span = [0, 40]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the differential equations
solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, gamma, delta), t_eval=t_eval)

# Prepare the training data
X_train = solution.y.T[:-1]  # Exclude the last timestep for inputs
Y_train = solution.y.T[1:]   # Exclude the first timestep for outputs

# Normalize the data
min_val = X_train.min(axis=0)
max_val = X_train.max(axis=0)

X_train = (X_train - min_val) / (max_val - min_val)
Y_train = (Y_train - min_val) / (max_val - min_val)

# Set the testing data
X_test = X_train
Y_test = Y_train


# Example data preparation and network training code
# layer_sizes = [input_dimension, hidden_layer_size, output_dimension]
# X_train, Y_train = prepare your data here
# params = train_network(X_train, Y_train, layer_sizes)
# Define the layer sizes of the network
layer_sizes = [2, 4, 2]  # Example: 1 input, one hidden layer with 10 neurons, 1 output
#20 Parameters?


# Train the network
params = train_network(X_train, Y_train, layer_sizes)

# Prepare the test data
initial_conditions = [20, 10]
t_span = [40, 70]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the differential equations
solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, gamma, delta), t_eval=t_eval)

# Prepare the training data
X_train = solution.y.T[:-1]  # Exclude the last timestep for inputs
Y_train = solution.y.T[1:]   # Exclude the first timestep for outputs

# Normalize the data
min_val = X_train.min(axis=0)
max_val = X_train.max(axis=0)

X_train = (X_train - min_val) / (max_val - min_val)
Y_train = (Y_train - min_val) / (max_val - min_val)

#Plot the results
weights, biases = params
predictions = vmap(lambda x: forward_pass(x, weights, biases))(X_train)
plt.plot(Y_train, label='Actual Data')
plt.plot(predictions, label='Predicted Data')
plt.title('Lotka-Volterra Model')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#Print the MSE of predictions vs actual data
mse = jnp.mean((predictions - Y_train) ** 2)
print(f'Mean Squared Error (test): {mse}')
