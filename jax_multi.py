import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer
import imageio
import os


os.environ['IMAGEIO_FFMPEG_EXE']= '/opt/homebrew/bin/ffmpeg'
def initialize_weights(layer_sizes, key):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)
        # Initialize weights
        real_w = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-0.1, maxval=0.1)
        imag_w = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-2.0, maxval=2.0)
        weights.append((real_w, imag_w))
        
        # Initialize biases
        key, subkey = random.split(key)
        real_b = random.uniform(subkey, (layer_sizes[i+1],), minval=-1.0, maxval=1.0)
        imag_b = random.uniform(subkey, (layer_sizes[i+1],), minval=-1.0, maxval=1.0)
        biases.append((real_b, imag_b))

    return weights, biases


@jit
def forward_pass(x, weights, biases):
    for (real, imag), (real_b, imag_b) in zip(weights[:-1], biases[:-1]):
        complex_weight = real + 1j * imag
        complex_bias = real_b + 1j * imag_b
        x = jnp.dot(x, complex_weight) + complex_bias
        x = jnp.exp(x)  # Using complex exponential as the activation function
    # Handle the final layer separately if needed
    real, imag = weights[-1]
    real_b, imag_b = biases[-1]
    complex_weight = real + 1j * imag
    complex_bias = real_b + 1j * imag_b
    return jnp.dot(x, complex_weight) + complex_bias


@jit
def loss(params, x, y):
    weights, biases = params
    y_hat = forward_pass(x, weights, biases)
    return jnp.mean(jnp.abs(y_hat - y) ** 2)

def update(params, opt_state, x, y, optimizer):
    def loss_fn(params):
        weights, biases = params
        return loss((weights, biases), x, y)
    
    grads = grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def train_network(X, Y, layer_sizes, epochs=10, learning_rate=0.001):
    key = random.PRNGKey(1)
    weights, biases = initialize_weights(layer_sizes, key)
    params = (weights, biases)  # Package parameters together
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        for x, y in zip(X, Y):
            params, opt_state = update(params, opt_state, x, y, optimizer)
            current_loss = loss(params, x, y)
        print(f"Epoch {epoch}, Loss: {current_loss}")

    return params


def inference(params, X):
    weights, biases = params
    predict = vmap(lambda x: forward_pass(x, weights, biases))
    return predict(X)

# Set device to CPU
jax.default_device(jax.devices('cpu')[0])

# Define the layer sizes of the network
layer_sizes = [1, 10, 1]  # Example: 1 input, two hidden layers with 10 neurons each, 1 output

# Data generation for training
X_train = jnp.linspace(0, 2 * jnp.pi, 1000).reshape(-1, 1)
Y_train = jnp.exp((.01+1j) * X_train) + (.01+1j) * jnp.exp((-.01+2j) * X_train) + 2

# Train the network
params = train_network(X_train, Y_train, layer_sizes)

# Validation dataset
X_validation = jnp.linspace(0, 2 * jnp.pi, 1000).reshape(-1, 1)
Y_validation = jnp.exp((.01+1j) * X_train) + (.01+1j) * jnp.exp((-.01+2j) * X_train) + 2

# Inference on the validation dataset
predictions = inference(params, X_validation)
print(params)

# Plotting results for validation
plt.figure(figsize=(8, 8))
plt.plot(jnp.real(Y_validation), jnp.imag(Y_validation), 'ro', label='Actual')
plt.plot(jnp.real(predictions), jnp.imag(predictions), 'bx', label='Predicted')
plt.title('Validation: Actual vs Predicted Values on the Unit Circle')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
