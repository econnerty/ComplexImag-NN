import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer

def initialize_weights(layer_sizes, key):
    weights = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)
        # Real and imaginary parts initialized separately
        real = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-0.1, maxval=0.1)
        imag = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-0.1, maxval=0.1)
        weights.append((real, imag))
    return weights

@jit
def forward_pass(x, weights):
    # Handle complex arithmetic manually
    for real, imag in weights[:-1]:
        complex_weight = real + 1j * imag
        x = complex_weight * x
        x = jnp.exp(x)  # Using complex exponential as the activation function
    real, imag = weights[-1]
    complex_weight = real + 1j * imag
    return jnp.sum(complex_weight * x)

@jit
def loss(params, x, y):
    y_hat = forward_pass(x, params)
    return jnp.mean(jnp.abs(y_hat - y) ** 2)

def update(params, opt_state, x, y, optimizer):
    """Update the parameters using an Optax optimizer."""
    def loss_fn(params):
        return loss(params, x, y)
    grads = grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def train_network(X, Y, layer_sizes, epochs=100, learning_rate=0.0001):
    key = random.PRNGKey(0)
    params = initialize_weights(layer_sizes, key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        for x, y in zip(X, Y):
            params, opt_state = update(params, opt_state, x, y, optimizer)
            current_loss = loss(params, x, y)
        print(f"Epoch {epoch}, Loss: {current_loss}")

    return params

def inference(params, X):
    predict = vmap(lambda x: forward_pass(x, params))
    return predict(X)

# Set device to CPU
jax.default_device(jax.devices('cpu')[0])

# Define the layer sizes of the network
layer_sizes = [1, 10, 1]  # Example: 1 input, two hidden layers with 10 neurons each, 1 output

# Data generation for training
X_train = jnp.linspace(0, 2 * jnp.pi, 1000)
Y_train = jnp.exp((.01+1j) * X_train)

# Train the network
params = train_network(X_train, Y_train, layer_sizes)

# Validation dataset
X_validation = jnp.linspace(10, 12 * jnp.pi, 1000)
Y_validation = jnp.exp((.01+1j) * X_validation)

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
