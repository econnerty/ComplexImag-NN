import jax.numpy as jnp
from jax import random, grad
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer

# Set device to CPU
jax.default_device(jax.devices('cpu')[0])

def complex_exp(x):
    """Complex exponential activation function."""
    return jnp.exp(x)

def complex_mse(pred, target):
    """Compute the mean squared error separately for the real and imaginary parts."""
    real_error = jnp.mean((jnp.real(pred) - jnp.real(target)) ** 2)
    imag_error = jnp.mean((jnp.imag(pred) - jnp.imag(target)) ** 2)
    return (real_error + imag_error) / 2

def initialize_params(layer_sizes, key):
    """Initialize weights for each layer in the network, excluding biases."""
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = random.split(key)
        real_weights = random.uniform(subkey, (n_in, n_out), maxval=1, minval=0) * 0
        imag_weights = random.uniform(subkey, (n_in, n_out), maxval=1, minval=0) * 1
        weights = real_weights + 1j * imag_weights
        params.append(weights)
    return params

def network(params, x):
    """A simple single-layer complex neural network without biases."""
    activations = x
    for w in params[:-1]:
        outputs = jnp.dot(activations, w)
        activations = complex_exp(outputs)
    final_w = params[-1]
    logits = jnp.dot(activations, final_w)
    return logits

# Update function modified to use Optax optimizer
def update(params, opt_state, x, y):
    """Update the parameters using an Optax optimizer."""
    def loss_fn(params):
        pred = network(params, x)
        return complex_mse(pred, y)
    
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)

    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Initialization
key = random.PRNGKey(2)
layer_sizes = [1, 1, 1]  # Input, hidden, and output sizes
params = initialize_params(layer_sizes, key)

# Create training data from unit circle
inputs = jnp.linspace(0, 2 * jnp.pi, 100).reshape(-1, 1)  # Reshape for batch processing
targets = jnp.exp((1j) * inputs).reshape(-1, 1)

# Create optimizer
optimizer = optax.adam(learning_rate=0.01)  # You can change the learning rate as needed
opt_state = optimizer.init(params)  # Initialize optimizer state

# Training loop
epochs = 10
for epoch in range(epochs):
    for x, y in zip(inputs, targets):
        params, opt_state = update(params, opt_state, x, y)
        pred = network(params, inputs)
        loss = complex_mse(pred, targets)
    print(f"Epoch {epoch}, Loss: {loss}")

# Inference
test_inputs = jnp.linspace(2, 4 * jnp.pi, 100).reshape(-1, 1)
test_targets = jnp.exp((1j) * test_inputs)
predictions = network(params, test_inputs)

# Plot the results
plt.figure(figsize=(8, 8))
plt.plot(jnp.real(test_targets).flatten(), jnp.imag(test_targets).flatten(), label='Actual')
plt.plot(jnp.real(predictions).flatten(), jnp.imag(predictions).flatten(), label='Predicted')
plt.legend()
plt.show()
