import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer

def initialize_weights(key):
    keys = random.split(key, 4)
    a = random.uniform(keys[0], minval=0, maxval=1)
    b = random.uniform(keys[1], minval=0, maxval=1)
    c = random.uniform(keys[2], minval=0, maxval=1)
    d = random.uniform(keys[3], minval=0, maxval=1)
    return [a, b, c, d]

@jit
def forward_pass(x, params):
    a, b, c, d = params
    complex_weight = a + b * 1j
    complex_exponent = c + d * 1j
    exp_term = jnp.exp(complex_exponent * x)
    return complex_weight * exp_term

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

def train_network(X, Y, epochs=100, learning_rate=0.007):
    key = random.PRNGKey(0)
    params = initialize_weights(key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        for x, y in zip(X, Y):
            params, opt_state = update(params, opt_state, x, y, optimizer)
            current_loss = loss(params, X, Y)
        print(f"Epoch {epoch}, Loss: {current_loss}")

    return params

def inference(params, X):
    predict = vmap(lambda x: forward_pass(x, params))
    return predict(X)

# Set device to CPU
jax.default_device(jax.devices('cpu')[0])

# Data generation for training
X_train = jnp.linspace(0, 2 * jnp.pi, 100)
Y_train = jnp.exp((.01+1j) * X_train)

# Train the network
params = train_network(X_train, Y_train)

# Validation dataset
X_validation = jnp.linspace(2, 25 * jnp.pi, 1000)
Y_validation = jnp.exp((.01+1j) * X_validation)

# Inference on the validation dataset
predictions = inference(params, X_validation)

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
