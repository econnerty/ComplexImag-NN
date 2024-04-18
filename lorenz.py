import jax.numpy as jnp
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer

def lorenz(t, state, sigma=10, rho=28, beta=2.667):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def encode_complex(t_values, x_values, y_values, z_values):
    x_complex = t_values + 1j * x_values
    y_complex = t_values + 1j * y_values
    z_complex = t_values + 1j * z_values
    return x_complex, y_complex, z_complex


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


t_span = [0, 25]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
initial_state = [1.0, 1.0, 1.0]

solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
t_values = solution.t

# Encoding the states with time as the real part
X_train, Y_train, Z_train = encode_complex(t_values, solution.y[0], solution.y[1], solution.y[2])

#Group the inputs
X_train = jnp.column_stack([t_values[:-1], X_train[:-1], Y_train[:-1], Z_train[:-1]])
Y_train = jnp.column_stack([t_values[1:], X_train[1:], Y_train[1:], Z_train[1:]])

print(X_train.shape, Y_train.shape)

# Set device to CPU
jax.default_device(jax.devices('cpu')[0])
layer_sizes = [3, 10, 10, 3]  # Input layer with 3 neurons, two hidden layers with 10 neurons each, output layer with 3 neurons

# Use existing training infrastructure
params = train_network(X_train, Y_train, layer_sizes)

# Validation dataset
X_validation = solution.y.T[:-1]
predictions = inference(params, X_validation)

# Plotting real and imaginary parts of predictions
plt.figure(figsize=(10, 6))
plt.plot(X_validation[:, 0], label='True X')
plt.plot([p[0].real for p in predictions], label='Predicted X', linestyle='--')
plt.plot([p[1].real for p in predictions], label='Predicted Y', linestyle='--')
plt.plot([p[2].real for p in predictions], label='Predicted Z', linestyle='--')
plt.title('Lorenz System Prediction')
plt.xlabel('Time Steps')
plt.ylabel('State Variables')
plt.legend()
plt.show()
