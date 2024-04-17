import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

def complex_exp(x):
    return jnp.exp(x)

def complex_mse(pred, target):
    real_error = jnp.mean((jnp.real(pred) - jnp.real(target)) ** 2)
    imag_error = jnp.mean((jnp.imag(pred) - jnp.imag(target)) ** 2)
    return (real_error + imag_error) / 2

def network(params, x):
    activations = complex_exp(jnp.dot(x, params[0]))
    logits = jnp.dot(activations, params[1])
    return logits

def compute_loss(params, inputs, targets):
    preds = network(params, inputs)
    return complex_mse(preds, targets)

# Initialize Parameters
key = random.PRNGKey(0)
params = [random.normal(key, (1, 1)) + 1j * random.normal(key, (1, 1)),
          random.normal(key, (1, 1)) + 1j * random.normal(key, (1, 1))]

# Define input and target data
inputs = jnp.linspace(0, 2 * jnp.pi, 100).reshape(-1, 1)
targets = jnp.exp(1j * inputs).reshape(-1, 1)

# Define range for real and imaginary parts of the second weight
fixed_real_parts_of_w1 = [0, 0.5, 1.0]  # Fixed real part values of the first weight
w2_real = np.linspace(-10, 10, 100)
w2_imag = np.linspace(-10, 10, 100)

for w1_real_fixed in fixed_real_parts_of_w1:
    Loss = np.zeros((100, 100))
    for i, re in enumerate(w2_real):
        for j, im in enumerate(w2_imag):
            params[0] = jnp.array([[w1_real_fixed + 1j * 0.0]])  # Fixing the real part of W1 and setting imaginary to zero
            params[1] = jnp.array([[re + 1j * im]])
            Loss[j, i] = compute_loss(params, inputs, targets)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(w2_real, w2_imag, Loss, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f'Loss Contour for Fixed W1 Real={w1_real_fixed}')
    plt.xlabel('Real Part of Weight 2')
    plt.ylabel('Imaginary Part of Weight 2')
    plt.show()
