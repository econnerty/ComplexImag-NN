import jax.numpy as jnp
from jax import grad, hessian, jit
import jax
import numpy as np

# Define the complex exponential function
def complex_exp(a, b, c, d, x):
    # A = a + bi, B = c + di
    # f(x) = A * exp(B * x) = (a + bi) * exp((c + di) * x)
    return (a + b * 1j) * jnp.exp((c + d * 1j) * x)

# Mean squared error loss
def mse_loss(params, x, targets):
    a, b, c, d = params
    predictions = complex_exp(a, b, c, d, x)
    real_error = jnp.mean((jnp.real(predictions) - jnp.real(targets)) ** 2)
    imag_error = jnp.mean((jnp.imag(predictions) - jnp.imag(targets)) ** 2)
    return (real_error + imag_error) / 2

# Generate data
x = jnp.linspace(0, 2 * jnp.pi, 100)  # Input data
targets = complex_exp(1.0, 2.0, 0.1, 0.2, x)  # Generating targets using some arbitrary parameters

# Initialize parameters
params = jnp.array([1.0, 0.5, 0.1, 0.2])  # [a, b, c, d]

# Compute the Hessian
hessian_fn = jit(hessian(mse_loss))
computed_hessian = hessian_fn(params, x, targets)

print("Computed Hessian matrix:")
print(computed_hessian)

# Convert the JAX array to a NumPy array
hessian_matrix_np = np.array(computed_hessian)

# Compute eigenvalues
eigenvalues = np.linalg.eigvals(hessian_matrix_np)

# Check if all eigenvalues are non-negative
is_positive_semidefinite = np.all(eigenvalues >= 0)

print("Eigenvalues of the Hessian:", eigenvalues)
print("Is the Hessian matrix positive semidefinite?", is_positive_semidefinite)
