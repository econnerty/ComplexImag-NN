import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer
import imageio
import os
from scipy.integrate import solve_ivp
from ComplexExpNN import ComplexEXPNN



# Define the layer sizes of the network
layer_sizes = [1,3, 1]  # Example: 1 input, two hidden layers with 10 neurons each, 1 output
#18 Params

# Data generation for training
# Define the Lotka-Volterra model
def lotka_volterra(t, z, alpha, beta, gamma, delta):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Simulate the Lotka-Volterra model
alpha, beta, gamma, delta = 1.3, 0.02, 0.3, 0.01
#alpha, beta, gamma, delta = 1.8, 0.001, .1, .0005
initial_conditions = [50, 20]
t_span = [0, 40]
t_eval = np.linspace(t_span[0], t_span[1], 500)
solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, gamma, delta), t_eval=t_eval)

#Prepare the train data
complex_data = solution.y[0] + 1j * solution.y[1]
X_train = complex_data[:-1]
Y_train = complex_data[1:]  # Prey as real part, Predator as imaginary part

#Min max normalization
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
Y_train = (Y_train - Y_train.min()) / (Y_train.max() - Y_train.min())

X_test = X_train
Y_test = Y_train
#Shuffle the data
#idx = jnp.arange(X_train.shape[0])
#idx = random.permutation(random.PRNGKey(0), idx)
#X_train = X_train[idx]
#Y_train = Y_train[idx]

# Train the network
model = ComplexEXPNN(layer_sizes, real_weight_scale=0.1, jax_key=0)
model.train(X_train, Y_train)
# Generate the video from saved plots


# Prepare the test data
initial_conditions = [20, 10]
t_span = [40, 70]
t_eval = np.linspace(t_span[0], t_span[1], 500)
solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, gamma, delta), t_eval=t_eval)

#Prepare the train data
complex_data = solution.y[0] + 1j * solution.y[1]
X_train = complex_data[:-1]
Y_train = complex_data[1:]  # Prey as real part, Predator as imaginary part

#Min max normalization
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
Y_train = (Y_train - Y_train.min()) / (Y_train.max() - Y_train.min())

#Plot predictions
predictions = model.inference(X_train)
plt.figure(figsize=(12, 6))
plt.plot(np.real(Y_train), label='Actual Real', color='red')
plt.plot(np.real(predictions[:,0,0]), label='Predicted Real', color='blue')
plt.plot(np.imag(Y_train), label='Actual Imaginary', color='green')
plt.plot(np.imag(predictions[:,0,0]), label='Predicted Imaginary', color='yellow')
plt.title('Real Parts Evolution')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#Print the complex MSE of predictions vs actual data
mse = jnp.mean(jnp.abs(predictions - Y_train) ** 2)
print(f"Complex MSE (test): {mse}")




