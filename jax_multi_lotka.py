import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer
import imageio
import os
from scipy.integrate import solve_ivp


os.environ['IMAGEIO_FFMPEG_EXE']= '/opt/homebrew/bin/ffmpeg'

def create_video(filenames, output_file='complex_neural_network.mp4'):
    with imageio.get_writer(output_file, fps=5) as writer:
        for filename in filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)
        # Keep the last frame for a bit longer
        for _ in range(10):
            writer.append_data(image)

def initialize_weights(layer_sizes, key):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)

        real_w = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-0.01, maxval=0.01)
        imag_w = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-5.0, maxval=5.0)
        
        weights.append((real_w, imag_w))
        key, subkey = random.split(key)
        real_b = random.uniform(subkey, (layer_sizes[i+1],), minval=-.5, maxval=.5)
        imag_b = random.uniform(subkey, (layer_sizes[i+1],), minval=-5.0, maxval=5.0)
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


def train_network(X, Y, layer_sizes, X_test,Y_test,epochs=100, learning_rate=0.003):
    key = random.PRNGKey(3)
    weights, biases = initialize_weights(layer_sizes, key)
    params = (weights, biases)

    #optimizer = optax.adam(learning_rate)
    #AMSGRAD
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Prepare directory for output images
    os.makedirs('./out', exist_ok=True)
    filenames = []

    for epoch in range(epochs):
        for x, y in zip(X, Y):
            params, opt_state = update(params, opt_state, x, y, optimizer)
            current_loss = loss(params, x, y).item()
        print(f"Epoch {epoch+1}, Loss: {current_loss}")
        # Generate and save plot for this epoch
        predictions = inference(params, X_test)
        # Prepare figure
        plt.figure(figsize=(12, 6), facecolor='black')

        # Plot Real parts
        plt.plot(np.real(Y_test), label='Actual Real', color='red')
        plt.plot(np.real(predictions), label='Predicted Real', color='blue')
        plt.plot(np.imag(Y_test), label='Actual Imaginary', color='green')
        plt.plot(np.imag(predictions), label='Predicted Imaginary', color='yellow')
        plt.title('Real Parts Evolution', color='white')
        plt.xlabel('Epoch', color='white')
        plt.ylabel('Real Value', color='white')
        plt.legend()
        plt.grid(True)
        # Generate and save plot for this epoch with a dark background
        """plt.figure(figsize=(8, 8), facecolor='black')  # Set facecolor to 'black'
        plt.scatter(np.real(Y), np.imag(Y), color='red', label='Actual')
        plt.scatter(np.real(predictions), np.imag(predictions), color='blue', label='Predicted')
        plt.title(f'Complex Exp NN, Epoch {epoch+1}, Loss: {current_loss}', color='white')  # Set text color to white for visibility
        plt.xlabel('Real Part', color='white')
        plt.ylabel('Imaginary Part', color='white')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        ax = plt.gca()  # Get current axes to change axis color
        ax.set_facecolor('black')  # Set the plot background to black
        ax.spines['bottom'].set_color('white')  # Set x-axis line color to white
        ax.spines['left'].set_color('white')  # Set y-axis line color to white
        ax.tick_params(axis='x', colors='white')  # Set x-axis tick colors to white
        ax.tick_params(axis='y', colors='white')  # Set y-axis tick colors to white"""
        filename = f'./out/plot_epoch_{epoch+1}.png'
        plt.savefig(filename)  # Ensure the saved figure retains the background color
        plt.close()
        filenames.append(filename)

    return params, filenames



def inference(params, X):
    weights, biases = params
    predict = vmap(lambda x: forward_pass(x, weights, biases))
    return predict(X)

# Set device to CPU
jax.default_device(jax.devices('cpu')[0])

# Define the layer sizes of the network
layer_sizes = [1,10,10, 1]  # Example: 1 input, two hidden layers with 10 neurons each, 1 output

# Data generation for training
# Define the Lotka-Volterra model
def lotka_volterra(t, z, alpha, beta, gamma, delta):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Simulate the Lotka-Volterra model
alpha, beta, gamma, delta = 0.3, 0.02, 0.3, 0.01
#alpha, beta, gamma, delta = 1.8, 0.001, .1, .0005
initial_conditions = [50, 20]
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, gamma, delta), t_eval=t_eval)

#Prepare the train data
X_train = jnp.array(solution.t).reshape(-1, 1)
Y_train = solution.y[0] + 1j * solution.y[1]  # Prey as real part, Predator as imaginary part

#Min max normalization
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
Y_train = (Y_train - Y_train.min()) / (Y_train.max() - Y_train.min())

X_test = X_train
Y_test = Y_train
#Shuffle the data
idx = jnp.arange(X_train.shape[0])
idx = random.permutation(random.PRNGKey(0), idx)
X_train = X_train[idx]
Y_train = Y_train[idx]

# Train the network
params,filenames = train_network(X_train, Y_train, layer_sizes,X_test,Y_test)
# Generate the video from saved plots
#Get all files from the out directory
"""filenames = [f'./out/{f}' for f in os.listdir('./out') if f.endswith('.png')]
#Sort the files based on the epoch number
filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))"""
create_video(filenames, 'complex_neural_net_lotka.mp4')
"""# Validation dataset
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
"""