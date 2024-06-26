import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax
import numpy as np
import matplotlib.pyplot as plt
import optax  # Import Optax for the optimizer
import imageio
import os
import bayex


os.environ['IMAGEIO_FFMPEG_EXE']= '/opt/homebrew/bin/ffmpeg'

def create_video(filenames, output_file='complex_neural_network.mp4'):
    with imageio.get_writer(output_file, fps=8) as writer:
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


def train_network(X, Y, layer_sizes, epochs=400, learning_rate=0.001):
    key = random.PRNGKey(1)
    weights, biases = initialize_weights(layer_sizes, key)
    params = (weights, biases)
    
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
        predictions = inference(params, X)
        # Generate and save plot for this epoch with a dark background
        plt.figure(figsize=(8, 8), facecolor='black')  # Set facecolor to 'black'
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
        ax.tick_params(axis='y', colors='white')  # Set y-axis tick colors to white
        filename = f'./out/plot_epoch_{epoch+1}.png'
        plt.savefig(filename, facecolor=ax.figure.get_facecolor())  # Ensure the saved figure retains the background color
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
layer_sizes = [1, 10, 1]  # Example: 1 input, two hidden layers with 10 neurons each, 1 output

# Data generation for training
X_train = jnp.linspace(0, 2 * jnp.pi, 1000).reshape(-1, 1)
Y_train = jnp.exp((.01+1j) * X_train) + (.01+1j) * jnp.exp((-.01+2j) * X_train) + 2


# Train the network
params,filenames = train_network(X_train, Y_train, layer_sizes)
# Generate the video from saved plots
#Get all files from the out directory
"""filenames = [f'./out/{f}' for f in os.listdir('./out') if f.endswith('.png')]
#Sort the files based on the epoch number
filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))"""
create_video(filenames, 'complex_neural_net_2.mp4')
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