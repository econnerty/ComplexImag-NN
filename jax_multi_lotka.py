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

        real_w = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-.1, maxval=.1)
        imag_w = random.uniform(subkey, (layer_sizes[i], layer_sizes[i+1]), minval=-2.0, maxval=2.0)
        
        weights.append((real_w, imag_w))
        key, subkey = random.split(key)
        real_b = random.uniform(subkey, (layer_sizes[i+1],), minval=-.1, maxval=.1)
        imag_b = random.uniform(subkey, (layer_sizes[i+1],), minval=-2.0, maxval=2.0)
        biases.append((real_b, imag_b))

    return weights, biases

"""def initialize_weights(layer_sizes, key):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)

        # Calculate the standard deviation for Xavier/Glorot initialization
        std_dev = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
        
        # Initialize real and imaginary parts of the weights
        real_w = random.normal(subkey, (layer_sizes[i], layer_sizes[i+1])) * std_dev
        imag_w = random.normal(subkey, (layer_sizes[i], layer_sizes[i+1])) * std_dev

        weights.append((real_w, imag_w))
        
        key, subkey = random.split(key)

        # Initialize real and imaginary parts of the biases (could be initialized to zero)
        real_b = np.zeros((layer_sizes[i+1],))
        imag_b = np.zeros((layer_sizes[i+1],))
        
        biases.append((real_b, imag_b))

    return weights, biases"""


def complex_relu(z):
    return jnp.maximum(0, jnp.real(z)) + 1j*jnp.maximum(0, jnp.imag(z))

def complex_sigmoid(z):
    return 1 / (1 + jnp.exp(-z))
def complex_tanh(z):
    return jnp.tanh(z)

"""@jit
def forward_pass(x, weights, biases):
    for (real, imag), (real_b, imag_b) in zip(weights[:-1], biases[:-1]):
        #complex_weight = real + 1j * imag
        #complex_bias = real_b + 1j * imag_b
        complex_weight = real + 1j * imag
        complex_bias = real_b + 1j * imag_b
        x = jnp.dot(x, complex_weight) + complex_bias
        x = 0 + jnp.imag(x)*1j
        x = jnp.exp(x)  # Using complex exponential as the activation function
    # Handle the final layer separately if needed
    real, imag = weights[-1]
    real_b, imag_b = biases[-1]
    complex_weight = real + 1j * imag
    complex_bias = real_b + 1j * imag_b
    return jnp.dot(x, complex_weight) + complex_bias"""

@jit
def forward_pass(x, weights, biases):
    first_layer = True
    for (real, imag), (real_b, imag_b) in zip(weights[:-1], biases[:-1]):
        complex_weight = real + 1j * imag
        complex_bias = real_b + 1j * imag_b
        x = jnp.dot(x, complex_weight) + complex_bias
        
        if first_layer:
            x = 0 + jnp.imag(x)*1j
            x = jnp.exp(x)  # Using complex exponential as the activation
            first_layer = True
        else:
            x = complex_tanh(x)  # Using complex relu as the activation function for subsequent layers
    
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

@jit
def component_wise_loss(params, x, y):
    weights, biases = params
    y_hat = forward_pass(x, weights, biases)
    # Separate real and imaginary parts
    real_error = jnp.abs(jnp.real(y_hat) - jnp.real(y)) ** 2
    imag_error = jnp.abs(jnp.imag(y_hat) - jnp.imag(y)) ** 2
    # Compute mean separately and then average them (or sum, depending on the use case)
    mean_real_error = jnp.mean(real_error)
    mean_imag_error = jnp.mean(imag_error)
    return (mean_real_error + mean_imag_error)


def update(params, opt_state, x, y, optimizer):
    #def loss_fn(params):
    #    weights, biases = params
    #    return loss((weights, biases), x, y)
    def loss_fn(params):
        return loss(params, x, y)
    
    
    grads = grad(loss_fn)(params)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def train_network(X, Y, layer_sizes, X_test,Y_test,epochs=100, learning_rate=0.001):
    key = random.PRNGKey(0)
    weights, biases = initialize_weights(layer_sizes, key)
    params = (weights, biases)

    #optimizer = optax.adam(learning_rate)
    #AMSGRAD
    optimizer = optax.chain(
    #optax.clip_by_global_norm(1.0),  # Add gradient clipping here as part of the optimizer chain
    optax.sgd(learning_rate,momentum=0.9),
)
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
        plt.plot(np.real(predictions[:,0,0]), label='Predicted Real', color='blue')
        plt.plot(np.imag(Y_test), label='Actual Imaginary', color='green')
        plt.plot(np.imag(predictions[:,0,0]), label='Predicted Imaginary', color='yellow')
        plt.title('Real Parts Evolution', color='white')
        plt.xlabel('Epoch', color='white')
        plt.ylabel('Real Value', color='white')
        plt.legend()
        plt.grid(True)
        """# Generate and save plot for this epoch with a dark background
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
params,filenames = train_network(X_train, Y_train, layer_sizes,X_test,Y_test)
# Generate the video from saved plots
#Get all files from the out directory
"""filenames = [f'./out/{f}' for f in os.listdir('./out') if f.endswith('.png')]
#Sort the files based on the epoch number
filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))"""
create_video(filenames, 'complex_neural_net_lotka.mp4')

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
predictions = inference(params, X_train)
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




