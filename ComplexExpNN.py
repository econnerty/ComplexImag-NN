from functools import partial
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
import numpy as np
import matplotlib.pyplot as plt

class ComplexEXPNN:
    def __init__(self, layer_sizes,jax_key=0):
        self.layer_sizes = layer_sizes
        #self.real_weight_scale = real_weight_scale
        self.params = None
        self.opt_state = None
        self.key = random.PRNGKey(jax_key)  # Initialize a random key for JAX
        self.filenames = []

    def initialize_weights(self):
        weights = []
        biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i+1]
            std_dev = jnp.sqrt(2.0 / (fan_in + fan_out))  # He initialization adapted for complex numbers
            key, subkey = random.split(self.key)
            real_w = random.normal(subkey, (fan_in, fan_out)) * std_dev
            key, subkey = random.split(key)
            imag_w = random.normal(subkey, (fan_in, fan_out)) * std_dev #* scale  # Scale down imaginary part

            weights.append((real_w, imag_w))

            # Initialize biases to zero or small random values
            real_b = jnp.zeros((fan_out,))
            imag_b = jnp.zeros((fan_out,))
            biases.append((real_b,imag_b))

        self.params = (weights, biases)

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, x, weights, biases):
        # Process the first layer separately
        real, imag = weights[0]
        real_b, imag_b = biases[0]
        complex_weight = real + 1j * imag
        complex_bias = real_b + 1j * imag_b
        x = jnp.dot(x, complex_weight) + complex_bias
        x = jnp.exp(x)  # Complex exponential activation for the first layer

        for (real, imag), (real_b, imag_b) in zip(weights[1:-1], biases[1:-1]):
            complex_weight = real + 1j * imag
            complex_bias = real_b + 1j * imag_b
            x = jnp.dot(x, complex_weight) + complex_bias
            x = self.complex_sigmoid(x)  # Using complex sigmoid as the activation for subsequent layers

        # Final layer processing
        real, imag = weights[-1]
        real_b, imag_b = biases[-1]
        complex_weight = real + 1j * imag
        complex_bias = real_b + 1j * imag_b
        x = jnp.dot(x, complex_weight) + complex_bias

        return x

    @partial(jit, static_argnums=(0,))
    def loss(self, params, x, y):
        weights, biases = params
        y_hat = self.forward_pass(x, weights, biases)
        return jnp.mean(jnp.abs(y_hat - y) ** 2)

    def train(self, X, Y, epochs=100, learning_rate=0.001, batch_size=32):
        self.initialize_weights()
        optimizer = optax.sgd(learning_rate,momentum=0.9)
        self.opt_state = optimizer.init(self.params)

        num_batches = X.shape[0] // batch_size
        for epoch in range(epochs):
            perm = np.random.permutation(X.shape[0])
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            losses = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                x_batch = X_shuffled[start_idx:end_idx]
                y_batch = Y_shuffled[start_idx:end_idx]
                self.params, self.opt_state = self.update(self.params, self.opt_state, x_batch, y_batch, optimizer)
                current_loss = self.loss(self.params, x_batch, y_batch)
                losses.append(current_loss)
            print(f"Epoch {epoch+1}, Loss: {jnp.mean(jnp.array(losses))}")
            # Generate and save plot for this epoch
            predictions = self.inference(X)

            #Plot the results
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(np.real(Y[:,0]), np.imag(Y[:,0]), np.real(Y[:,1]), lw=0.5, label='Actual')
            ax.plot(np.real(predictions[:,0]), np.imag(predictions[:,0]), np.real(predictions[:,1]), lw=0.5, label='Predicted')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.legend()
            plt.grid(True)
            filename = f'./lorenz_complex/plot_epoch_{epoch+1}.png'
            plt.savefig(filename)  # Ensure the saved figure retains the background color
            plt.close()
            self.filenames.append(filename)
        return self.filenames
            

    def update(self, params, opt_state, x, y, optimizer):
        grads = grad(self.loss)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def inference(self, X):
        predict = vmap(lambda x: self.forward_pass(x, *self.params))
        return predict(X)

    @partial(jit, static_argnums=(0,))
    def complex_sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    @partial(jit, static_argnums=(0,))
    def complex_relu(self, x):
        return jnp.maximum(0, jnp.real(x)) + 1j*jnp.maximum(0, np.imag(x))
