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

    """def initialize_weights(self):
        weights = []
        biases = []
        for i in range(len(self.layer_sizes) - 1):
            self.key, subkey = random.split(self.key)

            real_w = random.uniform(subkey, (self.layer_sizes[i], self.layer_sizes[i+1]), minval=-1, maxval=1) * self.real_weight_scale
            imag_w = random.uniform(subkey, (self.layer_sizes[i], self.layer_sizes[i+1]), minval=-3.0, maxval=3.0)
            
            weights.append((real_w, imag_w))
            self.key, subkey = random.split(self.key)
            real_b = random.uniform(subkey, (self.layer_sizes[i+1],), minval=-1, maxval=1) * self.real_weight_scale
            imag_b = random.uniform(subkey, (self.layer_sizes[i+1],), minval=-3.0, maxval=3.0)
            biases.append((real_b, imag_b))

        self.params = (weights, biases)"""
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
            x = self.complex_relu(x)  # Using complex sigmoid as the activation for subsequent layers

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

    def train(self, X, Y, epochs=100, learning_rate=0.001):
        self.initialize_weights()
        optimizer = optax.sgd(learning_rate,momentum=0.9)
        self.opt_state = optimizer.init(self.params)

        for epoch in range(epochs):
            losses = []
            for x, y in zip(X, Y):
                self.params, self.opt_state = self.update(self.params, self.opt_state, x, y, optimizer)
                current_loss = self.loss(self.params, x, y)
                losses.append(current_loss)
            print(f"Epoch {epoch+1}, Loss: {jnp.mean(jnp.array(losses))}")
        return self.params

    def update(self, params, opt_state, x, y, optimizer):
        grads = grad(self.loss)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def inference(self, X):
        predict = vmap(lambda x: self.forward_pass(x, *self.params))
        return predict(X)
    
    def iterative_inference(self, initial_input, num_steps):
        """Generate outputs iteratively using the model's prediction as the next input."""
        predictions = []
        current_input = initial_input
        for _ in range(num_steps):
            # Predict the next step
            next_output = self.forward_pass(current_input, *self.params)
            # Use the predicted output as the next input
            predictions.append(next_output)
            current_input = next_output
        return jnp.array(predictions)

    
    @partial(jit, static_argnums=(0,))
    def complex_sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    @partial(jit, static_argnums=(0,))
    def complex_relu(self, x):
        return jnp.maximum(0, jnp.real(x)) + 1j*jnp.maximum(0, np.imag(x))
