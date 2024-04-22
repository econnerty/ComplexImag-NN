import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax

class ComplexEXPNN:
    def __init__(self, layer_sizes, real_weight_scale=1.0,jax_key=0):
        self.layer_sizes = layer_sizes
        self.real_weight_scale = real_weight_scale
        self.params = None
        self.opt_state = None
        self.key = random.PRNGKey(jax_key)  # Initialize a random key for JAX

    def initialize_weights(self):
        weights = []
        biases = []
        for i in range(len(self.layer_sizes) - 1):
            self.key, subkey = random.split(self.key)

            real_w = random.uniform(subkey, (self.layer_sizes[i], self.layer_sizes[i+1]), minval=-1, maxval=1) * self.real_weight_scale
            imag_w = random.uniform(subkey, (self.layer_sizes[i], self.layer_sizes[i+1]), minval=-2.0, maxval=2.0)
            
            weights.append((real_w, imag_w))
            self.key, subkey = random.split(self.key)
            real_b = random.uniform(subkey, (self.layer_sizes[i+1],), minval=-1, maxval=1) * self.real_weight_scale
            imag_b = random.uniform(subkey, (self.layer_sizes[i+1],), minval=-2.0, maxval=2.0)
            biases.append((real_b, imag_b))

        self.params = (weights, biases)

    def forward_pass(self,x, weights, biases):
        for (real, imag), (real_b, imag_b) in zip(weights[:-1], biases[:-1]):
            complex_weight = real + 1j * imag
            complex_bias = real_b + 1j * imag_b
            x = jnp.dot(x, complex_weight) + complex_bias
            x = 0 + jnp.imag(x)*1j
            x = jnp.exp(x)  # Using complex exponential as the activation
        #Final layer
        real, imag = weights[-1]
        real_b, imag_b = biases[-1]
        complex_weight = real + 1j * imag
        complex_bias = real_b + 1j * imag_b
        return jnp.dot(x, complex_weight) + complex_bias

    def loss(self, params, x, y):
        weights, biases = params
        y_hat = self.forward_pass(x, weights, biases)
        return jnp.mean(jnp.abs(y_hat - y) ** 2)

    def train(self, X, Y, epochs=100, learning_rate=0.001):
        self.initialize_weights()
        optimizer = optax.adam(learning_rate)
        self.opt_state = optimizer.init(self.params)

        for epoch in range(epochs):
            losses = []
            for x, y in zip(X, Y):
                self.params, self.opt_state = self.update(self.params, self.opt_state, x, y, optimizer)
                current_loss = self.loss(self.params, x, y)
                losses.append(current_loss)
            print(f"Epoch {epoch}, Loss: {jnp.mean(jnp.array(losses))}")

    def update(self, params, opt_state, x, y, optimizer):
        grads = grad(self.loss)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def inference(self, X):
        predict = vmap(lambda x: self.forward_pass(x, *self.params))
        return predict(X)
