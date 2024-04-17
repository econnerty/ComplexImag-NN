import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

np.random.seed(0)

os.environ['IMAGEIO_FFMPEG_EXE']= '/opt/homebrew/bin/ffmpeg'

def initialize_weights():
    return np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()

def forward_pass(x, a, b, c, d):
    complex_weight = a + b * 1j
    complex_exponent = c + d * 1j
    
    exp_term = np.exp(complex_exponent * x)

    return complex_weight * exp_term

def compute_gradients(x, y, y_hat, a, b, c, d):
    u = np.exp(c * x) * np.cos(d * x)
    v = np.exp(c * x) * np.sin(d * x)
    epsilon = y_hat - y
    grad_a = 2 * np.real(epsilon * np.conj(u + v * 1j))
    grad_b = 2 * np.real(epsilon * np.conj(1j * (u + v * 1j)))
    grad_c = 2 * np.real(epsilon * np.conj((a + b * 1j) * x * (u + v * 1j)))
    grad_d = 2 * np.real(epsilon * np.conj((a + b * 1j) * 1j * x * (u + v * 1j)))

    #Clip gradients to avoid exploding gradients
    grad_a = np.clip(grad_a, -200, 200)
    grad_b = np.clip(grad_b, -200, 200)
    grad_c = np.clip(grad_c, -200, 200)
    grad_d = np.clip(grad_d, -200, 200)
    return grad_a, grad_b, grad_c, grad_d

def update_weights(a, b, c, d, grads, learning_rate):
    a -= learning_rate * grads[0]
    b -= learning_rate * grads[1]
    c -= learning_rate * grads[2]
    d -= learning_rate * grads[3]
    return a, b, c, d

def train_and_plot(X_train, Y_train, X_val, Y_val, epochs=100, learning_rate=0.01):
    a, b, c, d = initialize_weights()
    filenames = []
    
    for epoch in range(epochs):
        Loss = 0
        for x, y in zip(X_train, Y_train):
            y_hat = forward_pass(x, a, b, c, d)
            loss = np.abs(y_hat - y) ** 2
            Loss += loss
            grads = compute_gradients(x, y, y_hat, a, b, c, d)
            a, b, c, d = update_weights(a, b, c, d, grads, learning_rate)
        
        # Evaluate on the validation dataset
        predictions_val = [forward_pass(x, a, b, c, d) for x in X_val]

        #Round a b c d
        a_r = round(a, 4)
        b_r = round(b, 4)
        c_r = round(c, 4)
        d_r = round(d, 4)
        
        # Generate and save a plot for each epoch
        plt.figure(figsize=(8, 8))
        plt.scatter(np.real(Y_val), np.imag(Y_val), color='red', label='Actual')
        plt.scatter(np.real(predictions_val), np.imag(predictions_val), color='blue', label='Predicted')
        plt.title(r'Validation - Epoch {}, Training Loss: {:.4f}, $({}+{}i)e^{{({}+{}i)x}}$'.format(
            epoch + 1, Loss / len(X_train), a_r, b_r, c_r, d_r), fontsize=12)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        filename = f'./out/plot_validation_epoch_{epoch+1}.png'
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)
    
    return a, b, c, d, filenames

def inference(a, b, c, d, X):
    predictions = [forward_pass(x, a, b, c, d) for x in X]
    return predictions

def create_video(image_filenames, output_file='validation_progress.mp4'):
    writer = imageio.get_writer(output_file, format='FFMPEG', mode='I', fps=5)
    try:
        for filename in image_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        # Let the last frame linger for a bit by repeating it
        for _ in range(10):  # Repeat the last image 10 times
            writer.append_data(image)
    finally:
        writer.close()

# Data generation for training
X_train = np.linspace(0, 2 * np.pi, 100)
Y_train = (.01+.2j)*np.exp((.1+1j) * X_train)

#Shuffle the training data together
combined = list(zip(X_train, Y_train))
np.random.shuffle(combined)
X_train[:], Y_train[:] = zip(*combined)


# Validation dataset
X_validation = np.linspace(2, 25 * np.pi, 1000)
Y_validation = (.01+.2j)*np.exp((.1+1j) * X_validation)

# Train the network
a, b, c, d,filenames = train_and_plot(X_train, Y_train, X_validation, Y_validation, epochs=100, learning_rate=0.005)

create_video(filenames,'imaginary_exp.mp4')
# Inference on the validation dataset
"""predictions = inference(a, b, c, d, X_validation)

# Plotting results for validation
plt.figure(figsize=(8, 8))
plt.plot(np.real(Y_validation), np.imag(Y_validation), 'ro', label='Actual')
plt.plot(np.real(predictions), np.imag(predictions), 'bx', label='Predicted')
plt.title('Validation: Actual vs Predicted Values on the Unit Circle')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
"""