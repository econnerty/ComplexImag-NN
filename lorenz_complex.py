import numpy as np
from ComplexExpNN import ComplexEXPNN
import matplotlib.pyplot as plt
import pickle
import os
import imageio

os.environ['IMAGEIO_FFMPEG_EXE']= '/opt/homebrew/bin/ffmpeg'

def create_video(filenames, output_file='complex_lorenz.mp4'):
    with imageio.get_writer(output_file, fps=8) as writer:
        for filename in filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)
        # Keep the last frame for a bit longer
        for _ in range(10):
            writer.append_data(image)

def generate_lorenz_data(steps, dt=0.01, sigma=10, beta=8/3, rho=28):
    xs = np.zeros(steps + 1)
    ys = np.zeros(steps + 1)
    zs = np.zeros(steps + 1)
    xs[0], ys[0], zs[0] = 0., 1., 1.05  # Initial conditions

    for i in range(steps):
        x_dot = sigma * (ys[i] - xs[i])
        y_dot = xs[i] * (rho - zs[i]) - ys[i]
        z_dot = xs[i] * ys[i] - beta * zs[i]
        xs[i + 1] = xs[i] + x_dot * dt
        ys[i + 1] = ys[i] + y_dot * dt
        zs[i + 1] = zs[i] + z_dot * dt

    return xs, ys, zs

# Generate Lorenz data
xs, ys, zs = generate_lorenz_data(15000)
train_length = 7500
complex_data = xs[:train_length] + 1j * ys[:train_length]
z_data = zs[:train_length] + 1j * np.zeros_like(zs[:train_length])  # z is real; imaginary part is zero

# Prepare the training data
X_train = np.stack([complex_data[:-1], z_data[:-1]], axis=-1)  # stack along last dimension
Y_train = np.stack([complex_data[1:], z_data[1:]], axis=-1)

#Min max normalization
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
Y_train = (Y_train - Y_train.min()) / (Y_train.max() - Y_train.min())

#Plot the data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.real(X_train[:,0]), np.imag(X_train[:,0]), np.real(X_train[:,1]), lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#Later size
layer_sizes = [2, 4, 4, 2] 

# Create the neural network instance
nn = ComplexEXPNN(layer_sizes=layer_sizes, jax_key=0)

# Load the parameters if they exist
filenames = []
try:
    with open('params.pkl', 'rb') as file:
        nn.params = pickle.load(file)
except FileNotFoundError:
    # Train the model if no parameters file exists
    filenames = nn.train(X_train, Y_train, epochs=250, learning_rate=0.07,batch_size=64)
    # Save the parameters after training
    with open('params.pkl', 'wb') as file:
        pickle.dump(nn.params, file)

create_video(filenames)

# Generate the test data
complex_data_test = xs[train_length:] + 1j * ys[train_length:]
z_data_test = zs[train_length:] + 1j * np.zeros_like(zs[train_length:])  # z is real; imaginary part is zero

# Prepare the training data
X_test= np.stack([complex_data_test[:-1], z_data_test[:-1]], axis=-1)  # stack along last dimension
Y_test = np.stack([complex_data_test[1:], z_data_test[1:]], axis=-1)

#Min max normalization
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
Y_test = (Y_test - Y_test.min()) / (Y_test.max() - Y_test.min())

print(X_test.shape, Y_test.shape)

#Run the inference
predictions = nn.inference(X_test)
print(predictions.shape)

#Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.real(Y_test[:,0]), np.imag(Y_test[:,0]), np.real(Y_test[:,1]), lw=0.5, label='Actual')
ax.plot(np.real(predictions[:,0]), np.imag(predictions[:,0]), np.real(predictions[:,1]), lw=0.5, label='Predicted')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()

#Print the complex MSE of predictions vs actual data
mse = np.mean(np.abs(predictions - Y_test) ** 2)
print(f"Complex MSE (test): {mse}")




