import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network model
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer with 10 neurons
        self.fc2 = nn.Linear(10, 10) # Additional hidden layer
        self.fc3 = nn.Linear(10, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Activation function Tanh
        x = torch.tanh(self.fc2(x))  # Activation function Tanh
        x = self.fc3(x)             # Linear output
        return x

# Training the network
def train_model(model, criterion, optimizer, X_train, Y_train, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Generate training data
X_train = torch.linspace(0, 2*np.pi, 100).reshape(-1, 1)
Y_train = torch.sin(X_train)

# Initialize the model, loss function, and optimizer
model = SineNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
train_model(model, criterion, optimizer, X_train, Y_train, epochs=2000)

# Test the model on a wider range
X_test = torch.linspace(0, 4*np.pi, 200).reshape(-1, 1)
Y_test = torch.sin(X_test)
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(X_train.numpy(), Y_train.numpy(), 'go', label='Training data')
plt.plot(X_test.numpy(), Y_test.numpy(), 'r-', label='True function')
plt.plot(X_test.numpy(), predictions.numpy(), 'b--', label='NN predictions')
plt.title('Neural Network Approximation of Sine Function')
plt.legend()
plt.show()
