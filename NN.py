# Author: Shahriar Hossain

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = pd.read_csv(
    'sinusoidal_cylindrical_pipe_with_trend.csv')


# Split the data into training and testing sets
train_data, test_data = train_test_split(
    data, test_size=0.3, random_state=42)

# Prepare the training and testing data
X_train = torch.tensor(
    train_data[['x', 'y']].values, 
    dtype=torch.float32)
y_train = torch.tensor(
    train_data['z'].values, 
    dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(
    test_data[['x', 'y']].values, 
    dtype=torch.float32)
y_test = test_data['z'].values


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Create the model, define the loss 
# function and the optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict on the test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()


# Create subplots
fig = plt.figure(figsize=(14, 6))

# Subplot for Neural Network
ax1 = fig.add_subplot(121, projection='3d')
#ax1.scatter(train_data['x'], train_data['y'], train_data['z'], c='blue', marker='o', label='Training Data')
ax1.scatter(test_data['x'], test_data['y'], test_data['z'], c='blue', marker='o', label='Test Data')

ax1.scatter(test_data['x'], test_data['y'], y_pred, c='red', marker='^', label='Predicted Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Neural Network Prediction')
ax1.legend()

# Linear Regression to predict y given x and z
X_train_lr = train_data[['x', 'y']].values
y_train_lr = train_data['z'].values
X_test_lr = test_data[['x', 'y']].values
y_test_lr = test_data['z'].values


from sklearn.linear_model import LinearRegression

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# Predict on the test data
y_pred_lr = lr_model.predict(X_test_lr)


# Subplot for Linear Regression
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(test_data['x'], 
            test_data['y'], 
            test_data['z'], c='blue', 
            marker='o', label='Test Data')
ax2.scatter(test_data['x'], 
            test_data['y'], 
            y_pred_lr, c='red', 
            marker='^', 
            label='Predicted Data')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Linear Regression Prediction')
ax2.legend()

plt.show()


# In[ ]:




