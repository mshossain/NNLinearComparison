import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
length = 10  # length of the cylinder along the x-axis
r = 1       # radius of the cylinder
A = 2       # amplitude of the sine wave for the z-axis modulation
f = 2       # frequency of the sine wave for the z-axis modulation
angle = 25  # angle for the linear upward trend in degrees

# Convert angle to radians
angle_rad = np.deg2rad(angle)

# Generate data
num_points = 2000
x = np.linspace(0, length, num_points)
theta = np.linspace(0, 2 * np.pi, num_points)

# Generate points along the cylinder
z_center = A * np.sin(f * x) + x * np.tan(angle_rad)
x_points = []
y_points = []
z_points = []

for i in range(num_points):
    x_value = x[i]
    z_value = z_center[i]
    x_points.append(x_value)
    y_points.append(r * np.cos(theta[i]))
    z_points.append(z_value + r * np.sin(theta[i]))

# Create DataFrame
data = pd.DataFrame(
    {'x': x_points, 
     'y': y_points, 
     'z': z_points})

# Visualize the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x'], data['y'], data['z'], c=data['x'], cmap='viridis', marker='o', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Save the DataFrame to a CSV file
file_path = 'sinusoidal_cylindrical_pipe_with_trend.csv'
data.to_csv(file_path, index=False)

print(f"Data saved to {file_path}")
