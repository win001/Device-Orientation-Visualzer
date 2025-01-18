import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from collections import deque

from flask import Flask, render_template
from flask_socketio import SocketIO
import queue
import json
import socket as socket

import threading
import time
import random

app = Flask(__name__)
socketio = SocketIO(app)
motion_data_queue = queue.Queue()

class GyroVisualizer:
    def __init__(self, space_limit=10):
        # Initialize the 3D plot
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set space limits
        self.space_limit = space_limit
        self.ax.set_xlim([-space_limit, space_limit])
        self.ax.set_ylim([-space_limit, space_limit])
        self.ax.set_zlim([-space_limit, space_limit])

        self.ax.set_xlabel('Alpha (Z-axis)')
        self.ax.set_ylabel('Beta (X-axis)')
        self.ax.set_zlabel('Gamma (Y-axis)')
        
        # Initialize position and smoothing buffers
        self.position = np.array([0.0, 0.0, 0.0])
        self.window_size = 10
        self.x_buffer = deque(maxlen=self.window_size)
        self.y_buffer = deque(maxlen=self.window_size)
        self.z_buffer = deque(maxlen=self.window_size)
        
        # Initialize the point
        self.point, = self.ax.plot([self.position[0]], [self.position[1]], 
                                 [self.position[2]], 'ro', markersize=10)
        
        # Trail effect
        self.trail_length = 50
        self.trail_x = deque(maxlen=self.trail_length)
        self.trail_y = deque(maxlen=self.trail_length)
        self.trail_z = deque(maxlen=self.trail_length)
        self.trail, = self.ax.plot([], [], [], 'b-', alpha=0.3)

    def get_random_gyro_data(self):
        """Generate random gyro data within actual gyro ranges"""

        if not motion_data_queue.empty():
            data = motion_data_queue.get()
            print(f"Dequeued data: {data}")
            alpha, beta, gamma = data.get('alpha', 0), data.get('beta', 0), data.get('gamma', 0)
            return np.array([alpha, beta, gamma])

        # x = np.random.uniform(0, 360)     # [0, 360] for x
        # y = np.random.uniform(-180, 180)  # [-180, 180] for y
        # z = np.random.uniform(-90, 90)    # [-90, 90] for z     
        # return np.array([x, y, z])
        return np.array([0, 0, 0])

    def normalize_angle(self, angle, min_val, max_val):
        """Normalize angle to handle wraparound"""
        range_size = max_val - min_val
        while angle > max_val:
            angle -= range_size
        while angle < min_val:
            angle += range_size
        return angle

    def angular_mean(self, angles, min_val, max_val):
        """Calculate mean of angular values considering wraparound"""
        if not angles:
            return 0
        
        # Convert angles to radians and calculate mean using circular statistics
        range_size = max_val - min_val
        angles_rad = [2 * np.pi * (a - min_val) / range_size for a in angles]
        
        # Calculate mean of sine and cosine components
        sin_sum = np.mean([np.sin(a) for a in angles_rad])
        cos_sum = np.mean([np.cos(a) for a in angles_rad])
        
        # Convert back to original range
        mean_rad = np.arctan2(sin_sum, cos_sum)
        mean_angle = min_val + (mean_rad * range_size / (2 * np.pi))
        
        return self.normalize_angle(mean_angle, min_val, max_val)

    def smooth_data(self, new_data):
        """Apply moving average smoothing to the angular data"""
        # Add new data to buffers
        self.x_buffer.append(new_data[0])
        self.y_buffer.append(new_data[1])
        self.z_buffer.append(new_data[2])
        
        # Calculate smoothed angles considering their respective ranges
        x_smooth = self.angular_mean(self.x_buffer, 0, 360)
        y_smooth = self.angular_mean(self.y_buffer, -180, 180)
        z_smooth = self.angular_mean(self.z_buffer, -90, 90)
        
        return np.array([x_smooth, y_smooth, z_smooth])

    def angles_to_cartesian(self, angles):
        """Convert angular coordinates to cartesian coordinates"""
        x_rad = np.radians(angles[0])
        y_rad = np.radians(angles[1])
        z_rad = np.radians(angles[2])
        
        # Convert to cartesian coordinates (normalized to space_limit)
        x = self.space_limit * np.cos(y_rad) * np.cos(x_rad)
        y = self.space_limit * np.cos(y_rad) * np.sin(x_rad)
        z = self.space_limit * np.sin(z_rad)
        
        return np.array([x, y, z])

    def update_position(self, angles):
        """Update position using angular coordinates"""
        # Convert angles to cartesian coordinates
        self.position = self.angles_to_cartesian(angles)
        
        # Update trail
        self.trail_x.append(self.position[0])
        self.trail_y.append(self.position[1])
        self.trail_z.append(self.position[2])

    def animate(self, frame):
        """Animation update function"""
        # Get and smooth gyro data
        raw_gyro = self.get_random_gyro_data()
        smooth_angles = self.smooth_data(raw_gyro)
        
        # Update position
        self.update_position(smooth_angles)
        
        # Update point position
        self.point.set_data([self.position[0]], [self.position[1]])
        self.point.set_3d_properties([self.position[2]])
        
        # Update trail
        self.trail.set_data(list(self.trail_x), list(self.trail_y))
        self.trail.set_3d_properties(list(self.trail_z))
        
        return self.point, self.trail

    def run(self):
        """Run the visualization"""
        ani = FuncAnimation(self.fig, self.animate, frames=None, 
                          interval=50, blit=True)
        plt.show()

@app.route('/')
def index():
    return render_template('index.html')  # Serve the webpage with JavaScript to capture motion data

@socketio.on('motion_data')
def handle_motion_data(data):
    if(motion_data_queue.qsize() == 0):
        motion_data_queue.put(data)  # Store the received motion data
        print(f"Received motion data: {motion_data_queue.qsize()}, {data}")

def runserver():
    socketio.run(app, host='0.0.0.0', port=5001)

# Create and run the visualizer
# if __name__ == "__main__":
#     visualizer = GyroVisualizer(space_limit=10)
#     visualizer.run()

# Create and run the visualizer
if __name__ == "__main__":
    try:
        # Start runserver generator
        data_thread = threading.Thread(target=runserver, daemon=True)
        data_thread.start()
        
        # Start visualization in the main thread
        visualizer = GyroVisualizer(space_limit=10)
        visualizer.run()
        
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close('all')