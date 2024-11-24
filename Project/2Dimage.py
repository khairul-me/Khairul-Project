import serial
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc, Arrow

class FocusedScanner:
    def __init__(self, port='COM7', baud_rate=9600):
        self.serial_port = None
        self.port = port
        self.baud_rate = baud_rate
        self.data_points = []
        self.is_collecting = False
        
        # Scanning parameters
        self.center_angle = 90
        self.scan_range = 5
        self.min_distance = 20
        self.max_distance = 700
        
        # State tracking
        self.last_angle = None
        self.movement_direction = "UNKNOWN"
        self.scan_cycle = 0
        self.start_time = None
        self.current_state = "UNKNOWN"
        
        # Visualization setup
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Data buffers
        self.buffer_size = 100
        self.angle_buffer = []
        self.distance_buffer = []
        
        # Initial plot setup
        self.setup_plots()

    def determine_state(self, current_angle):
        """Determine the current state and movement direction"""
        if self.last_angle is None:
            self.last_angle = current_angle
            return "INITIALIZING", "UNKNOWN"
            
        # Calculate movement direction
        if abs(current_angle - self.last_angle) > 0.1:  # Threshold for movement detection
            if current_angle < self.last_angle:
                direction = "LEFT"
            else:
                direction = "RIGHT"
        else:
            direction = "STATIONARY"
            
        # Determine state based on angle and movement
        if abs(current_angle - self.center_angle) < 2:  # Near center
            state = "AT_CENTER"
        elif current_angle < self.center_angle:
            state = "SCANNING_LEFT"
        else:
            state = "SCANNING_RIGHT"
            
        # Update scan cycle if we've completed a full movement pattern
        if self.last_angle > self.center_angle and current_angle <= self.center_angle:
            self.scan_cycle += 1
            
        self.last_angle = current_angle
        return state, direction

    def setup_plots(self):
        """Configure both visualization plots"""
        for ax in [self.ax1, self.ax2]:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlim(-15, 15)
            ax.set_ylim(15, 75)
            
            # Add limit lines and labels
            ax.axhline(y=self.min_distance, color='red', linestyle='--', alpha=0.3)
            ax.axhline(y=self.max_distance, color='red', linestyle='--', alpha=0.3)
            
            ax.text(-14, self.min_distance, f'Min: {self.min_distance}cm', 
                   color='red', alpha=0.5, verticalalignment='bottom')
            ax.text(-14, self.max_distance, f'Max: {self.max_distance}cm', 
                   color='red', alpha=0.5, verticalalignment='bottom')
            
            ax.fill_between([-15, 15], 15, self.min_distance, 
                          color='red', alpha=0.1)
            ax.fill_between([-15, 15], self.max_distance, 75, 
                          color='red', alpha=0.1)
        
        self.ax1.set_title('Real-time Scan Data', pad=10)
        self.ax2.set_title('Density Analysis', pad=10)
        
        self.add_reference_lines(self.ax1)
        self.add_reference_lines(self.ax2)

    def add_reference_lines(self, ax):
        """Add reference lines and scan range indicators"""
        # Center line
        ax.plot([0, 0], [0, self.max_distance], '--', color='gray', alpha=0.3)
        
        # Scan range indicators
        for angle in [85, 90, 95]:  # Fixed angle labels
            rad = np.radians(angle - self.center_angle)
            x = self.max_distance * np.sin(rad)
            y = self.max_distance * np.cos(rad)
            ax.plot([0, x], [0, y], '--', color='gray', alpha=0.3)
            ax.text(x, y, f'{angle}°', color='white', alpha=0.5)

    def connect(self):
        """Establish connection with Arduino"""
        try:
            self.serial_port = serial.Serial(self.port, self.baud_rate)
            print(f"Connected to Arduino on {self.port}")
            self.start_time = datetime.now()
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False

    def read_sensor_data(self):
        """Read and parse sensor data"""
        if not self.serial_port:
            return None
            
        try:
            line = self.serial_port.readline().decode('utf-8').strip()
            print(f"Raw data: {line}")  # Debug output
            
            if "Angle" in line and "Distance" in line:
                parts = line.split(',')
                angle = float(parts[0].split(':')[1].strip())
                distance = float(parts[1].split(':')[1].replace('cm', '').strip())
                
                # Determine state and direction
                state, direction = self.determine_state(angle)
                
                # Calculate time elapsed
                time_elapsed = (datetime.now() - self.start_time).total_seconds()
                
                # Only process data within valid range
                if self.min_distance <= distance <= self.max_distance:
                    x, y = self.process_point(angle, distance)
                    data_point = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'angle': angle,
                        'distance': distance,
                        'x': x,
                        'y': y,
                        'state': state,
                        'direction': direction,
                        'time_elapsed': time_elapsed,
                        'scan_cycle': self.scan_cycle
                    }
                    self.data_points.append(data_point)
                    self.current_state = state
                    self.movement_direction = direction
                    return (angle, distance, state)
                    
        except Exception as e:
            print(f"Error reading sensor data: {e}")
        return None

    def process_point(self, angle, distance):
        """Convert polar to Cartesian coordinates"""
        rad = np.radians(angle - self.center_angle)
        x = distance * np.sin(rad)
        y = distance * np.cos(rad)
        return x, y

    def update_plot(self, frame):
        """Update both plots with new data"""
        data = self.read_sensor_data()
        if not data:
            return
            
        angle, distance, state = data
        
        # Update buffers
        self.angle_buffer.append(angle)
        self.distance_buffer.append(distance)
        if len(self.angle_buffer) > self.buffer_size:
            self.angle_buffer.pop(0)
            self.distance_buffer.pop(0)
        
        # Clear and setup plots
        self.setup_plots()
        
        # Convert points to Cartesian
        points = [self.process_point(a, d) 
                 for a, d in zip(self.angle_buffer, self.distance_buffer)]
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        
        # Plot 1: Real-time scan data
        scatter = self.ax1.scatter(x_points, y_points, 
                                 c=self.distance_buffer,
                                 cmap='viridis', 
                                 s=30, 
                                 alpha=0.6)
        
        # Add current scan line
        current_x, current_y = self.process_point(angle, self.max_distance)
        self.ax1.plot([0, current_x], [0, current_y], 'r-', alpha=0.5)
        
        # Plot 2: Density analysis
        if len(x_points) > 10:
            hist, xedges, yedges = np.histogram2d(
                x_points, y_points,
                bins=[20, 20],
                range=[[-15, 15], [15, 75]]
            )
            
            self.ax2.pcolormesh(xedges, yedges, hist.T,
                              cmap='YlOrRd', alpha=0.7)
        
        # Add state and cycle information
        status_text = f'State: {self.current_state}\n'
        status_text += f'Direction: {self.movement_direction}\n'
        status_text += f'Scan Cycle: {self.scan_cycle}'
        self.fig.suptitle(status_text, y=0.95)
        
        # Add servo position indicator
        angle_rad = np.radians(angle - self.center_angle)
        indicator = Arrow(0, 0,
                        5*np.sin(angle_rad),
                        5*np.cos(angle_rad),
                        color='red', width=2)
        self.ax1.add_patch(indicator)
        
        plt.tight_layout()

    def save_data(self, filename=None):
        """Save collected data to CSV file"""
        if not self.data_points:
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"radar_scan_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as file:
                fieldnames = ['timestamp', 'angle', 'distance', 'x', 'y', 
                            'state', 'direction', 'time_elapsed', 'scan_cycle']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data_points)
                    
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def collect_data(self, duration=60, save_interval=1000):
        """Collect and visualize data"""
        print("Starting focused scan...")
        self.is_collecting = True
        self.start_time = datetime.now()
        points_since_save = 0
        
        ani = FuncAnimation(self.fig, self.update_plot,
                          interval=50, cache_frame_data=False)
        
        try:
            while (datetime.now() - self.start_time).total_seconds() < duration and self.is_collecting:
                if len(self.data_points) >= save_interval:
                    self.save_data()
                    self.data_points = []
                    points_since_save = 0
                
                plt.pause(0.01)
                
        except KeyboardInterrupt:
            print("\nScan interrupted by user")
        finally:
            if self.data_points:
                self.save_data()
            self.is_collecting = False
            plt.show()

    def close(self):
        """Clean up resources"""
        if self.data_points:
            self.save_data()
        if self.serial_port:
            self.serial_port.close()
        plt.close()

def main():
    scanner = FocusedScanner(port='COM7')  # Adjust port as needed
    
    if not scanner.connect():
        return
        
    try:
        scanner.collect_data(duration=60, save_interval=1000)
    except KeyboardInterrupt:
        print("\nScan interrupted by user")
    finally:
        scanner.close()

if __name__ == "__main__":
    main()