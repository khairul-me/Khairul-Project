import serial
import time
import csv
import numpy as np
from datetime import datetime

class RadarDataCollector:
    def __init__(self, port='COM3', baud_rate=9600):
        """
        Initialize the radar data collector
        Args:
            port (str): Serial port for Arduino connection
            baud_rate (int): Baud rate for serial communication
        """
        self.serial_port = None
        self.port = port
        self.baud_rate = baud_rate
        self.data_points = []
        self.is_collecting = False

    def connect(self):
        """Establish connection with Arduino"""
        try:
            self.serial_port = serial.Serial(self.port, self.baud_rate)
            print(f"Connected to Arduino on {self.port}")
            time.sleep(2)  # Allow Arduino to reset
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False

    def read_sensor_data(self):
        """
        Read a single data point from the sensor.
        Returns:
            tuple: (angle, distance) if valid data, None otherwise
        """
        if not self.serial_port:
            return None

        try:
            # Read a line from the Arduino
            line = self.serial_port.readline().decode('utf-8').strip()
            print(f"Raw data: {line}")  # Debugging output
            
            # Expecting format: "Angle: X, Distance: Y cm"
            if "Angle" in line and "Distance" in line:
                angle_str = line.split(",")[0].split(":")[1].strip()
                distance_str = line.split(",")[1].split(":")[1].strip().replace("cm", "")
                
                # Convert to float
                angle = float(angle_str)
                distance = float(distance_str)

                # Validate the data
                if 0 <= angle <= 180 and 0 <= distance <= 400:  # Assuming 400 cm max range
                    return (angle, distance)
            return None
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            return None

    def process_data_point(self, angle, distance):
        """
        Convert polar coordinates (angle, distance) to Cartesian (x, y)
        Args:
            angle (float): Angle in degrees
            distance (float): Distance in cm
        Returns:
            tuple: (x, y) coordinates
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate x and y coordinates
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        
        return (x, y)

    def collect_data(self, duration=60, save_interval=1000):
        """
        Collect data for the specified duration
        Args:
            duration (int): Collection duration in seconds
            save_interval (int): Number of points before auto-saving
        """
        print(f"Starting data collection for {duration} seconds...")
        start_time = time.time()
        self.is_collecting = True
        points_since_save = 0

        try:
            while time.time() - start_time < duration and self.is_collecting:
                # Read sensor data
                sensor_data = self.read_sensor_data()
                if sensor_data:
                    angle, distance = sensor_data

                    # Process data point
                    x, y = self.process_data_point(angle, distance)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # Store data point
                    data_point = {
                        'timestamp': timestamp,
                        'angle': angle,
                        'distance': distance,
                        'x': x,
                        'y': y
                    }
                    self.data_points.append(data_point)
                    points_since_save += 1

                    # Auto-save at intervals
                    if points_since_save >= save_interval:
                        self.save_data()
                        points_since_save = 0

                time.sleep(0.01)  # Small delay to prevent overwhelming the serial port

        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        finally:
            self.is_collecting = False
            self.save_data()

    def save_data(self, filename=None):
        """
        Save collected data to a CSV file
        Args:
            filename (str): Optional custom filename
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"radar_data_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['timestamp', 'angle', 'distance', 'x', 'y'])
                writer.writeheader()
                writer.writerows(self.data_points)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def close(self):
        """Close the serial connection"""
        if self.serial_port:
            self.serial_port.close()
            print("Serial connection closed")

def main():
    # Initialize collector
    collector = RadarDataCollector(port='COM7')  # Adjust port as needed

    # Connect to Arduino
    if not collector.connect():
        return

    try:
        # Collect data for 1 minute
        collector.collect_data(duration=60)
    finally:
        collector.close()

if __name__ == "__main__":
    main()
