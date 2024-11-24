import serial
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow, Rectangle

class EnhancedGapDetector:
    def __init__(self, port='COM7', baud_rate=9600):
        self.serial_port = None
        self.port = port
        self.baud_rate = baud_rate
        self.data_points = []
        self.is_collecting = False
        
        # Scanning parameters
        self.center_angle = 90
        self.scan_range = 30
        self.min_angle = self.center_angle - self.scan_range//2
        self.max_angle = self.center_angle + self.scan_range//2
        
        # Gap detection parameters - updated for true gaps
        self.baseline_distance = None
        self.calibration_samples = []
        self.gap_threshold = 15  # Looking for gaps at least 15cm further than baseline
        self.min_gap_width = 3
        self.distance_buffer = []
        self.buffer_size = 3  # Reduced for faster response
        self.detected_gaps = []
        
        # Visualization setup
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.init_plots()
        
        # CSV logging
        self.csv_filename = f"gap_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()

    def init_plots(self):
        """Initialize plot settings"""
        for ax in [self.ax1, self.ax2]:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlim(-30, 30)
            ax.set_ylim(0, 100)  # Increased to show larger distances
            
        self.ax1.set_title('Real-time Scan Data', pad=10)
        self.ax2.set_title('Gap Analysis', pad=10)
        
        if self.baseline_distance:
            self.ax1.axhline(y=self.baseline_distance, color='yellow', 
                           linestyle='--', alpha=0.3)
            self.ax1.text(-28, self.baseline_distance, 
                         f'Baseline: {self.baseline_distance:.1f}cm', 
                         color='yellow', alpha=0.5)
            # Add expected gap threshold line
            gap_line = self.baseline_distance + self.gap_threshold
            self.ax1.axhline(y=gap_line, color='red', 
                           linestyle='--', alpha=0.3)
            self.ax1.text(-28, gap_line, 
                         f'Gap Threshold: {gap_line:.1f}cm', 
                         color='red', alpha=0.5)

    def init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'angle', 'distance', 'is_gap', 'baseline_distance'])
            print(f"Created CSV file: {self.csv_filename}")

    def detect_gap(self, distance, angle):
        """Enhanced gap detection looking for significantly larger distances"""
        if distance is None:
            return False
            
        # Smooth distance reading
        self.distance_buffer.append(distance)
        if len(self.distance_buffer) > self.buffer_size:
            self.distance_buffer.pop(0)
        
        smoothed_distance = np.mean(self.distance_buffer)
        
        # Check for significant distance increase
        if self.baseline_distance:
            distance_diff = smoothed_distance - self.baseline_distance
            is_gap = distance_diff > self.gap_threshold  # Must be significantly larger
            
            # Update gap tracking for visualization
            if is_gap:
                if not self.detected_gaps or abs(angle - self.detected_gaps[-1]["end_angle"]) > 2:
                    self.detected_gaps.append({
                        "start_angle": angle,
                        "end_angle": angle,
                        "distance": smoothed_distance,
                        "diff": distance_diff  # Store the difference for analysis
                    })
                else:
                    self.detected_gaps[-1]["end_angle"] = angle
                    self.detected_gaps[-1]["distance"] = smoothed_distance
                    self.detected_gaps[-1]["diff"] = distance_diff
            
            return is_gap
        return False

    def log_to_csv(self, angle, distance, is_gap):
        """Log data point to CSV file"""
        try:
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    f"{angle:.1f}",
                    f"{distance:.1f}",
                    is_gap,
                    f"{self.baseline_distance:.1f}" if self.baseline_distance else "N/A"
                ])
        except Exception as e:
            print(f"Error logging to CSV: {e}")

    def read_sensor_data(self):
        """Read and process sensor data with improved error handling"""
        if not self.serial_port:
            return None

        try:
            line = self.serial_port.readline().decode('utf-8').strip()
            if not line:  # Skip empty lines
                return None
                
            if "Angle" in line and "Distance" in line:
                parts = line.split(',')
                if len(parts) != 2:  # Check if we have both angle and distance
                    return None
                    
                try:
                    angle_part = parts[0].split(':')[1].strip()
                    distance_part = parts[1].split(':')[1].replace('cm', '').strip()
                    
                    # Only process if we have valid numbers
                    if angle_part and distance_part:
                        angle = float(angle_part)
                        distance = float(distance_part)
                        
                        if self.min_angle <= angle <= self.max_angle:
                            is_gap = self.detect_gap(distance, angle)
                            self.log_to_csv(angle, distance, is_gap)
                            return (angle, distance, is_gap)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing values: {e}")
                    return None
                    
        except Exception as e:
            print(f"Error reading sensor data: {e}")
        return None

    def update_plot(self, frame):
        """Update visualization with improved gap display"""
        data = self.read_sensor_data()
        if not data:
            return

        angle, distance, is_gap = data
        
        # Convert to plotting coordinates
        rad = np.radians(angle - self.center_angle)
        x = distance * np.sin(rad)
        y = distance * np.cos(rad)
        
        self.init_plots()  # Reset plots
        
        # Plot data point
        color = 'red' if is_gap else 'green'
        self.ax1.scatter(x, y, c=color, s=30, alpha=0.6)
        
        # Plot scan line
        scan_x = 100 * np.sin(rad)
        scan_y = 100 * np.cos(rad)
        self.ax1.plot([0, scan_x], [0, scan_y], 'r-', alpha=0.5)
        
        # Plot detected gaps
        for gap in self.detected_gaps:
            if abs(gap["end_angle"] - gap["start_angle"]) >= self.min_gap_width:
                start_rad = np.radians(gap["start_angle"] - self.center_angle)
                end_rad = np.radians(gap["end_angle"] - self.center_angle)
                
                angles = np.linspace(start_rad, end_rad, 20)
                xs = gap["distance"] * np.sin(angles)
                ys = gap["distance"] * np.cos(angles)
                self.ax1.fill_between(xs, ys, color='red', alpha=0.3)
        
        # Update status information
        status_text = f'Angle: {angle:.1f}°\n'
        status_text += f'Distance: {distance:.1f}cm\n'
        status_text += f'Gaps Found: {len(self.detected_gaps)}\n'
        if is_gap:
            status_text += f'GAP DETECTED (+{distance - self.baseline_distance:.1f}cm)'
        self.fig.suptitle(status_text, y=0.95)
        
        plt.tight_layout()

    def calibrate(self):
        """Calibrate baseline distance"""
        print("Starting calibration...")
        self.calibration_samples = []
        
        start_time = time.time()
        while len(self.calibration_samples) < 10 and (time.time() - start_time) < 10:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if "Angle" in line and "Distance" in line:
                    parts = line.split(',')
                    angle_part = parts[0].split(':')[1].strip()
                    distance_part = parts[1].split(':')[1].strip()
                    
                    angle = float(''.join(c for c in angle_part if c.isdigit() or c == '.'))
                    distance = float(''.join(c for c in distance_part if c.isdigit() or c == '.'))
                    
                    if abs(angle - self.center_angle) < 2:
                        self.calibration_samples.append(distance)
                        print(f"Calibration sample {len(self.calibration_samples)}: {distance:.1f}cm")
            except Exception as e:
                print(f"Calibration error: {e}")
                continue

        if self.calibration_samples:
            self.baseline_distance = np.median(self.calibration_samples)
            print(f"Calibration complete. Baseline distance: {self.baseline_distance:.1f}cm")
            print(f"Gap detection threshold: {self.baseline_distance + self.gap_threshold:.1f}cm")
        else:
            print("Calibration failed. Using default baseline of 30cm")
            self.baseline_distance = 30.0
        
        return self.baseline_distance

    def connect(self):
        """Establish connection with Arduino"""
        try:
            time.sleep(1)  # Small delay before connecting
            self.serial_port = serial.Serial(self.port, self.baud_rate)
            print(f"Connected to Arduino on {self.port}")
            time.sleep(2)
            return self.calibrate()
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False

    def collect_data(self, duration=60):
        """Collect and visualize data"""
        print("Starting gap detection scan...")
        self.is_collecting = True
        
        ani = FuncAnimation(self.fig, self.update_plot,
                          interval=50, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nScan interrupted by user")
        finally:
            self.is_collecting = False
            print(f"Data saved to {self.csv_filename}")

    def close(self):
        """Clean up resources"""
        if self.serial_port:
            self.serial_port.close()
        plt.close()

def main():
    detector = EnhancedGapDetector(port='COM7')  # Adjust port as needed
    
    if not detector.connect():
        return
        
    try:
        detector.collect_data(duration=60)
    except KeyboardInterrupt:
        print("\nScan interrupted by user")
    finally:
        detector.close()

if __name__ == "__main__":
    main()