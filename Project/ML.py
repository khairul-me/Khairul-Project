import serial
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow, Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from filterpy.kalman import KalmanFilter
from scipy import stats
import threading
import queue

class EnhancedGapDetector:
    def __init__(self, port='COM7', baud_rate=9600):
        self.serial_port = None
        self.port = port
        self.baud_rate = baud_rate
        self.data_points = []
        self.is_collecting = False
        
        # Enhanced scanning parameters
        self.center_angle = 90
        self.scan_range = 30
        self.min_angle = self.center_angle - self.scan_range//2
        self.max_angle = self.center_angle + self.scan_range//2
        
        # Advanced gap detection parameters
        self.baseline_distance = None
        self.calibration_samples = []
        self.gap_threshold = 8
        self.min_gap_width = 2
        self.distance_buffer = []
        self.buffer_size = 3  # Increased for better smoothing
        self.detected_gaps = []
        self.last_distances = [] 
        self.rate_change_threshold = 5
        # ML and statistical parameters
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.kalman_filters = {}
        self.confidence_scores = []
        self.measurement_errors = []
        self.data_queue = queue.Queue()
        
        # Dynamic thresholding
        self.threshold_window = 20
        self.threshold_history = []
        
        # Visualization setup
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(20, 7))
        self.init_plots()
        
        # CSV logging
        self.csv_filename = f"gap_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()
        
        # Initialize ML components
        self.init_ml_components()

    def init_ml_components(self):
        """Initialize machine learning components"""
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Initialize Kalman filter for each angle
        for angle in range(self.min_angle, self.max_angle + 1):
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([[0.], [0.]])  # state (distance, velocity)
            kf.F = np.array([[1., 1.], [0., 1.]])  # state transition matrix
            kf.H = np.array([[1., 0.]])  # measurement function
            kf.P *= 1000.  # covariance matrix
            kf.R = 5  # measurement noise
            kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])  # process noise
            self.kalman_filters[angle] = kf

    def update_dynamic_threshold(self, distance):
        """More sensitive dynamic threshold updating"""
        self.threshold_history.append(distance)
        if len(self.threshold_history) > self.threshold_window:
            self.threshold_history.pop(0)
        
        if len(self.threshold_history) >= 5:
            median = np.median(self.threshold_history)
            mad = np.median(np.abs(np.array(self.threshold_history) - median))
            std = 1.4826 * mad  # Robust standard deviation estimate
        
        # More sensitive threshold
            self.gap_threshold = max(8, median + 1.5 * std)

    def calculate_confidence_score(self, distance, angle):
        """Calculate confidence score for a measurement"""
        if not self.baseline_distance:
            return 0.5
            
        # Multiple factors for confidence
        factors = []
        
        # Distance variation factor
        if self.distance_buffer:
            std_dev = np.std(self.distance_buffer)
            variation_score = 1 / (1 + std_dev)
            factors.append(variation_score)
        
        # Kalman filter innovation factor
        kf = self.kalman_filters.get(int(angle))
        if kf:
            innovation = abs(distance - kf.x[0][0])
            innovation_score = 1 / (1 + innovation)
            factors.append(innovation_score)
        
        # Baseline deviation factor
        baseline_diff = abs(distance - self.baseline_distance)
        baseline_score = 1 / (1 + baseline_diff/100)
        factors.append(baseline_score)
        
        # Combine scores
        return np.mean(factors) if factors else 0.5

    def process_measurement(self, angle, distance):
        """Process and filter measurement data with enhanced smoothing"""
        # Apply Kalman filtering
        kf = self.kalman_filters.get(int(angle))
        if kf:
            kf.predict()
            kf.update(distance)
            filtered_distance = float(kf.x[0])
        else:
            filtered_distance = distance
        
        # Apply moving average smoothing
        self.distance_buffer.append(filtered_distance)
        if len(self.distance_buffer) > self.buffer_size:
            self.distance_buffer.pop(0)
        
        smoothed_distance = np.mean(self.distance_buffer) if self.distance_buffer else filtered_distance
        
        # Calculate rate of change
        rate_of_change = 0
        if len(self.last_distances) > 0:
            rate_of_change = abs(smoothed_distance - self.last_distances[-1])
        self.last_distances.append(smoothed_distance)
        if len(self.last_distances) > 3:
            self.last_distances.pop(0)
        
        # Update dynamic threshold
        self.update_dynamic_threshold(smoothed_distance)
        
        # Calculate confidence score
        confidence = self.calculate_confidence_score(smoothed_distance, angle)
        
        # Detect anomalies
        is_anomaly = False
        if len(self.data_points) > 50:
            X = np.array([[angle, smoothed_distance]])
            X_scaled = self.scaler.transform(X)
            is_anomaly = self.isolation_forest.predict(X_scaled)[0] == -1
        
        return smoothed_distance, confidence, is_anomaly, rate_of_change

    def detect_gap(self, distance, angle):
        """Enhanced gap detection with multiple criteria"""
        if distance is None:
            return False, 0
    
        # Process measurement with new rate of change
        filtered_distance, confidence, is_anomaly, rate_of_change = self.process_measurement(angle, distance)
    
    # Multi-criteria gap detection
        if self.baseline_distance:
            distance_diff = filtered_distance - self.baseline_distance
        
        # Combined criteria for gap detection
            is_gap = any([
            # Criterion 1: Basic threshold with relaxed confidence
                distance_diff > self.gap_threshold and confidence > 0.5,
            
            # Criterion 2: Sudden change detection
                rate_of_change > self.rate_change_threshold,
            
            # Criterion 3: Anomaly with significant difference
                is_anomaly and distance_diff > self.gap_threshold * 1.2
            ])
        
        # Update gap tracking with improved merging
            if is_gap:
                if not self.detected_gaps or abs(angle - self.detected_gaps[-1]["end_angle"]) > 2:
                # New gap
                    self.detected_gaps.append({
                    "start_angle": angle,
                    "end_angle": angle,
                    "distance": filtered_distance,
                    "confidence": confidence,
                    "diff": distance_diff,
                    "max_rate_change": rate_of_change
                })
            else:
                # Merge with existing gap
                last_gap = self.detected_gaps[-1]
                last_gap["end_angle"] = angle
                last_gap["distance"] = max(last_gap["distance"], filtered_distance)
                last_gap["confidence"] = max(last_gap["confidence"], confidence)
                last_gap["diff"] = max(last_gap["diff"], distance_diff)
                last_gap["max_rate_change"] = max(last_gap["max_rate_change"], rate_of_change)
        
            return is_gap, confidence
        return False, 0

    def init_plots(self):
        """Initialize enhanced plotting"""
        for ax in [self.ax1, self.ax2]:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlim(-30, 30)
            ax.set_ylim(0, 100)
        
        self.ax3.clear()
        self.ax3.grid(True, linestyle='--', alpha=0.3)
        self.ax3.set_xlim(self.min_angle, self.max_angle)
        self.ax3.set_ylim(0, 1)
        
        self.ax1.set_title('Real-time Scan Data', pad=10)
        self.ax2.set_title('Gap Analysis', pad=10)
        self.ax3.set_title('Confidence Metrics', pad=10)
        
        if self.baseline_distance:
            for ax in [self.ax1, self.ax2]:
                ax.axhline(y=self.baseline_distance, color='yellow', 
                          linestyle='--', alpha=0.3)
                ax.text(-28, self.baseline_distance, 
                       f'Baseline: {self.baseline_distance:.1f}cm', 
                       color='yellow', alpha=0.5)
                gap_line = self.baseline_distance + self.gap_threshold
                ax.axhline(y=gap_line, color='red', 
                          linestyle='--', alpha=0.3)
                ax.text(-28, gap_line, 
                       f'Gap Threshold: {gap_line:.1f}cm', 
                       color='red', alpha=0.5)

    def init_csv(self):
        """Initialize enhanced CSV logging"""
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'angle', 'raw_distance', 'filtered_distance',
                'is_gap', 'confidence', 'baseline_distance', 'threshold',
                'anomaly_score'
            ])

    def log_to_csv(self, angle, raw_distance, filtered_distance, is_gap, confidence):
        """Enhanced data logging"""
        try:
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    f"{angle:.1f}",
                    f"{raw_distance:.1f}",
                    f"{filtered_distance:.1f}",
                    is_gap,
                    f"{confidence:.3f}",
                    f"{self.baseline_distance:.1f}" if self.baseline_distance else "N/A",
                    f"{self.gap_threshold:.1f}",
                    f"{self.isolation_forest.score_samples([[angle, filtered_distance]])[0]:.3f}"
                    if self.isolation_forest else "N/A"
                ])
        except Exception as e:
            print(f"Error logging to CSV: {e}")

    def read_sensor_data(self):
  
        if not self.serial_port:
            return None

        try:
            # Read a line from the serial port
            line = self.serial_port.readline().decode('utf-8').strip()
            
            # Debugging: Print raw sensor data
            if not line:
                print("Empty line received from sensor")
                return None
            print(f"Raw line: {line}")

            # Ensure the line contains expected format
            if "Angle" in line and "Distance" in line:
                # Split the line into parts
                line_parts = line.split(',')
                
                # Validate that the line has two parts (Angle and Distance)
                if len(line_parts) != 2:
                    print(f"Unexpected format: {line}")
                    return None
                
                # Parse the angle
                try:
                    angle_str = line_parts[0].replace('Angle:', '').strip()
                    angle = float(angle_str)
                except ValueError:
                    print(f"Failed to parse angle from: {line_parts[0]}")
                    return None
                
                # Parse the distance
                try:
                    distance_str = line_parts[1].replace('Distance:', '').replace('cm', '').strip()
                    raw_distance = float(distance_str)
                except ValueError:
                    print(f"Failed to parse distance from: {line_parts[1]}")
                    return None
                
                # Ensure the angle is within the expected range
                if self.min_angle <= angle <= self.max_angle:
                    # Process the measurement
                    filtered_distance, confidence, is_anomaly, rate_of_change = self.process_measurement(
                        angle, raw_distance)
                    
                    # Detect gaps
                    is_gap, confidence = self.detect_gap(raw_distance, angle)
                    
                    # Store the processed data
                    self.data_points.append([angle, raw_distance])
                    
                    # Log data to CSV
                    self.log_to_csv(angle, raw_distance, filtered_distance, is_gap, confidence)
                    
                    # Return the processed data
                    return (angle, filtered_distance, is_gap, confidence)
                else:
                    print(f"Angle {angle} out of range ({self.min_angle} to {self.max_angle})")
                    return None
            else:
                print(f"Invalid data format: {line}")
                return None

        except Exception as e:
            print(f"Error reading sensor data: {e}")
            return None


    def update_plot(self, frame):
        """Enhanced visualization update"""
        data = self.read_sensor_data()
        if not data:
            return

        angle, distance, is_gap, confidence = data
        
        # Convert to plotting coordinates
        rad = np.radians(angle - self.center_angle)
        x = distance * np.sin(rad)
        y = distance * np.cos(rad)
        
        self.init_plots()
        
        # Plot data point
        color = 'red' if is_gap else 'green'
        self.ax1.scatter(x, y, c=color, s=30, alpha=max(0.3, confidence))
        
        # Plot scan line
        scan_x = 100 * np.sin(rad)
        scan_y = 100 * np.cos(rad)
        self.ax1.plot([0, scan_x], [0, scan_y], 'r-', alpha=0.3)
        
        # Plot detected gaps with confidence visualization
        for gap in self.detected_gaps:
            if abs(gap["end_angle"] - gap["start_angle"]) >= self.min_gap_width:
                start_rad = np.radians(gap["start_angle"] - self.center_angle)
                end_rad = np.radians(gap["end_angle"] - self.center_angle)
                
                angles = np.linspace(start_rad, end_rad, 20)
                xs = gap["distance"] * np.sin(angles)
                ys = gap["distance"] * np.cos(angles)
                
                # Plot in both ax1 and ax2 for different visualizations
                self.ax1.fill_between(xs, ys, color='red', alpha=max(0.2, gap.get("confidence", 0.5)))
                
                # Plot distance vs angle in ax2
                gap_angles = np.linspace(gap["start_angle"], gap["end_angle"], 20)
                gap_distances = np.full_like(gap_angles, gap["distance"])
                self.ax2.plot(gap_angles - self.center_angle, gap_distances, 'r-', alpha=0.5)
                
                # Add confidence bar in ax3
                self.ax3.bar(
                    gap["start_angle"],
                    gap.get("confidence", 0.5),
                    width=gap["end_angle"]-gap["start_angle"],
                    color='blue',
                    alpha=0.5
                )
        
        # Plot the current measurement in ax2
        self.ax2.scatter(angle - self.center_angle, distance, 
                        c=color, s=30, alpha=max(0.3, confidence))
        
        # Update status information
        status_text = (
            f'Angle: {angle:.1f}°\n'
            f'Distance: {distance:.1f}cm\n'
            f'Confidence: {confidence:.2f}\n'
            f'Gaps Found: {len(self.detected_gaps)}\n'
            f'Current Threshold: {self
.gap_threshold:.1f}cm'
        )
        if is_gap:
            status_text += f'\nGAP DETECTED (+{distance - self.baseline_distance:.1f}cm)'
        
        self.fig.suptitle(status_text, y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def calibrate(self):
        """Calibrate baseline distance with enhanced error checking"""
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
                        # Apply initial Kalman filtering during calibration
                        kf = self.kalman_filters.get(int(angle))
                        if kf:
                            kf.predict()
                            kf.update(distance)
                            filtered_distance = float(kf.x[0])
                            self.calibration_samples.append(filtered_distance)
                        else:
                            self.calibration_samples.append(distance)
                        print(f"Calibration sample {len(self.calibration_samples)}: {distance:.1f}cm")
            except Exception as e:
                print(f"Calibration error: {e}")
                continue

        if self.calibration_samples:
            # Use robust statistics for baseline calculation
            self.baseline_distance = np.median(self.calibration_samples)
            mad = np.median(np.abs(self.calibration_samples - self.baseline_distance))
            std_est = 1.4826 * mad  # Robust estimate of standard deviation
            
            # Initialize the isolation forest with calibration data
            X = np.array([[self.center_angle, d] for d in self.calibration_samples])
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.isolation_forest.fit(X_scaled)
            
            print(f"Calibration complete. Baseline distance: {self.baseline_distance:.1f}cm")
            print(f"Estimated noise level: {std_est:.2f}cm")
            print(f"Gap detection threshold: {self.baseline_distance + self.gap_threshold:.1f}cm")
        else:
            print("Calibration failed. Using default baseline of 30cm")
            self.baseline_distance = 30.0
        
        return self.baseline_distance

    def connect(self):
        """Establish connection with Arduino with enhanced error handling"""
        try:
            print(f"Attempting to connect to Arduino on {self.port}...")
            time.sleep(1)  # Small delay before connecting
            
            # Try to connect with different baud rates if necessary
            baud_rates = [9600, 115200]  # Common baud rates
            for baud_rate in baud_rates:
                try:
                    self.serial_port = serial.Serial(self.port, baud_rate, timeout=1)
                    print(f"Connected to Arduino on {self.port} at {baud_rate} baud")
                    time.sleep(2)  # Allow connection to stabilize
                    
                    # Test communication
                    if self.test_connection():
                        return self.calibrate()
                    else:
                        self.serial_port.close()
                except:
                    continue
            
            raise Exception("Failed to establish stable connection")
            
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False

    def test_connection(self):
        """Test if the connection is working properly"""
        try:
            # Try to read a few lines to ensure we're getting valid data
            for _ in range(5):
                line = self.serial_port.readline().decode('utf-8').strip()
                if "Angle" in line and "Distance" in line:
                    return True
            return False
        except:
            return False

    def collect_data(self, duration=60):
        """Collect and visualize data with enhanced error handling"""
        print("Starting gap detection scan...")
        self.is_collecting = True
        
        try:
            # Start the animation
            ani = FuncAnimation(self.fig, self.update_plot,
                              interval=50, cache_frame_data=False)
            
            # Set up plot closing handler
            def on_close(event):
                self.is_collecting = False
            self.fig.canvas.mpl_connect('close_event', on_close)
            
            plt.show()
            
        except KeyboardInterrupt:
            print("\nScan interrupted by user")
        except Exception as e:
            print(f"Error during data collection: {e}")
        finally:
            self.is_collecting = False
            print(f"Data saved to {self.csv_filename}")

    def close(self):
        """Clean up resources"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        plt.close('all')

def main():
    port = 'COM7'  # Adjust port as needed
    detector = EnhancedGapDetector(port=port)
    
    try:
        if not detector.connect():
            print("Failed to initialize the gap detector. Please check the connection and try again.")
            return
            
        detector.collect_data(duration=60)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        detector.close()

if __name__ == "__main__":
    main()