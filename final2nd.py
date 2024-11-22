import serial
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from filterpy.kalman import KalmanFilter
import pandas as pd
import queue

class EnhancedGapDetector:
    def __init__(self, port='COM7', baud_rate=9600):
        # Serial and basic parameters
        self.serial_port = None
        self.port = port
        self.baud_rate = baud_rate
        self.data_points = []
        self.is_collecting = False

        # ML components initialization
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.kalman_filters = {}
        self.confidence_scores = []
        self.measurement_errors = []
        self.data_queue = queue.Queue()
        self.regression_model = None
        self.mlp_model = None

        # Enhanced scanning parameters
        self.center_angle = 90
        self.scan_range = 30
        self.min_angle = self.center_angle - self.scan_range // 2
        self.max_angle = self.center_angle + self.scan_range // 2

        # Gap detection parameters
        self.baseline_distance = None
        self.calibration_samples = []
        self.gap_threshold = 8
        self.min_gap_width = 2
        self.distance_buffer = []
        self.buffer_size = 3
        self.detected_gaps = []
        self.last_distances = []
        self.rate_change_threshold = 5

        # Dynamic thresholding
        self.threshold_window = 20
        self.threshold_history = []

        # Initialize ML components
        self.init_ml_components()

        # Visualization setup
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 7))
        gs = self.fig.add_gridspec(1, 4)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.polar_ax = self.fig.add_subplot(gs[0, 3], polar=True)

        # CSV logging
        self.csv_filename = f"gap_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()
    
    
    #2nd Chunk
    
    def init_ml_components(self):
        """Initialize machine learning components."""
        # Initialize IsolationForest
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )

        # Initialize Kalman filters
        for angle in range(self.min_angle, self.max_angle + 1):
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([[0.], [0.]])
            kf.F = np.array([[1., 1.], [0., 1.]])
            kf.H = np.array([[1., 0.]])
            kf.P *= 1000
            kf.R = 5
            kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])
            self.kalman_filters[angle] = kf

        # Train models if data available
        try:
            data = pd.read_csv("gap_training_data.csv")
            X = data[['angle_start', 'angle_end', 'distance']].values
            y = data['gap_width'].values

            self.regression_model = LinearRegression()
            self.regression_model.fit(X, y)

            self.mlp_model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42
            )
            self.mlp_model.fit(X, y)
        except FileNotFoundError:
            print("Warning: Training data file not found. Using basic gap detection.")
        except Exception as e:
            print(f"Warning: Could not initialize ML models: {e}")

    def apply_clustering(self):
        """Apply DBSCAN clustering to detect gaps."""
        if len(self.data_points) < 10:
            return

        X = np.array([[point[0], point[1]] for point in self.data_points])
        dbscan = DBSCAN(eps=5, min_samples=3)
        labels = dbscan.fit_predict(X)

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_points = X[labels == cluster_id]
            avg_distance = np.mean(cluster_points[:, 1])

            if avg_distance < self.baseline_distance - self.gap_threshold:
                start_angle = np.min(cluster_points[:, 0])
                end_angle = np.max(cluster_points[:, 0])
                gap_width = 2 * avg_distance * np.tan(np.radians((end_angle - start_angle) / 2))

                if self.mlp_model:
                    predicted_gap = self.predict_gap_size_mlp(start_angle, end_angle, avg_distance)
                elif self.regression_model:
                    predicted_gap = self.predict_gap_size(start_angle, end_angle, avg_distance)
                else:
                    predicted_gap = gap_width

                print(f"Gap detected: Start angle: {start_angle:.1f}°, End angle: {end_angle:.1f}°, "
                      f"Width: {predicted_gap:.2f} cm")
                
    #3rd Chunk
    def process_measurement(self, angle, distance):
        """Process and filter measurement data."""
        kf = self.kalman_filters.get(int(angle))
        if kf:
            kf.predict()
            kf.update(distance)
            filtered_distance = float(kf.x[0])
        else:
            filtered_distance = distance
        
        self.distance_buffer.append(filtered_distance)
        if len(self.distance_buffer) > self.buffer_size:
            self.distance_buffer.pop(0)
        
        smoothed_distance = np.mean(self.distance_buffer) if self.distance_buffer else filtered_distance
        
        rate_of_change = 0
        if self.last_distances:
            rate_of_change = abs(smoothed_distance - self.last_distances[-1])
        self.last_distances.append(smoothed_distance)
        if len(self.last_distances) > 3:
            self.last_distances.pop(0)
        
        self.update_dynamic_threshold(smoothed_distance)
        confidence = self.calculate_confidence_score(smoothed_distance, angle)
        
        is_anomaly = False
        if len(self.data_points) > 50:
            X = np.array([[angle, smoothed_distance]])
            X_scaled = self.scaler.transform(X)
            is_anomaly = self.isolation_forest.predict(X_scaled)[0] == -1
        
        return smoothed_distance, confidence, is_anomaly, rate_of_change

    def detect_gap(self, distance, angle):
        """Enhanced gap detection."""
        if distance is None:
            return False, 0
        
        filtered_distance, confidence, is_anomaly, rate_of_change = self.process_measurement(angle, distance)
        
        if self.baseline_distance:
            distance_diff = filtered_distance - self.baseline_distance
            is_gap = any([
                distance_diff > self.gap_threshold and confidence > 0.5,
                rate_of_change > self.rate_change_threshold,
                is_anomaly and distance_diff > self.gap_threshold * 1.2
            ])
            
            if is_gap:
                if not self.detected_gaps or abs(angle - self.detected_gaps[-1]["end_angle"]) > 2:
                    self.detected_gaps.append({
                        "start_angle": angle,
                        "end_angle": angle,
                        "distance": filtered_distance,
                        "confidence": confidence,
                        "diff": distance_diff,
                        "max_rate_change": rate_of_change
                    })
                else:
                    last_gap = self.detected_gaps[-1]
                    last_gap["end_angle"] = angle
                    last_gap["distance"] = max(last_gap["distance"], filtered_distance)
                    last_gap["confidence"] = max(last_gap["confidence"], confidence)
                    last_gap["diff"] = max(last_gap["diff"], distance_diff)
                    last_gap["max_rate_change"] = max(last_gap["max_rate_change"], rate_of_change)
            
            return is_gap, confidence
        return False, 0

    def update_dynamic_threshold(self, distance):
        """Update dynamic threshold."""
        self.threshold_history.append(distance)
        if len(self.threshold_history) > self.threshold_window:
            self.threshold_history.pop(0)
        
        if len(self.threshold_history) >= 5:
            median = np.median(self.threshold_history)
            mad = np.median(np.abs(np.array(self.threshold_history) - median))
            std = 1.4826 * mad
            self.gap_threshold = max(8, median + 1.5 * std)

    def calculate_confidence_score(self, distance, angle):
        """Calculate measurement confidence."""
        if not self.baseline_distance:
            return 0.5
        
        factors = []
        
        if self.distance_buffer:
            std_dev = np.std(self.distance_buffer)
            factors.append(1 / (1 + std_dev))
        
        kf = self.kalman_filters.get(int(angle))
        if kf:
            innovation = abs(distance - kf.x[0][0])
            factors.append(1 / (1 + innovation))
        
        baseline_diff = abs(distance - self.baseline_distance)
        factors.append(1 / (1 + baseline_diff/100))
        
        return np.mean(factors) if factors else 0.5
    
    #4rth Chunk
    def init_plots(self):
        """Initialize plots."""
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
                ax.axhline(y=self.baseline_distance, color='yellow', linestyle='--', alpha=0.3)
                ax.text(-28, self.baseline_distance, f'Baseline: {self.baseline_distance:.1f}cm',
                        color='yellow', alpha=0.5)
                gap_line = self.baseline_distance + self.gap_threshold
                ax.axhline(y=gap_line, color='red', linestyle='--', alpha=0.3)
                ax.text(-28, gap_line, f'Gap Threshold: {gap_line:.1f}cm', color='red', alpha=0.5)

        if self.polar_ax:
            self.polar_ax.clear()
            self.polar_ax.set_title("Polar Plot of Detected Points")

    def plot_polar(self):
        """Plot polar data."""
        if not self.polar_ax:
            return

        angles = np.radians([point[0] for point in self.data_points])
        distances = [point[1] for point in self.data_points]

        self.polar_ax.clear()
        self.polar_ax.scatter(angles, distances, c='blue', s=10, alpha=0.6)
        self.polar_ax.set_title("Polar Plot of Detected Points")

    def update_plot(self, frame):
        """Update plot animation."""
        data = self.read_sensor_data()
        if not data:
            return

        angle, distance, is_gap, confidence = data
        
        rad = np.radians(angle - self.center_angle)
        x = distance * np.sin(rad)
        y = distance * np.cos(rad)
        
        self.init_plots()
        
        color = 'red' if is_gap else 'green'
        self.ax1.scatter(x, y, c=color, s=30, alpha=max(0.3, confidence))
        
        scan_x = 100 * np.sin(rad)
        scan_y = 100 * np.cos(rad)
        self.ax1.plot([0, scan_x], [0, scan_y], 'r-', alpha=0.3)
        
        for gap in self.detected_gaps:
            if abs(gap["end_angle"] - gap["start_angle"]) >= self.min_gap_width:
                start_rad = np.radians(gap["start_angle"] - self.center_angle)
                end_rad = np.radians(gap["end_angle"] - self.center_angle)
                
                angles = np.linspace(start_rad, end_rad, 20)
                xs = gap["distance"] * np.sin(angles)
                ys = gap["distance"] * np.cos(angles)
                
                self.ax1.fill_between(xs, ys, color='red', alpha=max(0.2, gap.get("confidence", 0.5)))
                
                gap_angles = np.linspace(gap["start_angle"], gap["end_angle"], 20)
                gap_distances = np.full_like(gap_angles, gap["distance"])
                self.ax2.plot(gap_angles - self.center_angle, gap_distances, 'r-', alpha=0.5)
                
                self.ax3.bar(
                    gap["start_angle"],
                    gap.get("confidence", 0.5),
                    width=gap["end_angle"]-gap["start_angle"],
                    color='blue',
                    alpha=0.5
                )
        
        self.ax2.scatter(angle - self.center_angle, distance, c=color, s=30, alpha=max(0.3, confidence))
        
        status_text = (
            f'Angle: {angle:.1f}°\n'
            f'Distance: {distance:.1f}cm\n'
            f'Confidence: {confidence:.2f}\n'
            f'Gaps Found: {len(self.detected_gaps)}\n'
            f'Current Threshold: {self.gap_threshold:.1f}cm'
        )
        if is_gap:
            status_text += f'\nGAP DETECTED (+{distance - self.baseline_distance:.1f}cm)'
        
        self.fig.suptitle(status_text, y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.enhanced_visualization()             
    
    def enhanced_visualization(self):
        """Enhanced visualization for gaps and clusters."""
        # Call clustering if enough data points exist
        if len(self.data_points) >= 10:
            self.apply_clustering()

        # Update polar plot
        if self.polar_ax:
            self.plot_polar()

        # Add gap size annotations
        for gap in self.detected_gaps:
            if abs(gap["end_angle"] - gap["start_angle"]) >= self.min_gap_width:
                self.ax1.text(gap["start_angle"], gap["distance"],
                            f"Gap: {gap['diff']:.2f} cm", color='white')
    
    
    #5th Chunk##############
    
    def init_csv(self):
        """Initialize CSV logging."""
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'angle', 'raw_distance', 'filtered_distance',
                'is_gap', 'confidence', 'baseline_distance', 'threshold',
                'anomaly_score'
            ])

    def log_to_csv(self, angle, raw_distance, filtered_distance, is_gap, confidence):
        """Log data to CSV."""
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
        """Read and process sensor data."""
        if not self.serial_port:
            return None

        try:
            line = self.serial_port.readline().decode('utf-8').strip()
            
            if not line:
                print("Empty line received from sensor")
                return None
                
            print(f"Raw line: {line}")

            if "Angle" in line and "Distance" in line:
                line_parts = line.split(',')
                
                if len(line_parts) != 2:
                    print(f"Unexpected format: {line}")
                    return None
                
                try:
                    angle_str = line_parts[0].replace('Angle:', '').strip()
                    angle = float(angle_str)
                except ValueError:
                    print(f"Failed to parse angle from: {line_parts[0]}")
                    return None
                
                try:
                    distance_str = line_parts[1].replace('Distance:', '').replace('cm', '').strip()
                    raw_distance = float(distance_str)
                except ValueError:
                    print(f"Failed to parse distance from: {line_parts[1]}")
                    return None
                
                if self.min_angle <= angle <= self.max_angle:
                    filtered_distance, confidence, is_anomaly, rate_of_change = self.process_measurement(
                        angle, raw_distance)
                    
                    is_gap, confidence = self.detect_gap(raw_distance, angle)
                    self.data_points.append([angle, raw_distance])
                    self.log_to_csv(angle, raw_distance, filtered_distance, is_gap, confidence)
                    
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
        
    #6th Chunk #######################################################################################################################
    
    def calibrate(self):
        """Calibrate baseline distance."""
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
            self.baseline_distance = np.median(self.calibration_samples)
            mad = np.median(np.abs(self.calibration_samples - self.baseline_distance))
            std_est = 1.4826 * mad
            
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
        """Establish connection with Arduino."""
        try:
            print(f"Attempting to connect to Arduino on {self.port}...")
            time.sleep(1)
            
            baud_rates = [9600, 115200]
            for baud_rate in baud_rates:
                try:
                    self.serial_port = serial.Serial(self.port, baud_rate, timeout=1)
                    print(f"Connected to Arduino on {self.port} at {baud_rate} baud")
                    time.sleep(2)
                    
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
        """Test connection stability."""
        try:
            for _ in range(5):
                line = self.serial_port.readline().decode('utf-8').strip()
                if "Angle" in line and "Distance" in line:
                    return True
            return False
        except:
            return False

    def collect_data(self, duration=60):
        """Collect and visualize data."""
        print("Starting gap detection scan...")
        self.is_collecting = True
        
        try:
            ani = FuncAnimation(self.fig, self.update_plot,
                              interval=50, cache_frame_data=False)
            
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
        """Clean up resources."""
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
    
    
    