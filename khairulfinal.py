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
from sklearn.metrics import precision_score, recall_score, f1_score
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

        # Validation metrics
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.ground_truth_gaps = []
        self.validation_history = []  # Added to store historical validation metrics
        
        # Enhanced confidence metrics
        self.angle_distance_history = {}
        self.anomaly_history = []
        self.min_confidence_threshold = 0.3
        self.temporal_consistency_window = 5
        self.spatial_consistency_window = 3
        self.confidence_history = []
        self.historical_confidence_scores = {}  # Added for tracking confidence over time

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
            return self.baseline_distance
        else:
            print("Calibration failed. Using default baseline of 30cm")
            self.baseline_distance = 30.0
            return self.baseline_distance        
    
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
    
    
    def update_dynamic_threshold(self, distance):
        """Update dynamic threshold based on recent measurements."""
        self.threshold_history.append(distance)
        if len(self.threshold_history) > self.threshold_window:
            self.threshold_history.pop(0)
        
        if len(self.threshold_history) >= 5:
            median = np.median(self.threshold_history)
            mad = np.median(np.abs(np.array(self.threshold_history) - median))
            std = 1.4826 * mad
            self.gap_threshold = max(8, median + 1.5 * std)
    
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

    
    def record_gap(self, min_angle, max_angle, avg_distance, cluster_confidence):
        """Record a detected gap with validation."""
        new_gap = {
            "start_angle": min_angle,
            "end_angle": max_angle,
            "distance": avg_distance,
            "confidence": cluster_confidence,
            "last_updated": time.time(),
            "persistence": 0
        }
        
        # Validate gap if ground truth available
        if self.ground_truth_gaps:
            validation_score = self.validate_detection(new_gap)
            new_gap["validation_score"] = validation_score
            
            if validation_score > 0.5:
                self.true_positives += 1
            else:
                self.false_positives += 1
        
        # Check for overlap with existing gaps
        overlap = False
        for existing_gap in self.detected_gaps:
            if (min_angle <= existing_gap["end_angle"] and 
                max_angle >= existing_gap["start_angle"]):
                overlap = True
                # Update existing gap if new one has higher confidence
                if cluster_confidence > existing_gap["confidence"]:
                    existing_gap.update(new_gap)
                break
        
        if not overlap:
            self.detected_gaps.append(new_gap)
            self.update_validation_metrics(True, bool(self.ground_truth_gaps))
        
    ######## PART 2 #########################
    
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

    def calculate_confidence_score(self, distance, angle):
        """Enhanced confidence calculation with multiple features."""
        base_factors = []
        
        # Distance buffer stability
        if self.distance_buffer:
            std_dev = np.std(self.distance_buffer)
            base_factors.append(1 / (1 + std_dev))
        
        # Kalman filter innovation
        kf = self.kalman_filters.get(int(angle))
        if kf:
            innovation = abs(distance - kf.x[0][0])
            base_factors.append(1 / (1 + innovation))
        
        # Baseline difference
        if self.baseline_distance:
            baseline_diff = abs(distance - self.baseline_distance)
            base_factors.append(1 / (1 + baseline_diff/100))
        
        # Angle-based distance variability
        if angle in self.angle_distance_history:
            history = self.angle_distance_history[angle]
            if len(history) >= self.spatial_consistency_window:
                angle_std = np.std(history[-self.spatial_consistency_window:])
                base_factors.append(1 / (1 + angle_std/5))
        
        # Historical anomaly trends
        if len(self.anomaly_history) >= self.temporal_consistency_window:
            recent_anomalies = sum(self.anomaly_history[-self.temporal_consistency_window:])
            anomaly_factor = 1 - (recent_anomalies / self.temporal_consistency_window)
            base_factors.append(anomaly_factor)
        
        # Temporal consistency with previous gaps
        if self.detected_gaps:
            last_gap = self.detected_gaps[-1]
            time_since_last = abs(angle - last_gap["end_angle"])
            temporal_factor = 1 / (1 + time_since_last/10)
            base_factors.append(temporal_factor)
        
        # Rate of change stability
        if len(self.last_distances) >= 2:
            rate_change = abs(self.last_distances[-1] - self.last_distances[-2])
            rate_factor = 1 / (1 + rate_change/self.rate_change_threshold)
            base_factors.append(rate_factor)
        
        # Historical confidence trend
        if angle in self.historical_confidence_scores:
            hist_scores = self.historical_confidence_scores[angle]
            if len(hist_scores) > 0:
                trend_factor = np.mean(hist_scores)
                base_factors.append(trend_factor)
        
        # Combine all factors
        confidence = np.mean(base_factors) if base_factors else 0.5
        confidence = max(self.min_confidence_threshold, confidence)
        
        # Update confidence history
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.temporal_consistency_window:
            self.confidence_history.pop(0)
            
        # Update historical confidence scores
        if angle not in self.historical_confidence_scores:
            self.historical_confidence_scores[angle] = []
        self.historical_confidence_scores[angle].append(confidence)
        if len(self.historical_confidence_scores[angle]) > self.temporal_consistency_window:
            self.historical_confidence_scores[angle].pop(0)
        
        return confidence
    


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
    
    
    ####### Part 3 #################
    
    def update_validation_metrics(self, predicted_gap, actual_gap):
        """Update validation metrics based on predictions vs ground truth."""
        if predicted_gap and actual_gap:
            self.true_positives += 1
        elif predicted_gap and not actual_gap:
            self.false_positives += 1
        elif not predicted_gap and actual_gap:
            self.false_negatives += 1
            
        # Update validation history for trending
        metrics = self.calculate_validation_metrics()
        self.validation_history.append([
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ])
        
        # Keep validation history manageable
        if len(self.validation_history) > self.temporal_consistency_window:
            self.validation_history.pop(0)

    def calculate_validation_metrics(self):
        """Calculate precision, recall, and F1 score."""
        try:
            precision = self.true_positives / (self.true_positives + self.false_positives)
        except ZeroDivisionError:
            precision = 0
            
        try:
            recall = self.true_positives / (self.true_positives + self.false_negatives)
        except ZeroDivisionError:
            recall = 0
            
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }

    def validate_detection(self, detected_gap):
        """Validate a detected gap against ground truth."""
        if not self.ground_truth_gaps:
            return None
            
        for truth_gap in self.ground_truth_gaps:
            # Calculate temporal overlap
            overlap = (
                detected_gap["start_angle"] <= truth_gap["end_angle"] and
                detected_gap["end_angle"] >= truth_gap["start_angle"]
            )
            
            if overlap:
                # Calculate IoU (Intersection over Union)
                intersection_start = max(detected_gap["start_angle"], truth_gap["start_angle"])
                intersection_end = min(detected_gap["end_angle"], truth_gap["end_angle"])
                union_start = min(detected_gap["start_angle"], truth_gap["start_angle"])
                union_end = max(detected_gap["end_angle"], truth_gap["end_angle"])
                
                intersection = intersection_end - intersection_start
                union = union_end - union_start
                
                iou = intersection / union if union > 0 else 0
                
                # Calculate distance similarity
                distance_diff = abs(detected_gap["distance"] - truth_gap["distance"])
                distance_similarity = 1 / (1 + distance_diff/10)  # Normalized distance difference
                
                # Combined validation score
                validation_score = (iou + distance_similarity) / 2
                
                return validation_score >= 0.5  # Consider it a match if validation score >= 0.5
                
        return False

    def set_ground_truth(self, ground_truth_gaps):
        """Set ground truth data for validation."""
        self.ground_truth_gaps = ground_truth_gaps
        # Reset validation metrics when new ground truth is set
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.validation_history = []

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
        
        # Calculate smoothed distance
        smoothed_distance = np.mean(self.distance_buffer) if self.distance_buffer else filtered_distance
        
        # Calculate rate of change
        rate_of_change = 0
        if self.last_distances:
            rate_of_change = abs(smoothed_distance - self.last_distances[-1])
        self.last_distances.append(smoothed_distance)
        if len(self.last_distances) > 3:
            self.last_distances.pop(0)
        
        # Update dynamic threshold
        self.update_dynamic_threshold(smoothed_distance)
        
        # Calculate confidence score
        confidence = self.calculate_confidence_score(smoothed_distance, angle)
        
        # Check for anomalies
        is_anomaly = False
        if len(self.data_points) > 50:
            X = np.array([[angle, smoothed_distance]])
            X_scaled = self.scaler.transform(X)
            is_anomaly = self.isolation_forest.predict(X_scaled)[0] == -1
            self.anomaly_history.append(1 if is_anomaly else 0)
            if len(self.anomaly_history) > self.temporal_consistency_window:
                self.anomaly_history.pop(0)
        
        return smoothed_distance, confidence, is_anomaly, rate_of_change
    
    
    ######### Part 3 #################
    
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
        self.ax3.set_title('Metrics & Confidence', pad=10)

        if self.baseline_distance:
            for ax in [self.ax1, self.ax2]:
                ax.axhline(y=self.baseline_distance, color='yellow', linestyle='--', alpha=0.3)
                ax.text(-28, self.baseline_distance, f'Baseline: {self.baseline_distance:.1f}cm',
                        color='yellow', alpha=0.5)
                gap_line = self.baseline_distance + self.gap_threshold
                ax.axhline(y=gap_line, color='red', linestyle='--', alpha=0.3)
                ax.text(-28, gap_line, f'Gap Threshold: {gap_line:.1f}cm', color='red', alpha=0.5)

    def plot_polar(self):
        """Enhanced polar plot visualization."""
        if not self.polar_ax:
            return

        self.polar_ax.clear()
        
        # Plot all data points with confidence-based coloring
        if self.data_points:
            angles = np.radians([point[0] for point in self.data_points])
            distances = [point[1] for point in self.data_points]
            
            # Create confidence-based colormap
            colors = np.array([score for score in self.confidence_scores[-len(angles):]] 
                            if len(self.confidence_scores) >= len(angles) 
                            else [0.5] * len(angles))
            
            scatter = self.polar_ax.scatter(angles, distances, 
                                          c=colors, cmap='viridis', 
                                          s=30, alpha=0.6)
            
            # Add colorbar
            plt.colorbar(scatter, ax=self.polar_ax, label='Confidence')

        # Plot baseline and threshold circles
        if self.baseline_distance:
            angles = np.linspace(0, 2*np.pi, 100)
            baseline_circle = np.full_like(angles, self.baseline_distance)
            self.polar_ax.plot(angles, baseline_circle, 'y--', alpha=0.3, label='Baseline')
            
            threshold_circle = np.full_like(angles, self.baseline_distance + self.gap_threshold)
            self.polar_ax.plot(angles, threshold_circle, 'r--', alpha=0.3, label='Threshold')

        # Highlight detected gaps with confidence-based coloring
        for gap in self.detected_gaps:
            if abs(gap["end_angle"] - gap["start_angle"]) >= self.min_gap_width:
                gap_angles = np.linspace(np.radians(gap["start_angle"]), 
                                       np.radians(gap["end_angle"]), 20)
                gap_distances = np.full_like(gap_angles, gap["distance"])
                
                # Use confidence for color intensity
                confidence = gap.get("confidence", 0.5)
                alpha = max(0.3, confidence)
                
                self.polar_ax.plot(gap_angles, gap_distances, 'r-', 
                                 linewidth=2, alpha=alpha)
                
                # Add gap annotations with validation score if available
                mid_angle = np.radians((gap["start_angle"] + gap["end_angle"]) / 2)
                annotation_text = f'{self.calculate_gap_width(gap):.1f}cm'
                if "validation_score" in gap:
                    annotation_text += f'\nVal: {gap["validation_score"]:.2f}'
                self.polar_ax.text(mid_angle, gap["distance"], 
                                 annotation_text,
                                 fontsize=8, color='white')

        # Set plot properties
        self.polar_ax.set_title("Polar View")
        self.polar_ax.set_theta_zero_location("N")
        self.polar_ax.set_theta_direction(-1)
        self.polar_ax.legend(loc='upper right', fontsize=8)

    def update_plot(self, frame):
        """Update plot with enhanced metrics."""
        data = self.read_sensor_data()
        if not data:
            return

        angle, distance, is_gap, confidence = data
        
        # Update visualization
        self.init_plots()
        
        # Update metrics display
        metrics = self.calculate_validation_metrics()
        validation_text = (
            f'Validation Metrics:\n'
            f'Precision: {metrics["precision"]:.3f}\n'
            f'Recall: {metrics["recall"]:.3f}\n'
            f'F1 Score: {metrics["f1_score"]:.3f}\n\n'
            f'Confidence Metrics:\n'
            f'Current: {confidence:.3f}\n'
            f'Avg (Window): {np.mean(self.confidence_history):.3f}\n'
            f'Std (Window): {np.std(self.confidence_history):.3f}'
        )
        
        # Position text with enhanced visibility
        self.ax3.text(0.02, 0.98, validation_text,
                     transform=self.ax3.transAxes,
                     verticalalignment='top',
                     fontsize=8,
                     bbox=dict(facecolor='black', alpha=0.5))
        
        # Plot data point
        rad = np.radians(angle - self.center_angle)
        x = distance * np.sin(rad)
        y = distance * np.cos(rad)
        
        color = 'red' if is_gap else 'green'
        self.ax1.scatter(x, y, c=color, s=30, alpha=max(0.3, confidence))
        
        # Add scan line
        scan_x = 100 * np.sin(rad)
        scan_y = 100 * np.cos(rad)
        self.ax1.plot([0, scan_x], [0, scan_y], 'r-', alpha=0.3)
        
        # Visualize detected gaps
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
    
    
    ######Part 4###################
    
    # Add gap width and confidence annotations
                gap_width = self.calculate_gap_width(gap)
                self.ax1.text(xs[len(xs)//2], ys[len(ys)//2], 
                            f'Width: {gap_width:.1f}cm\nConf: {gap["confidence"]:.2f}',
                            color='white', alpha=0.7, fontsize=8)
        
        # Plot measurement point in analysis view
        self.ax2.scatter(angle - self.center_angle, distance, c=color, s=30, alpha=max(0.3, confidence))
        
        # Update status information with enhanced metrics
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
        
        # Update additional visualizations
        self.plot_polar()
        self.update_confidence_trends(angle, distance, confidence)
        self.enhanced_visualization()

    def update_confidence_trends(self, angle, distance, confidence):
        """Update and plot confidence trends."""
        # Store angle-specific distance history
        if angle not in self.angle_distance_history:
            self.angle_distance_history[angle] = []
        self.angle_distance_history[angle].append(distance)
        
        # Keep history size manageable
        if len(self.angle_distance_history[angle]) > self.spatial_consistency_window:
            self.angle_distance_history[angle].pop(0)
        
        # Plot confidence trends
        if self.confidence_history:
            angles = np.linspace(self.min_angle, self.max_angle, len(self.confidence_history))
            self.ax3.plot(angles, self.confidence_history, 'b-', alpha=0.5)
            self.ax3.fill_between(angles, 0, self.confidence_history, color='blue', alpha=0.2)

    def enhanced_visualization(self):
        """Additional visualization enhancements."""
        if len(self.data_points) >= 10:
            self.apply_clustering()
            
            # Plot confidence trends with smoothing
            confidence_data = np.array(self.confidence_history)
            if len(confidence_data) > 0:
                window_size = min(20, len(confidence_data))
                smoothed_confidence = np.convolve(confidence_data, 
                                                np.ones(window_size)/window_size, 
                                                mode='valid')
                x_vals = np.arange(len(smoothed_confidence))
                self.ax3.plot(x_vals, smoothed_confidence, 'g-', 
                            alpha=0.5, label='Smoothed Confidence')
                self.ax3.fill_between(x_vals, 0, smoothed_confidence, 
                                    color='green', alpha=0.1)
            
            # Plot validation metrics history
            if len(self.validation_history) > 0:
                metrics_data = np.array(self.validation_history)
                x_vals = np.arange(len(metrics_data))
                self.ax3.plot(x_vals, metrics_data[:, 0], 'b-', label='Precision', alpha=0.5)
                self.ax3.plot(x_vals, metrics_data[:, 1], 'r-', label='Recall', alpha=0.5)
                self.ax3.plot(x_vals, metrics_data[:, 2], 'y-', label='F1', alpha=0.5)
            
            self.ax3.legend(loc='upper right', fontsize=8)
            self.ax3.set_ylim(0, 1)
            self.ax3.set_xlabel('Time')
            self.ax3.set_ylabel('Score')

    def calculate_gap_width(self, gap):
        """Calculate physical width of gap in centimeters."""
        angle_diff = gap["end_angle"] - gap["start_angle"]
        avg_distance = gap["distance"]
        return 2 * avg_distance * np.tan(np.radians(angle_diff / 2))

    def predict_gap_size(self, start_angle, end_angle, distance):
        """Predict gap size using regression model."""
        if self.regression_model:
            features = np.array([[start_angle, end_angle, distance]])
            predicted_size = self.regression_model.predict(features)[0]
            # Add confidence based on model's historical performance
            confidence = 1.0  # You could calculate this based on model metrics
            return predicted_size, confidence
        return self.calculate_gap_width({
            "start_angle": start_angle,
            "end_angle": end_angle,
            "distance": distance
        }), 0.5

    def predict_gap_size_mlp(self, start_angle, end_angle, distance):
        """Predict gap size using neural network model."""
        if self.mlp_model:
            features = np.array([[start_angle, end_angle, distance]])
            predicted_size = self.mlp_model.predict(features)[0]
            # Get prediction confidence from model
            confidence = self.mlp_model.predict_proba(features)[0].max() if hasattr(self.mlp_model, 'predict_proba') else 0.5
            return predicted_size, confidence
        return self.calculate_gap_width({
            "start_angle": start_angle,
            "end_angle": end_angle,
            "distance": distance
        }), 0.5
        
        
    ############# part 5 ################
    
    def apply_clustering(self):
        """Apply advanced clustering for gap detection."""
        if len(self.data_points) < 10:
            return

        # Prepare data for clustering
        X = np.array([[point[0], point[1]] for point in self.data_points])
        
        # Normalize features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply DBSCAN clustering with dynamic parameters
        eps = 0.3 * np.std(X_scaled)  # Dynamic epsilon based on data spread
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(X_scaled)
        
        # Process clusters
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            # Get points in current cluster
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]
            
            # Calculate cluster properties
            avg_distance = np.mean(cluster_points[:, 1])
            min_angle = np.min(cluster_points[:, 0])
            max_angle = np.max(cluster_points[:, 0])
            
            # Check if cluster represents a gap
            if (self.baseline_distance and 
                avg_distance > self.baseline_distance + self.gap_threshold):
                
                # Calculate advanced cluster properties
                spread = np.std(cluster_points[:, 1])
                density = len(cluster_points) / (max_angle - min_angle)
                temporal_consistency = self.calculate_temporal_consistency(
                    min_angle, max_angle)
                
                # Calculate confidence based on multiple factors
                cluster_confidence = self.calculate_cluster_confidence(
                    spread, density, avg_distance, temporal_consistency)
                
                # Record gap if confidence is sufficient
                if cluster_confidence > self.min_confidence_threshold:
                    self.record_gap(min_angle, max_angle, avg_distance, cluster_confidence)

    def calculate_temporal_consistency(self, start_angle, end_angle):
        """Calculate temporal consistency of gap detection."""
        # Check recent gap detections in similar angular region
        recent_gaps = [gap for gap in self.detected_gaps[-5:] 
                      if abs(gap["start_angle"] - start_angle) < 10 and
                         abs(gap["end_angle"] - end_angle) < 10]
        
        if not recent_gaps:
            return 0.5
        
        # Calculate consistency score based on recent detections
        consistencies = []
        for gap in recent_gaps:
            angle_consistency = 1 - (abs(gap["start_angle"] - start_angle) +
                                   abs(gap["end_angle"] - end_angle)) / 20
            consistencies.append(angle_consistency)
        
        return np.mean(consistencies)

    def calculate_cluster_confidence(self, spread, density, distance, temporal_consistency):
        """Calculate enhanced confidence score for a cluster."""
        # Normalize factors
        spread_factor = 1 / (1 + spread/10)  # Lower spread -> higher confidence
        density_factor = min(density/5, 1)    # Higher density -> higher confidence
        
        # Distance factor (confidence decreases with distance)
        distance_factor = 1 / (1 + abs(distance - self.baseline_distance)/100)
        
        # Weight the factors based on importance
        weighted_factors = [
            (spread_factor, 0.3),
            (density_factor, 0.3),
            (distance_factor, 0.2),
            (temporal_consistency, 0.2)
        ]
        
        # Calculate weighted confidence
        confidence = sum(factor * weight for factor, weight in weighted_factors)
        return confidence

    def init_csv(self):
        """Initialize CSV logging with enhanced metrics."""
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'angle', 'raw_distance', 'filtered_distance',
                'is_gap', 'confidence', 'baseline_distance', 'threshold',
                'anomaly_score', 'validation_score', 'precision', 'recall',
                'f1_score', 'temporal_consistency', 'spatial_consistency',
                'cluster_confidence', 'prediction_confidence'
            ])

    def log_to_csv(self, angle, raw_distance, filtered_distance, is_gap, confidence):
        """Enhanced CSV logging with all metrics."""
        try:
            metrics = self.calculate_validation_metrics()
            temporal_consistency = np.mean(self.confidence_history) if self.confidence_history else 0
            
            spatial_consistency = (
                np.std(self.angle_distance_history.get(int(angle), []))
                if angle in self.angle_distance_history else 0
            )
            
            # Get additional confidence metrics
            cluster_confidence = self.calculate_cluster_confidence(
                np.std([raw_distance, filtered_distance]),
                1.0,  # Default density
                filtered_distance,
                temporal_consistency
            )
            
            # Get prediction confidence
            _, prediction_confidence = self.predict_gap_size(
                angle, angle, filtered_distance)
            
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
                    if self.isolation_forest else "N/A",
                    f"{self.validate_detection({'start_angle': angle, 'end_angle': angle, 'distance': filtered_distance}):.3f}"
                    if self.ground_truth_gaps else "N/A",
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1_score']:.3f}",
                    f"{temporal_consistency:.3f}",
                    f"{spatial_consistency:.3f}",
                    f"{cluster_confidence:.3f}",
                    f"{prediction_confidence:.3f}"
                ])
        except Exception as e:
            print(f"Error logging to CSV: {e}")

    def collect_data(self, duration=60):
        """Collect and visualize data with enhanced monitoring."""
        print("Starting enhanced gap detection scan...")
        self.is_collecting = True
        start_time = time.time()
        
        try:
            ani = FuncAnimation(self.fig, self.update_plot,
                              interval=50, cache_frame_data=False)
            
            def on_close(event):
                self.is_collecting = False
            self.fig.canvas.mpl_connect('close_event', on_close)
            
            plt.show()
            
            # Final metrics report
            if time.time() - start_time >= duration:
                metrics = self.calculate_validation_metrics()
                print("\nFinal Detection Metrics:")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1 Score: {metrics['f1_score']:.3f}")
                print(f"Total Gaps Detected: {len(self.detected_gaps)}")
            
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
        
##############part 06 ######################
def main():
    """Main execution function."""
    port = 'COM7'  # Adjust port as needed
    detector = EnhancedGapDetector(port=port)
    
    try:
        if not detector.connect():
            print("Failed to initialize the gap detector. Please check the connection and try again.")
            return
        
        # Optional: Set ground truth data for validation
        ground_truth = [
            {"start_angle": 85, "end_angle": 95, "distance": 50},
            {"start_angle": 120, "end_angle": 125, "distance": 45}
        ]
        detector.set_ground_truth(ground_truth)
        
        # Start data collection and visualization
        detector.collect_data(duration=60)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        detector.close()

if __name__ == "__main__":
    main()
                                        