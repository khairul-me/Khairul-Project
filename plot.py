import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Filepath
file_path = r"K:\Khairul_Ultrasonic\gap_scan_20241122_194117.csv"

# Load the CSV data
df = pd.read_csv(file_path)

# Parse the necessary columns
angles = df['angle']
distances = df['filtered_distance']
confidences = df['confidence']
is_gap = df['is_gap'].astype(bool)

# Fancy 2D Plot
def plot_2d_gap_detection():
    plt.figure(figsize=(12, 6))
    plt.scatter(angles, distances, c=confidences, cmap='viridis', s=50, label="Distance Data")
    plt.colorbar(label='Confidence Level')
    plt.axhline(y=df['baseline_distance'].iloc[0], color='yellow', linestyle='--', label='Baseline')
    plt.axhline(y=df['threshold'].iloc[0], color='red', linestyle='--', label='Threshold')
    plt.scatter(angles[is_gap], distances[is_gap], c='red', s=80, label='Detected Gaps', edgecolors='black')

    plt.title("Gap Detection Visualization (2D)", fontsize=14)
    plt.xlabel("Angle (Degrees)", fontsize=12)
    plt.ylabel("Filtered Distance (cm)", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)  # Moved to the top-right corner
    plt.grid(alpha=0.3)
    plt.show()

# Fancy 3D Plot
def plot_3d_gap_detection():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter plot
    scatter = ax.scatter(
        angles,
        distances,
        confidences,
        c=confidences,
        cmap='coolwarm',
        s=50,
        label='Data Points'
    )
    ax.scatter(angles[is_gap], distances[is_gap], confidences[is_gap],
               c='red', s=80, label='Detected Gaps', edgecolors='black')

    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.2)
    cbar.set_label('Confidence Level', fontsize=12)

    # Labels and title
    ax.set_title("Gap Detection in 3D Space", fontsize=14)
    ax.set_xlabel("Angle (Degrees)", fontsize=12)
    ax.set_ylabel("Filtered Distance (cm)", fontsize=12)
    ax.set_zlabel("Confidence Level", fontsize=12)

    plt.legend(loc='upper right', fontsize=10)  # Moved to the top-right corner
    plt.show()

# Call the plotting functions
plot_2d_gap_detection()
plot_3d_gap_detection()
