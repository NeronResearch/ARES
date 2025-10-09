import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_brightness(image_path):
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert("L")
    data = np.array(img)

    # Compute brightness statistics
    mean_brightness = np.mean(data)
    median_brightness = np.median(data)
    min_brightness = np.min(data)
    max_brightness = np.max(data)
    non_zero_pixels = np.count_nonzero(data)

    print(f"Image: {image_path}")
    print(f"Dimensions: {data.shape[1]} x {data.shape[0]}")
    print(f"Mean Brightness: {mean_brightness:.2f}")
    print(f"Median Brightness: {median_brightness:.2f}")
    print(f"Min Brightness: {min_brightness}")
    print(f"Max Brightness: {max_brightness}")
    print(f"Total Non-Zero Pixels: {non_zero_pixels}")

    # Plot histogram of brightness values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data.flatten(), bins=256, color='gray', alpha=0.8)
    plt.title("Brightness Histogram")
    plt.xlabel("Brightness (0–255)")
    plt.ylabel("Pixel Count")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot average brightness per row
    row_brightness = np.mean(data, axis=1)
    plt.subplot(1, 2, 2)
    plt.plot(row_brightness, color='black')
    plt.title("Average Brightness Per Row")
    plt.xlabel("Row")
    plt.ylabel("Brightness (0–255)")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_brightness.py <path_to_grayscale_png>")
        sys.exit(1)

    image_path = sys.argv[1]
    analyze_brightness(image_path)
