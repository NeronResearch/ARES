#!/usr/bin/env python3
import sys
from PIL import Image

def convert_to_grayscale(input_path):
    # Open the original image
    img = Image.open(input_path)

    # Convert to 1-channel grayscale ("L" mode = 8-bit pixels, black and white)
    gray_img = img.convert("L")

    # Save the grayscale image
    output_path = input_path.rsplit('.', 1)[0] + "_grayscale.jpg"
    gray_img.save(output_path)

    print(f"Grayscale image saved as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 convert_grayscale.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    convert_to_grayscale(image_path)
