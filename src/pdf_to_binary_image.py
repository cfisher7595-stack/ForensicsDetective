#!/usr/bin/env python3
"""
PDF Binary-to-Image Converter for Forensic Analysis

Converts PDF files to grayscale images where each byte value (0-255) 
represents a pixel intensity. This reveals binary-level signatures 
embedded by different PDF generation engines.
"""

import os
import math
from PIL import Image
import numpy as np

def pdf_to_binary_image(pdf_path, output_path, width=None):
    """
    Convert PDF binary data to grayscale image.
    
    Args:
        pdf_path (str): Path to input PDF file
        output_path (str): Path for output PNG image
        width (int): Image width in pixels (auto-calculated if None)
    """
    # Read binary data from PDF
    with open(pdf_path, 'rb') as f:
        binary_data = f.read()
    
    # Convert bytes to numpy array of uint8 values (0-255)
    byte_array = np.frombuffer(binary_data, dtype=np.uint8)
    
    # Calculate image dimensions
    total_pixels = len(byte_array)
    
    if width is None:
        # Calculate square-ish dimensions
        width = int(math.sqrt(total_pixels))
    
    height = math.ceil(total_pixels / width)
    
    # Pad with zeros if necessary to fill rectangle
    required_pixels = width * height
    if total_pixels < required_pixels:
        padding = np.zeros(required_pixels - total_pixels, dtype=np.uint8)
        byte_array = np.concatenate([byte_array, padding])
    
    # Reshape to 2D array (height, width)
    image_array = byte_array[:width * height].reshape(height, width)
    
    # Create grayscale image
    image = Image.fromarray(image_array, mode='L')
    
    # Save as PNG
    image.save(output_path, 'PNG')
    
    return image_array.shape

def convert_pdf_directory(input_dir, output_dir):
    """
    Convert all PDFs in a directory to binary images.
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save PNG images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    print(f"Converting {len(pdf_files)} PDF files to binary images...")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(input_dir, pdf_file)
        png_file = pdf_file.replace('.pdf', '.png')
        png_path = os.path.join(output_dir, png_file)
        
        try:
            shape = pdf_to_binary_image(pdf_path, png_path)
            print(f"[{i:3d}/{len(pdf_files)}] {pdf_file} -> {png_file} ({shape[1]}x{shape[0]})")
        except Exception as e:
            print(f"[{i:3d}/{len(pdf_files)}] ERROR converting {pdf_file}: {e}")

def main():
    """Convert Word, Google Docs, and Python PDFs to binary images."""
    
    # Convert Word-generated PDFs
    print("=== Converting Word-generated PDFs ===")
    convert_pdf_directory('word_pdfs', 'word_pdfs_png')
    
    print("\n=== Converting Google Docs-generated PDFs ===")
    convert_pdf_directory('google_docs_pdfs', 'google_docs_pdfs_png')
    
    print("\n=== Converting Python-generated PDFs ===")
    convert_pdf_directory('python_pdfs', 'python_pdfs_png')
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()