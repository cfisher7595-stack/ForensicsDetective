#!/usr/bin/env python3
"""
PDF Binary Image Comparison Tool

Creates side-by-side visualizations of Word, Google Docs, and Python-generated
PDF binary images to understand what the classifiers are seeing.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def create_side_by_side_comparison(word_dir='word_pdfs_png', google_dir='google_docs_pdfs_png', 
                                 python_dir='python_pdfs_png', output_dir='comparison_images',
                                 num_comparisons=10):
    """
    Create side-by-side comparisons of the same document across all three generation methods.
    
    Args:
        word_dir (str): Directory containing Word-generated PDF images
        google_dir (str): Directory containing Google Docs-generated PDF images  
        python_dir (str): Directory containing Python-generated PDF images
        output_dir (str): Directory to save comparison images
        num_comparisons (int): Number of comparison images to create
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get common files across all three directories
    word_files = set([f.replace('.png', '') for f in os.listdir(word_dir) if f.endswith('.png')])
    google_files = set([f.replace('.png', '') for f in os.listdir(google_dir) if f.endswith('.png')])
    python_files = set([f.replace('.png', '') for f in os.listdir(python_dir) if f.endswith('.png')])
    
    # Find files that exist in all three directories
    common_files = list(word_files & google_files & python_files)
    
    print(f"Found {len(common_files)} files common to all three PDF types")
    
    # Limit to requested number
    common_files = common_files[:num_comparisons]
    
    print(f"Creating {len(common_files)} comparison images...")
    
    for i, base_filename in enumerate(common_files, 1):
        try:
            # Load the three images
            word_path = os.path.join(word_dir, f"{base_filename}.png")
            google_path = os.path.join(google_dir, f"{base_filename}.png")
            python_path = os.path.join(python_dir, f"{base_filename}.png")
            
            word_img = Image.open(word_path).convert('RGB')
            google_img = Image.open(google_path).convert('RGB')
            python_img = Image.open(python_path).convert('RGB')
            
            # Get original dimensions for labels
            word_dims = f"{word_img.size[0]}x{word_img.size[1]}"
            google_dims = f"{google_img.size[0]}x{google_img.size[1]}"
            python_dims = f"{python_img.size[0]}x{python_img.size[1]}"
            
            # Resize all to same height for comparison (keep aspect ratio)
            target_height = 300
            
            word_ratio = target_height / word_img.size[1]
            google_ratio = target_height / google_img.size[1]
            python_ratio = target_height / python_img.size[1]
            
            word_resized = word_img.resize((int(word_img.size[0] * word_ratio), target_height), Image.LANCZOS)
            google_resized = google_img.resize((int(google_img.size[0] * google_ratio), target_height), Image.LANCZOS)
            python_resized = python_img.resize((int(python_img.size[0] * python_ratio), target_height), Image.LANCZOS)
            
            # Create comparison image
            padding = 20
            label_height = 30
            total_width = word_resized.size[0] + google_resized.size[0] + python_resized.size[0] + 4 * padding
            total_height = target_height + label_height + 2 * padding
            
            comparison = Image.new('RGB', (total_width, total_height), 'white')
            
            # Paste images
            x_offset = padding
            comparison.paste(word_resized, (x_offset, padding + label_height))
            x_offset += word_resized.size[0] + padding
            comparison.paste(google_resized, (x_offset, padding + label_height))
            x_offset += google_resized.size[0] + padding
            comparison.paste(python_resized, (x_offset, padding + label_height))
            
            # Add labels
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Labels with dimensions
            x_offset = padding
            draw.text((x_offset, 5), f"Word PDF\n{word_dims}", fill='black', font=font)
            x_offset += word_resized.size[0] + padding
            draw.text((x_offset, 5), f"Google Docs PDF\n{google_dims}", fill='black', font=font)
            x_offset += google_resized.size[0] + padding
            draw.text((x_offset, 5), f"Python PDF\n{python_dims}", fill='black', font=font)
            
            # Save comparison
            output_path = os.path.join(output_dir, f"{i:02d}_{base_filename}_comparison.png")
            comparison.save(output_path)
            
            print(f"[{i:2d}/{len(common_files)}] Created: {base_filename}_comparison.png")
            
        except Exception as e:
            print(f"[{i:2d}/{len(common_files)}] ERROR with {base_filename}: {e}")
    
    print(f"\nComparison images saved to: {output_dir}/")

def create_intensity_histograms(word_dir='word_pdfs_png', google_dir='google_docs_pdfs_png',
                              python_dir='python_pdfs_png', output_dir='comparison_images',
                              num_samples=10):
    """Create histograms showing intensity distributions for each PDF type."""
    
    print("\nCreating intensity distribution histograms...")
    
    # Sample files
    word_files = [f for f in os.listdir(word_dir) if f.endswith('.png')][:num_samples]
    google_files = [f for f in os.listdir(google_dir) if f.endswith('.png')][:num_samples]
    python_files = [f for f in os.listdir(python_dir) if f.endswith('.png')][:num_samples]
    
    # Collect pixel intensities
    word_intensities = []
    google_intensities = []
    python_intensities = []
    
    for filename in word_files:
        img = Image.open(os.path.join(word_dir, filename)).convert('L')
        word_intensities.extend(list(np.array(img).flatten()))
    
    for filename in google_files:
        img = Image.open(os.path.join(google_dir, filename)).convert('L')
        google_intensities.extend(list(np.array(img).flatten()))
    
    for filename in python_files:
        img = Image.open(os.path.join(python_dir, filename)).convert('L')
        python_intensities.extend(list(np.array(img).flatten()))
    
    # Create histogram plot
    plt.figure(figsize=(12, 6))
    
    plt.hist(word_intensities, bins=50, alpha=0.7, label=f'Word (mean: {np.mean(word_intensities):.1f})', color='blue')
    plt.hist(google_intensities, bins=50, alpha=0.7, label=f'Google (mean: {np.mean(google_intensities):.1f})', color='green')
    plt.hist(python_intensities, bins=50, alpha=0.7, label=f'Python (mean: {np.mean(python_intensities):.1f})', color='red')
    
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.title('Pixel Intensity Distributions by PDF Generation Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    histogram_path = os.path.join(output_dir, 'intensity_distributions.png')
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Intensity histogram saved to: {histogram_path}")

def main():
    """Create comparison visualizations."""
    
    print("PDF Binary Image Comparison Tool")
    print("=" * 40)
    
    # Create side-by-side comparisons
    create_side_by_side_comparison(num_comparisons=10)
    
    # Create intensity histograms
    create_intensity_histograms(num_samples=20)
    
    print("\nComparison analysis complete!")
    print("Check the 'comparison_images/' directory to see:")
    print("- Side-by-side binary image comparisons")
    print("- Pixel intensity distribution histograms")

if __name__ == "__main__":
    main()