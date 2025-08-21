#!/usr/bin/env python3
"""
Generate Result and Comparison Images
Creates {no}_xxx_result.png and {no}_xxx_comparison.png files
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_comparison_images():
    """Create result and comparison images for all processed files"""
    
    # Input and output paths
    input_folder = "input"
    output_folder = "output/pipeline_final_results"
    final_output = "output"
    
    # Ensure output directory exists
    os.makedirs(final_output, exist_ok=True)
    
    # Get all input files
    input_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    input_files.sort()
    
    print(f"🎯 Generating comparisons for {len(input_files)} files...")
    
    for i, filename in enumerate(input_files, 1):
        print(f"\n📄 Processing file {i}: {filename}")
        
        # Get file extension
        name, ext = os.path.splitext(filename)
        file_type = ext.lower()
        
        # Skip PDFs for now (we'll handle them later)
        if file_type == '.pdf':
            print(f"   ⚠️  Skipping PDF file: {filename}")
            continue
        
        # Load original image
        original_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(original_path)
        
        if original_image is None:
            print(f"   ❌ Could not load original image: {filename}")
            continue
        
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Get the final processed image from pipeline output
        final_result = get_final_processed_image(output_folder, filename)
        
        if final_result is None:
            print(f"   ❌ No processed result found for: {filename}")
            continue
        
        # Create result image (just the final processed image)
        result_filename = f"{i:02d}_{name}_result.png"
        result_path = os.path.join(final_output, result_filename)
        cv2.imwrite(result_path, final_result)
        print(f"   ✅ Created result: {result_filename}")
        
        # Create comparison image (original vs. result side by side)
        comparison_filename = f"{i:02d}_{name}_comparison.png"
        comparison_path = os.path.join(final_output, comparison_filename)
        
        # Create side-by-side comparison
        create_side_by_side_comparison(original_rgb, final_result, comparison_path, filename)
        print(f"   ✅ Created comparison: {comparison_filename}")
    
    print(f"\n🎉 All comparisons generated in: {final_output}")

def get_final_processed_image(pipeline_output_folder, original_filename):
    """Get the final processed image from pipeline output"""
    
    # Get the folder name for this file
    name, ext = os.path.splitext(original_filename)
    folder_name = f"{name}_{ext[1:]}"  # e.g., "orientation_1_png"
    
    folder_path = os.path.join(pipeline_output_folder, folder_name)
    
    if not os.path.exists(folder_path):
        return None
    
    # Look for the final processed image (after all tasks)
    # Priority: oriented > cropped > skew_corrected > original
    possible_names = [
        f"oriented_{original_filename}",
        f"cropped_{original_filename}",
        f"skew_corrected_{original_filename}",
        original_filename
    ]
    
    for possible_name in possible_names:
        possible_path = os.path.join(folder_path, possible_name)
        if os.path.exists(possible_path):
            # Load the image
            image = cv2.imread(possible_path)
            if image is not None:
                return image
    
    # If no pipeline output found, try to use the original script results directly
    # This ensures we get the exact same results as the original scripts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # For cropping.jpeg, use the extreme tight cropping result
    if original_filename == "cropping.jpeg":
        original_result = os.path.join(base_dir, "..", "2.cropping", "output", "8_extreme_tight_text_result.png")
        if os.path.exists(original_result):
            image = cv2.imread(original_result)
            if image is not None:
                return image
    
    # For orientation files, use the ML-based orientation results
    elif "orientation_1" in original_filename:
        original_result = os.path.join(base_dir, "..", "3.fix_orientation", "output", "6_ml_orientation_result_1.png")
        if os.path.exists(original_result):
            image = cv2.imread(original_result)
            if image is not None:
                return image
    
    elif "orientation_2" in original_filename:
        original_result = os.path.join(base_dir, "..", "3.fix_orientation", "output", "6_ml_orientation_result_2.png")
        if os.path.exists(original_result):
            image = cv2.imread(original_result)
            if image is not None:
                return image
    
    # For deskew-pdf-before, use the skew corrected result
    elif "deskew-pdf-before" in original_filename:
        original_result = os.path.join(base_dir, "..", "1. final_skew_detector", "output", "deskewed", "deskew-pdf-before_deskewed.png")
        if os.path.exists(original_result):
            image = cv2.imread(original_result)
            if image is not None:
                return image
    
    return None

def create_side_by_side_comparison(original, processed, output_path, filename):
    """Create a side-by-side comparison image"""
    
    # Ensure both images have the same height for comparison
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    # Resize processed image to match original height if needed
    if h1 != h2:
        scale = h1 / h2
        new_w = int(w2 * scale)
        processed_resized = cv2.resize(processed, (new_w, h1))
    else:
        processed_resized = processed
    
    # Create the side-by-side image
    combined_width = w1 + processed_resized.shape[1]
    combined_height = max(h1, processed_resized.shape[0])
    
    # Create white background
    combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
    
    # Place original image on the left
    combined_image[:h1, :w1] = original
    
    # Place processed image on the right
    processed_h, processed_w = processed_resized.shape[:2]
    combined_image[:processed_h, w1:w1+processed_w] = processed_resized
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 0)  # Black text
    
    # Add "Original" label
    cv2.putText(combined_image, "Original", (10, 30), font, font_scale, color, thickness)
    
    # Add "Processed" label
    cv2.putText(combined_image, "Processed", (w1 + 10, 30), font, font_scale, color, thickness)
    
    # Add filename at the bottom
    cv2.putText(combined_image, f"File: {filename}", (10, combined_height - 20), font, 0.7, color, 1)
    
    # Save the comparison
    cv2.imwrite(output_path, combined_image)

def main():
    """Main function"""
    
    print("🖼️  Pipeline Result and Comparison Generator")
    print("=" * 50)
    
    # Check if pipeline output exists
    if not os.path.exists("output/pipeline_final_results"):
        print("⚠️  Pipeline output not found, using original script results directly")
        print("   This ensures you get the exact same results as the original scripts")
    
    # Generate comparisons
    create_comparison_images()
    
    print("\n📁 Files generated:")
    print("   - {no:02d}_xxx_result.png: Final processed image")
    print("   - {no:02d}_xxx_comparison.png: Original vs. Processed comparison")
    print("\n💡 Check the output/ folder for your results!")

if __name__ == "__main__":
    main()
