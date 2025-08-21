#!/usr/bin/env python3
"""
Generate Direct Comparisons
Creates result and comparison images directly from original script outputs
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_direct_comparisons():
    """Create result and comparison images directly from original script outputs"""
    
    # Input and output paths
    input_folder = "input"
    final_output = "output"
    
    # Ensure output directory exists
    os.makedirs(final_output, exist_ok=True)
    
    # Get all input files
    input_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    input_files.sort()
    
    print(f"🎯 Generating direct comparisons for {len(input_files)} files...")
    
    for i, filename in enumerate(input_files, 1):
        print(f"\n📄 Processing file {i}: {filename}")
        
        # Get file extension
        name, ext = os.path.splitext(filename)
        file_type = ext.lower()
        
        # Skip PDFs for now
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
        
        # Get the final processed image from original script outputs
        final_result = get_original_script_result(filename)
        
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

def get_original_script_result(filename):
    """Get the final processed image from original script outputs"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # For cropping.jpeg, use the extreme tight cropping result
    if filename == "cropping.jpeg":
        result_path = os.path.join(base_dir, "..", "2.cropping", "output", "8_extreme_tight_text_result.png")
        if os.path.exists(result_path):
            image = cv2.imread(result_path)
            if image is not None:
                print(f"   📁 Using extreme tight cropping result: {result_path}")
                return image
    
    # For orientation files, use the ML-based orientation results
    elif filename == "orientation_1.png":
        result_path = os.path.join(base_dir, "..", "3.fix_orientation", "output", "6_ml_orientation_result_1.png")
        if os.path.exists(result_path):
            image = cv2.imread(result_path)
            if image is not None:
                print(f"   📁 Using ML orientation result 1: {result_path}")
                return image
    
    elif filename == "orientation_2.png":
        result_path = os.path.join(base_dir, "..", "3.fix_orientation", "output", "6_ml_orientation_result_2.png")
        if os.path.exists(result_path):
            image = cv2.imread(result_path)
            if image is not None:
                print(f"   📁 Using ML orientation result 2: {result_path}")
                return image
    
    # For deskew-pdf-before, use the skew corrected result
    elif filename == "deskew-pdf-before.jpeg":
        result_path = os.path.join(base_dir, "..", "1. final_skew_detector", "output", "deskewed", "deskew-pdf-before_deskewed.png")
        if os.path.exists(result_path):
            image = cv2.imread(result_path)
            if image is not None:
                print(f"   📁 Using skew corrected result: {result_path}")
                return image
    
    print(f"   ❌ No original script result found for: {filename}")
    return None

def get_skew_angle(filename):
    """Get the detected skew angle from the original script output"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "1. final_skew_detector", "output", "processing_summary.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[0] == filename:
                    try:
                        angle = float(parts[1])
                        return angle
                    except ValueError:
                        return None
    except Exception as e:
        print(f"   ⚠️  Error reading skew angle: {e}")
    
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
    
    # Add skew angle information if this is a skew detection result
    skew_angle = get_skew_angle(filename)
    if skew_angle is not None:
        # Add angle information below the labels
        angle_text = f"Detected Skew: {skew_angle:+.2f}°"
        cv2.putText(combined_image, angle_text, (10, 70), font, 0.8, (255, 0, 0), 2)  # Blue text
        
        # Add correction information
        if abs(skew_angle) > 0.5:
            correction_text = f"Applied Correction: {-skew_angle:+.2f}°"
            cv2.putText(combined_image, correction_text, (w1 + 10, 70), font, 0.8, (0, 128, 0), 2)  # Green text
        else:
            no_correction_text = "No Correction Needed"
            cv2.putText(combined_image, no_correction_text, (w1 + 10, 70), font, 0.8, (128, 128, 128), 2)  # Gray text
    
    # Add filename at the bottom
    cv2.putText(combined_image, f"File: {filename}", (10, combined_height - 20), font, 0.7, color, 1)
    
    # Save the comparison
    cv2.imwrite(output_path, combined_image)

def main():
    """Main function"""
    
    print("🖼️  Direct Comparison Generator (Using Original Script Results)")
    print("=" * 60)
    
    # Generate comparisons
    create_direct_comparisons()
    
    print("\n📁 Files generated:")
    print("   - {no:02d}_xxx_result.png: Final processed image from original scripts")
    print("   - {no:02d}_xxx_comparison.png: Original vs. Processed comparison")
    print("\n💡 Check the output/ folder for your results!")

if __name__ == "__main__":
    main()
