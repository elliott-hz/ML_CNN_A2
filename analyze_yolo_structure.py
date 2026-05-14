"""
Analyze YOLO format dataset structure
Student ID: 25509225
"""

import os
from pathlib import Path
from collections import Counter

# Configuration
STUDENT_ID = "25509225"
BASE_PATH = Path(f"data/{STUDENT_ID}/Object_Detection/yolo")

print("="*80)
print("YOLO DATASET STRUCTURE ANALYSIS")
print("="*80)

# Check directory structure
print("\n1. Directory Structure:")
print("-" * 80)
for split in ['train', 'valid', 'test']:
    split_path = BASE_PATH / split
    if split_path.exists():
        print(f"\n{split.upper()}:")
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        if images_dir.exists():
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            print(f"  Images: {len(image_files)} files")
            if image_files:
                print(f"    Sample: {image_files[0].name}")
        else:
            print(f"  Images: Directory not found")
            
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            print(f"  Labels: {len(label_files)} files")
            if label_files:
                print(f"    Sample: {label_files[0].name}")
        else:
            print(f"  Labels: Directory not found")
    else:
        print(f"\n{split.upper()}: Directory not found")

# Analyze label file format
print("\n\n2. Label File Format Analysis:")
print("-" * 80)

train_labels_dir = BASE_PATH / 'train' / 'labels'
if train_labels_dir.exists():
    # Find a label file with content
    sample_label_file = None
    for txt_file in train_labels_dir.glob('*.txt'):
        if txt_file.stat().st_size > 0:
            sample_label_file = txt_file
            break
    
    if sample_label_file:
        print(f"\nSample label file: {sample_label_file.name}")
        print(f"File size: {sample_label_file.stat().st_size} bytes")
        
        with open(sample_label_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Number of objects: {len(lines)}")
        print(f"\nFirst 5 lines (format: class_id center_x center_y width height):")
        for i, line in enumerate(lines[:5]):
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, cx, cy, w, h = parts
                print(f"  Line {i+1}: class={class_id}, cx={cx}, cy={cy}, w={w}, h={h}")
    else:
        print("\nNo label files with content found")

# Analyze class distribution
print("\n\n3. Class Distribution in Train Set:")
print("-" * 80)

class_counter = Counter()
total_objects = 0
files_with_objects = 0

if train_labels_dir.exists():
    for txt_file in train_labels_dir.glob('*.txt'):
        if txt_file.stat().st_size > 0:
            files_with_objects += 1
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
                        total_objects += 1

print(f"Total label files: {len(list(train_labels_dir.glob('*.txt')))}")
print(f"Files with objects: {files_with_objects}")
print(f"Total objects: {total_objects}")
print(f"\nClass distribution:")
for class_id in sorted(class_counter.keys()):
    count = class_counter[class_id]
    percentage = (count / total_objects * 100) if total_objects > 0 else 0
    print(f"  Class {class_id}: {count:6d} objects ({percentage:5.1f}%)")

# Analyze image-label correspondence
print("\n\n4. Image-Label Correspondence:")
print("-" * 80)

train_images_dir = BASE_PATH / 'train' / 'images'
if train_images_dir.exists() and train_labels_dir.exists():
    image_names = set([f.stem for f in train_images_dir.glob('*.jpg')])
    label_names = set([f.stem for f in train_labels_dir.glob('*.txt')])
    
    matched = image_names & label_names
    only_images = image_names - label_names
    only_labels = label_names - image_names
    
    print(f"Images: {len(image_names)}")
    print(f"Labels: {len(label_names)}")
    print(f"Matched (both image and label): {len(matched)}")
    print(f"Only images (no label): {len(only_images)}")
    print(f"Only labels (no image): {len(only_labels)}")
    
    if only_images:
        print(f"\nSample images without labels: {list(only_images)[:5]}")
    if only_labels:
        print(f"\nSample labels without images: {list(only_labels)[:5]}")

# Check image dimensions
print("\n\n5. Image Dimensions:")
print("-" * 80)

try:
    from PIL import Image
    widths = []
    heights = []
    
    for img_file in list(train_images_dir.glob('*.jpg'))[:10]:
        with Image.open(img_file) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    
    if widths:
        print(f"Sample images analyzed: {len(widths)}")
        print(f"Width range: {min(widths)} - {max(widths)}")
        print(f"Height range: {min(heights)} - {max(heights)}")
        print(f"Most common size: {Counter(zip(widths, heights)).most_common(1)[0]}")
except Exception as e:
    print(f"Error analyzing images: {e}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
