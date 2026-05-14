"""
Analyze COCO format dataset structure
Student ID: 25509225
"""

import os
import json
from pathlib import Path
from collections import Counter

# Configuration
STUDENT_ID = "25509225"
BASE_PATH = Path(f"data/{STUDENT_ID}/Object_Detection/coco")

print("="*80)
print("COCO DATASET STRUCTURE ANALYSIS")
print("="*80)

# Check directory structure
print("\n1. Directory Structure:")
print("-" * 80)
for split in ['train', 'valid', 'test']:
    split_path = BASE_PATH / split
    if split_path.exists():
        print(f"\n{split.upper()}:")
        
        # List all files in the split directory
        all_files = list(split_path.iterdir())
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        json_files = [f for f in all_files if f.is_file() and f.suffix == '.json']
        
        print(f"  Images: {len(image_files)} files")
        if image_files:
            print(f"    Sample: {image_files[0].name}")
        
        print(f"  Annotation JSON: {len(json_files)} files")
        if json_files:
            for json_file in json_files:
                print(f"    - {json_file.name}")
    else:
        print(f"\n{split.upper()}: Directory not found")

# Analyze COCO JSON annotation format
print("\n\n2. COCO JSON Annotation Format Analysis:")
print("-" * 80)

train_json_path = BASE_PATH / 'train' / 'train_annotations.json'
if train_json_path.exists():
    print(f"\nLoading: {train_json_path.name}")
    
    with open(train_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\nCOCO JSON Structure:")
    print(f"  Top-level keys: {list(coco_data.keys())}")
    
    # Analyze images
    if 'images' in coco_data:
        print(f"\n  Images section:")
        print(f"    Total images: {len(coco_data['images'])}")
        if coco_data['images']:
            sample_img = coco_data['images'][0]
            print(f"    Sample image info:")
            for key, value in sample_img.items():
                print(f"      {key}: {value}")
    
    # Analyze annotations
    if 'annotations' in coco_data:
        print(f"\n  Annotations section:")
        print(f"    Total annotations: {len(coco_data['annotations'])}")
        if coco_data['annotations']:
            sample_ann = coco_data['annotations'][0]
            print(f"    Sample annotation info:")
            for key, value in sample_ann.items():
                print(f"      {key}: {value}")
    
    # Analyze categories
    if 'categories' in coco_data:
        print(f"\n  Categories section:")
        print(f"    Total categories: {len(coco_data['categories'])}")
        print(f"    Category details:")
        for cat in coco_data['categories']:
            print(f"      id={cat['id']}, name='{cat.get('name', 'N/A')}'")
else:
    print(f"\n⚠ Train annotation file not found: {train_json_path}")

# Analyze class distribution
print("\n\n3. Class Distribution in Train Set:")
print("-" * 80)

if train_json_path.exists():
    with open(train_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Count objects per category
    class_counter = Counter()
    total_objects = len(coco_data.get('annotations', []))
    
    for ann in coco_data.get('annotations', []):
        category_id = ann.get('category_id', -1)
        class_counter[category_id] += 1
    
    # Get category names
    category_map = {}
    for cat in coco_data.get('categories', []):
        category_map[cat['id']] = cat.get('name', f'class_{cat["id"]}')
    
    print(f"Total annotations (objects): {total_objects}")
    print(f"Number of unique categories: {len(class_counter)}")
    print(f"\nClass distribution:")
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        class_name = category_map.get(class_id, 'unknown')
        print(f"  Class {class_id} ({class_name}): {count:6d} objects ({percentage:5.1f}%)")

# Analyze image-annotation correspondence
print("\n\n4. Image-Annotation Correspondence:")
print("-" * 80)

if train_json_path.exists():
    with open(train_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get all image IDs from images section
    image_ids_in_images = set([img['id'] for img in coco_data.get('images', [])])
    
    # Get all image IDs from annotations section
    image_ids_in_annotations = set([ann['image_id'] for ann in coco_data.get('annotations', [])])
    
    # Find images without annotations
    images_without_annotations = image_ids_in_images - image_ids_in_annotations
    images_with_annotations = image_ids_in_images & image_ids_in_annotations
    
    # Count annotations per image
    annotations_per_image = Counter()
    for ann in coco_data.get('annotations', []):
        annotations_per_image[ann['image_id']] += 1
    
    # Statistics
    num_images = len(image_ids_in_images)
    num_annotated_images = len(images_with_annotations)
    num_unannotated_images = len(images_without_annotations)
    
    print(f"Total images in 'images' section: {num_images}")
    print(f"Images with annotations: {num_annotated_images}")
    print(f"Images without annotations: {num_unannotated_images}")
    print(f"Total annotations: {total_objects}")
    
    if num_annotated_images > 0:
        avg_annotations = total_objects / num_annotated_images
        print(f"Average annotations per annotated image: {avg_annotations:.1f}")
    
    # Show annotation count distribution
    if annotations_per_image:
        counts = list(annotations_per_image.values())
        print(f"\nAnnotations per image statistics:")
        print(f"  Min: {min(counts)}")
        print(f"  Max: {max(counts)}")
        print(f"  Mean: {sum(counts)/len(counts):.1f}")
        print(f"  Median: {sorted(counts)[len(counts)//2]}")
    
    if images_without_annotations:
        print(f"\nSample images without annotations (first 5 image IDs): {list(images_without_annotations)[:5]}")

# Analyze bounding box characteristics
print("\n\n5. Bounding Box Characteristics:")
print("-" * 80)

if train_json_path.exists():
    with open(train_json_path, 'r') as f:
        coco_data = json.load(f)
    
    bbox_areas = []
    bbox_widths = []
    bbox_heights = []
    
    for ann in coco_data.get('annotations', []):
        bbox = ann.get('bbox', [0, 0, 0, 0])  # [x, y, width, height]
        if len(bbox) == 4:
            x, y, w, h = bbox
            area = w * h
            bbox_areas.append(area)
            bbox_widths.append(w)
            bbox_heights.append(h)
    
    if bbox_areas:
        print(f"Total bounding boxes analyzed: {len(bbox_areas)}")
        print(f"\nBounding box area (pixels²):")
        print(f"  Min: {min(bbox_areas):.1f}")
        print(f"  Max: {max(bbox_areas):.1f}")
        print(f"  Mean: {sum(bbox_areas)/len(bbox_areas):.1f}")
        print(f"  Median: {sorted(bbox_areas)[len(bbox_areas)//2]:.1f}")
        
        print(f"\nBounding box dimensions:")
        print(f"  Width - Min: {min(bbox_widths):.1f}, Max: {max(bbox_widths):.1f}, Mean: {sum(bbox_widths)/len(bbox_widths):.1f}")
        print(f"  Height - Min: {min(bbox_heights):.1f}, Max: {max(bbox_heights):.1f}, Mean: {sum(bbox_heights)/len(bbox_heights):.1f}")
        
        # Check for small objects (area < 32² = 1024 pixels)
        small_objects = sum(1 for area in bbox_areas if area < 1024)
        medium_objects = sum(1 for area in bbox_areas if 1024 <= area < 9216)  # 32² to 96²
        large_objects = sum(1 for area in bbox_areas if area >= 9216)  # >= 96²
        
        print(f"\nObject size distribution:")
        print(f"  Small (< 32² pixels): {small_objects} ({small_objects/len(bbox_areas)*100:.1f}%)")
        print(f"  Medium (32² - 96² pixels): {medium_objects} ({medium_objects/len(bbox_areas)*100:.1f}%)")
        print(f"  Large (>= 96² pixels): {large_objects} ({large_objects/len(bbox_areas)*100:.1f}%)")

# Check image dimensions
print("\n\n6. Image Dimensions:")
print("-" * 80)

if train_json_path.exists():
    with open(train_json_path, 'r') as f:
        coco_data = json.load(f)
    
    widths = []
    heights = []
    
    for img_info in coco_data.get('images', [])[:100]:  # Sample first 100 images
        if 'width' in img_info and 'height' in img_info:
            widths.append(img_info['width'])
            heights.append(img_info['height'])
    
    if widths:
        print(f"Images analyzed: {len(widths)}")
        print(f"Width range: {min(widths)} - {max(widths)}")
        print(f"Height range: {min(heights)} - {max(heights)}")
        print(f"Most common size: {Counter(zip(widths, heights)).most_common(1)[0]}")
        
        # Check if all images have same size
        if len(set(zip(widths, heights))) == 1:
            print(f"✓ All images have the same dimensions: {widths[0]}x{heights[0]}")
        else:
            print(f"⚠ Images have varying dimensions")
    else:
        print("No image dimension information found in JSON")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
