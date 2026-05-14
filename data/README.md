# Dataset Documentation - Assignment 2

**Student ID:** 25509225  
**Last Updated:** 2026-05-12

---

## Overview

Two datasets for Assignment 2:

| Task | Dataset | Images | Classes | Size |
|------|---------|--------|---------|------|
| Classification | Birds (10 species) | 1,589 | 10 | ~37 MB |
| Detection | Solar Panel Damage | 1,667 | 5 | ~387 MB |

⚠️ **Important:** Use only your assigned dataset (based on student ID). Sharing datasets results in 0 marks.

---

## Image Classification Dataset

### Classes (10 bird species)

| Class | Images | Class | Images |
|-------|--------|-------|--------|
| Crested Kingfisher | 158 | Laughing Gull | 179 |
| Crow | 158 | Palila | 156 |
| Eastern Meadowlark | 173 | Paradise Tanager | 165 |
| Fairy Bluebird | 156 | Rainbow Lorikeet | 146 |
| Harlequin Quail | 139 | Townsend's Warbler | 159 |
| **Total** | **1,589** | | |

### Structure

```
data/25509225/Image_Classification/dataset/
├── CRESTED KINGFISHER/    # 158 images
├── CROW/                  # 158 images
└── ... (8 more classes)
```

**Note:** Dataset is NOT pre-split. Run `notebooks/data_processing/classification_dataset_split.ipynb` to create train/valid/test splits.

```
data/25509225/Image_Classification/split_dataset/
├── test/    # 10 classes images
├── train/   # 10 classes images
└── valid/   # 10 classes images
```

---

## Object Detection Dataset

### Classes (5 damage types)

1. **Cell** - Single cell damage
2. **Cell-Multi** - Multiple cell damage
3. **No-Anomaly** - No damage detected
4. **Shadowing** - Shadow interference
5. **Unclassified** - Unknown damage type

### Formats Available

Three annotation formats provided:
- **COCO** (`coco/`) - JSON annotations
- **Pascal VOC** (`pascal/`) - XML annotations
- **YOLO** (`yolo/`) - TXT annotations with data.yaml

### Structure

```
data/25509225/Object_Detection/
├── coco/
│   ├── train/     # images + annotations.json
│   ├── valid/
│   └── test/
├── pascal/
│   ├── train/     # images + annotations.xml
│   ├── valid/
│   └── test/
└── yolo/
    ├── train/     # images/ + labels/
    ├── valid/
    └── test/
```

Total: 1,667 images across train/valid/test splits

---

## Classification Dataset Splitting

```bash
# Split dataset first
notebooks/data_processing/classification_dataset_split.ipynb
```


## Data Format Details

### Classification Images
- **Format:** JPEG/PNG
- **Structure:** ImageFolder (class-based folders)
- **Preprocessing:** Resize to 224x224 for ResNet50

### Detection Annotations

**COCO Format:**
```json
{
  "images": [{"id": 1, "file_name": "img.jpg", ...}],
  "annotations": [{"image_id": 1, "bbox": [x,y,w,h], "category_id": 1, ...}],
  "categories": [{"id": 1, "name": "Cell"}, ...]
}
```

**YOLO Format:**
```
0 0.5 0.5 0.2 0.3  # class x_center y_center width height (normalized)
```