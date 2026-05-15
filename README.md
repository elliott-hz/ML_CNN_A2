# ML_CNN_A2 - Image Classification and Object Detection with CNNs

**Student ID:** 25509225  
**Last Updated:** 2026-05-15


⚠️ **Data Folder:** Raw datasets and split classification datasets are **not included** in this repository due to size constraints. To re-run the notebooks, please place your personal datasets in the `data/25509225/` directory, then run [`notebooks/data_processing/classification_dataset_split.ipynb`](notebooks/data_processing/classification_dataset_split.ipynb) to generate the complete dataset structure.

⚠️ **Outputs Folder:** The `outputs/` directory is **not tracked by git** due to large file sizes (training checkpoints, processing data, and per-epoch model weights). All important information including training metrics, visualizations, and results are displayed directly in the notebook outputs during execution.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Classification Experiments](#classification-experiments)
- [Detection Experiments](#detection-experiments)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## Introduction

This project explores Convolutional Neural Networks (CNNs) for two fundamental computer vision tasks: **image classification** and **object detection**. The work is divided into two main experimental pipelines, each investigating different model architectures and design choices to understand their impact on performance.

### Key Objectives

1. **Image Classification**: Classify bird species using ResNet50-based architectures with various modifications
2. **Object Detection**: Detect solar panel damage types using Faster R-CNN and YOLOv8 models with customized configurations

The project systematically compares baseline models against customized variants (deeper/shallower architectures, modified hyperparameters) to evaluate architectural design decisions without focusing on final performance metrics.

---

## Dataset

### Image Classification Dataset

**Task:** Bird Species Classification (10 classes)

| Metric | Value |
|--------|-------|
| Total Images | 1,589 |
| Number of Classes | 10 |
| Dataset Size | ~37 MB |

#### Classes

- Crested Kingfisher (158 images)
- Crow (158 images)
- Eastern Meadowlark (173 images)
- Fairy Bluebird (156 images)
- Harlequin Quail (139 images)
- Laughing Gull (179 images)
- Palila (156 images)
- Paradise Tanager (165 images)
- Rainbow Lorikeet (146 images)
- Townsend's Warbler (159 images)

#### Data Split Strategy

The dataset is split using student ID (25509225) as a random seed for reproducibility:

- **Training:** 70% (1,109 images)
- **Validation:** 15% (231 images)
- **Testing:** 15% (249 images)

Splitting is performed by [`notebooks/data_processing/classification_dataset_split.ipynb`](notebooks/data_processing/classification_dataset_split.ipynb), which creates an ImageFolder-compatible directory structure.

### Object Detection Dataset

**Task:** Solar Panel Damage Detection (5 classes)

| Metric | Value |
|--------|-------|
| Total Images | 1,667 |
| Number of Classes | 5 |
| Dataset Size | ~387 MB |

#### Classes

1. **Cell** - Single cell damage
2. **Cell-Multi** - Multiple cell damage
3. **No-Anomaly** - No damage detected
4. **Shadowing** - Shadow interference
5. **Unclassified** - Unknown damage type

#### Annotation Formats

Three annotation formats are provided for flexibility:

- **COCO Format** (`coco/`) - JSON annotations for standard detection frameworks
- **Pascal VOC Format** (`pascal/`) - XML annotations for legacy compatibility
- **YOLO Format** (`yolo/`) - TXT annotations with data.yaml configuration

Each format includes pre-split train/valid/test sets.

---

## Classification Experiments

All classification experiments use ResNet50 as the base architecture, trained on the bird species dataset. The experiments investigate how architectural modifications affect model capacity and feature extraction capabilities.

### Shared Configuration

Across all classification experiments:

- **Input Size:** 224×224 pixels (standard ResNet50 input)
- **Batch Size:** 16
- **Epochs:** 200
- **Optimizer:** AdamW with learning rate 5e-4
- **Weight Decay:** 5e-4 to 1e-3 (varies by experiment)
- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=7, factor=0.5)
- **Warmup:** 10 epochs linear warmup
- **Early Stopping:** Patience of 50 epochs
- **Label Smoothing:** 0.1
- **Data Augmentation:** None (basic preprocessing only)
- **Pretrained Weights:** Not used (training from scratch)

### Experiment 1: Baseline ResNet50

**Notebook:** [`notebooks/classification_ResNet50/classification_ResNet50_baseline.ipynb`](notebooks/classification_ResNet50/classification_ResNet50_baseline.ipynb)

**Architecture:** Standard ResNet50 with single fully connected layer (2048→10)

**Design Rationale:**
- Establishes a reference point for comparison with modified architectures
- Uses vanilla ResNet50 backbone without any structural modifications
- Single FC layer directly maps 2048-dimensional features to 10 classes
- Dropout rate: 0.5 with batch normalization enabled

**Key Characteristics:**
- Total parameters: ~23.5M (standard ResNet50 classifier head)
- Backbone remains untouched (layer1-layer4 intact)
- Simplest architecture serving as the control experiment

### Experiment 2: Deeper V1 - Added Convolutional Block

**Notebook:** [`notebooks/classification_ResNet50/classification_ResNet50_deeper_v1.ipynb`](notebooks/classification_ResNet50/classification_ResNet50_deeper_v1.ipynb)

**Architecture:** Modified ResNet50 with additional convolutional block after layer1

**Design Rationale:**
- Tests whether increasing network depth improves feature extraction
- Adds a custom conv block immediately after layer1 to enhance early-stage feature processing
- Hypothesis: Additional convolutional layers may capture more nuanced patterns in shallow features before they propagate through deeper layers

**Architectural Modifications:**
- `modify_backbone=True` with `add_conv_after_layer='layer1'`
- Inserts extra convolutional block between layer1 and layer2
- Maintains residual connections and skip pathways
- Weight decay increased to 1e-3 to regularize the deeper architecture

**Key Characteristics:**
- Total parameters: ~24.7M (increase from baseline due to added conv block)
- Deeper feature hierarchy with enhanced shallow feature processing
- Same classifier head as baseline (2048→10)

### Experiment 3: Reduced V1 - Removed Layer3

**Notebook:** [`notebooks/classification_ResNet50/classification_ResNet50_reduced_v1.ipynb`](notebooks/classification_ResNet50/classification_ResNet50_reduced_v1.ipynb)

**Architecture:** Modified ResNet50 with layer3 removed from backbone

**Design Rationale:**
- Investigates whether reducing model complexity maintains competitive performance
- Removes entire layer3 to create a shallower network with fewer parameters
- Hypothesis: For 10-class classification, the full ResNet50 depth may be excessive; removing intermediate layers could reduce overfitting while preserving essential feature representations

**Architectural Modifications:**
- `modify_backbone=True` with `remove_layer='layer3'`
- Completely removes layer3 from the ResNet50 backbone
- Direct connection from layer2 to layer4
- Reduces computational cost and memory footprint

**Key Characteristics:**
- Total parameters: ~15.1M (significant reduction from baseline's ~23.5M)
- Approximately 36% fewer parameters than baseline
- Shallower feature hierarchy with faster forward pass
- Tests trade-off between model capacity and efficiency

### Comparison Summary

| Experiment | Architecture | Parameters | Modification Type | Design Goal |
|------------|-------------|------------|-------------------|-------------|
| Baseline | Standard ResNet50 | ~23.5M | None | Reference model |
| Deeper V1 | ResNet50 + conv block | ~24.7M | Added depth | Enhanced feature extraction |
| Reduced V1 | ResNet50 - layer3 | ~15.1M | Reduced depth | Efficiency vs. capacity trade-off |

All three experiments share identical training protocols to isolate the impact of architectural changes.

---

## Detection Experiments

Object detection experiments utilize two distinct frameworks: **Faster R-CNN** (two-stage detector) and **YOLOv8** (single-stage detector). Each framework explores different customization strategies tailored to solar panel damage detection characteristics.

### Shared Dataset Configuration

- **Annotation Format:** COCO (for Faster R-CNN), YOLO (for YOLOv8)
- **Classes:** 5 damage types (+ background class for Faster R-CNN = 6 total)
- **Input Size:** 640×640 pixels
- **Pretrained Weights:** Enabled for all experiments

---

### Faster R-CNN Experiments

Faster R-CNN uses a two-stage detection pipeline: Region Proposal Network (RPN) generates candidate boxes, followed by ROI pooling and classification. All experiments use ResNet50+FPN backbone.

#### Shared Training Configuration (All Faster R-CNN Variants)

- **Optimizer:** SGD with momentum 0.9
- **Learning Rate:** 0.005
- **Batch Size:** 4
- **Epochs:** 300
- **Weight Decay:** 5e-4
- **Early Stopping:** Patience of 30 epochs
- **Image Size:** min_size=640, max_size=640

#### Experiment V1: Baseline Faster R-CNN

**Notebook:** [`notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v1.ipynb`](notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v1.ipynb)

**Configuration:** Default torchvision Faster R-CNN with standard hyperparameters

**Design Rationale:**
- Pure default configuration serves as the true baseline
- No architectural or hyperparameter modifications
- Standard anchor sizes and aspect ratios for general object detection

**Key Hyperparameters:**
- Anchor sizes: ((32,), (64,), (128,), (256,), (512,))
- Anchor aspect ratios: ((0.5, 1.0, 2.0),) × 5 feature levels
- RPN proposals (train): pre-NMS=2000, post-NMS=1000
- RPN proposals (test): pre-NMS=1000, post-NMS=1000
- NMS thresholds: RPN=0.7, box=0.5
- IoU thresholds: foreground=0.7, background=0.3

#### Experiment V2: Small Anchor Sizes

**Notebook:** [`notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v2.ipynb`](notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v2.ipynb)

**Configuration:** Modified anchor sizes optimized for tiny objects

**Design Rationale:**
- Solar panel damage (especially single cells) can be very small relative to image size
- Standard anchors (starting at 32px) may be too large for tiny defects
- Smaller anchors improve recall for small objects by providing better initial proposals

**Key Modifications:**
- Anchor sizes: ((4,), (8,), (16,), (32,), (64,)) - significantly smaller scale
- All other hyperparameters identical to V1
- Focus: Improve detection of fine-grained damage patterns

**Architectural Impact:**
- RPN generates more proposals for small regions
- Better coverage of tiny damage areas in early feature maps
- May increase false positives but improves small object recall

#### Experiment V3: Large Proposal Capacity

**Notebook:** [`notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v3.ipynb`](notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v3.ipynb)

**Configuration:** Increased RPN proposal counts for crowded scenes

**Design Rationale:**
- Solar panels may contain multiple damage instances in close proximity
- Standard proposal limits (1000 post-NMS) might discard valid detections in dense scenarios
- Higher proposal capacity preserves more candidate boxes for final classification

**Key Modifications:**
- RPN proposals (train): pre-NMS=**4000**, post-NMS=**2000** (doubled from V1)
- RPN proposals (test): pre-NMS=**2000**, post-NMS=1000
- All other hyperparameters identical to V1
- Focus: Handle overlapping/crowded damage instances

**Architectural Impact:**
- RPN retains more diverse proposals through NMS filtering
- Second stage classifier processes more ROIs
- Increased computational cost but better coverage of dense damage patterns

#### Experiment V4: Relaxed NMS Filtering

**Notebook:** [`notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v4.ipynb`](notebooks/detection_FasterRCNN/detection_FasterRCNN_baseline_v4.ipynb)

**Configuration:** Higher NMS thresholds to prevent suppression of nearby objects

**Design Rationale:**
- Standard NMS thresholds (0.7 for RPN, 0.5 for boxes) may aggressively suppress valid detections
- Adjacent damage cells or multi-cell damage patterns could be incorrectly filtered
- Relaxed NMS allows nearby boxes to coexist, improving detection of clustered damage

**Key Modifications:**
- RPN NMS threshold: **0.85** (increased from 0.7)
- Box NMS threshold: **0.6** (increased from 0.5)
- All other hyperparameters identical to V1
- Focus: Preserve spatially proximate detections

**Architectural Impact:**
- Less aggressive non-maximum suppression
- More overlapping boxes survive to final output
- Better suited for scenarios with tightly packed damage instances

#### Faster R-CNN Comparison Summary

| Experiment | Anchor Sizes | RPN Proposals (Train) | NMS Thresholds | Design Focus |
|------------|-------------|----------------------|----------------|--------------|
| V1 (Baseline) | (32,64,128,256,512) | 2000→1000 | RPN=0.7, Box=0.5 | Default configuration |
| V2 (Small Anchors) | **(4,8,16,32,64)** | 2000→1000 | RPN=0.7, Box=0.5 | Tiny object detection |
| V3 (Large Capacity) | (32,64,128,256,512) | **4000→2000** | RPN=0.7, Box=0.5 | Crowded scenes |
| V4 (Relaxed NMS) | (32,64,128,256,512) | 2000→1000 | **RPN=0.85, Box=0.6** | Nearby object preservation |

All Faster R-CNN experiments maintain identical backbone architecture (ResNet50+FPN) and training protocol, isolating the impact of RPN and NMS hyperparameter choices.

---

### YOLOv8 Experiments

YOLOv8 is a single-stage detector that predicts bounding boxes and class probabilities directly from feature maps in one forward pass. All experiments use YOLOv8m (medium-sized variant) as the base model.

#### Shared Training Configuration (All YOLOv8 Variants)

- **Optimizer:** Adam
- **Epochs:** 300
- **Mixed Precision:** Enabled (AMP)
- **Early Stopping:** Patience of 50 epochs
- **Cosine LR Schedule:** Enabled
- **Mosaic Augmentation:** Closed in last 10 epochs
- **Input Size:** 640×640 pixels
- **Confidence Threshold:** 0.5
- **NMS IoU Threshold:** 0.45

#### Experiment V1: Baseline YOLOv8m

**Notebook:** [`notebooks/detection_YOLOv8/detection_YOLOv8_baseline.ipynb`](notebooks/detection_YOLOv8/detection_YOLOv8_baseline.ipynb)

**Configuration:** Standard YOLOv8m with default architecture

**Design Rationale:**
- Establishes baseline performance for YOLOv8 framework
- Uses unmodified YOLOv8m backbone and neck architecture
- Provides comparison point for customized variants

**Key Hyperparameters:**
- Learning rate: 0.001
- Batch size: 12 (reduced from 16 to fit Tesla T4 GPU memory)
- Weight decay: 1e-4
- Pretrained weights: Enabled (COCO-pretrained)
- Customization: None

**Architecture:**
- Standard YOLOv8m CSPDarknet backbone
- PANet neck for multi-scale feature fusion
- Three detection heads (P3, P4, P5)
- C2f modules with default repeat counts

#### Experiment V2: Deeper Backbone

**Notebook:** [`notebooks/detection_YOLOv8/detection_YOLOv8_deeper.ipynb`](notebooks/detection_YOLOv8/detection_YOLOv8_deeper.ipynb)

**Configuration:** YOLOv8m with 2 additional convolutional layers in backbone

**Design Rationale:**
- Enhances shallow-layer feature extraction capability
- Additional conv layers after backbone layer 2 provide richer low-level feature representations
- Hypothesis: Deeper backbone improves detection of subtle damage patterns

**Key Modifications:**
- `customize_type='deeper'` - adds 2 conv layers after backbone index 2
- Learning rate reduced to **0.0005** (lower LR for deeper model stability)
- Weight decay increased to **5e-4** (stronger regularization for deeper architecture)
- Batch size: 12 (same as baseline due to increased model size)

**Architectural Impact:**
- Enhanced feature hierarchy with additional processing stages
- Improved gradient flow through deeper pathways
- Slightly increased parameter count and computational cost
- Better representation learning for complex damage patterns

#### Experiment V3: Shallower Backbone

**Notebook:** [`notebooks/detection_YOLOv8/detection_YOLOv8_shallower.ipynb`](notebooks/detection_YOLOv8/detection_YOLOv8_shallower.ipynb)

**Configuration:** YOLOv8m with reduced C2f module repeats

**Design Rationale:**
- Creates lighter, faster model with reduced computational requirements
- Halves C2f module repeat count at backbone index 6
- Hypothesis: For solar panel damage detection, full YOLOv8m capacity may be unnecessary; shallower model offers efficiency gains with minimal performance loss

**Key Modifications:**
- `customize_type='shallower'` - reduces C2f repeats in backbone
- Learning rate: **0.001** (standard LR sufficient for lighter model)
- Weight decay: **1e-4** (standard regularization)
- Batch size: **20** (increased from 12 due to smaller model footprint)

**Architectural Impact:**
- Reduced parameter count and FLOPs
- Faster inference speed and lower memory usage
- Larger batch size enables better gradient estimation
- Tests efficiency vs. accuracy trade-off

#### YOLOv8 Comparison Summary

| Experiment | Backbone | Learning Rate | Batch Size | Weight Decay | Design Focus |
|------------|----------|---------------|------------|--------------|--------------|
| V1 (Baseline) | YOLOv8m (standard) | 0.001 | 12 | 1e-4 | Reference model |
| V2 (Deeper) | YOLOv8m + 2 conv layers | **0.0005** | 12 | **5e-4** | Enhanced feature extraction |
| V3 (Shallower) | YOLOv8m (reduced C2f) | 0.001 | **20** | 1e-4 | Efficiency optimization |

All YOLOv8 experiments maintain identical training duration (300 epochs) and evaluation protocols, enabling direct comparison of architectural modifications.

---

## Project Structure

```
ML_CNN_A2/
├── data/
│   └── 25509225/
│       ├── Image_Classification/
│       │   ├── dataset/              # Original dataset (not in repo)
│       │   └── split_dataset/        # Generated split (not in repo)
│       └── Object_Detection/
│           ├── coco/                 # COCO format annotations
│           ├── pascal/               # Pascal VOC format annotations
│           └── yolo/                 # YOLO format annotations
├── notebooks/
│   ├── data_processing/
│   │   ├── classification_dataset_split.ipynb    # Dataset splitting utility
│   │   └── dataset_analysis.ipynb                # Exploratory data analysis
│   ├── classification_ResNet50/
│   │   ├── ResNet50_modules.ipynb                # Shared model/training modules
│   │   ├── classification_ResNet50_baseline.ipynb
│   │   ├── classification_ResNet50_deeper_v1.ipynb
│   │   └── classification_ResNet50_reduced_v1.ipynb
│   ├── detection_FasterRCNN/
│   │   ├── FasterRCNN_modules.ipynb              # Shared model/training modules
│   │   ├── FasterRCNN_DataLoader.ipynb           # Data loading utilities
│   │   ├── detection_FasterRCNN_baseline_v1.ipynb
│   │   ├── detection_FasterRCNN_baseline_v2.ipynb
│   │   ├── detection_FasterRCNN_baseline_v3.ipynb
│   │   └── detection_FasterRCNN_baseline_v4.ipynb
│   └── detection_YOLOv8/
│       ├── YOLOv8_modules.ipynb                  # Shared model/training modules
│       ├── detection_YOLOv8_baseline.ipynb
│       ├── detection_YOLOv8_deeper.ipynb
│       └── detection_YOLOv8_shallower.ipynb
├── outputs/                          # Training outputs and reports
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.8.0+
- Torchvision 0.24.0+
- CUDA-capable GPU (recommended: Tesla T4 or higher)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ML_CNN_A2

# Install dependencies
pip install torch torchvision opencv-python numpy matplotlib Pillow
pip install ultralytics  # For YOLOv8 experiments
pip install torchinfo    # For model summary visualization
```

### Dataset Setup

1. **Place your datasets** in the appropriate directories:
   ```
   data/25509225/Image_Classification/dataset/     # Bird classification images
   data/25509225/Object_Detection/coco/            # COCO format detection data
   data/25509225/Object_Detection/yolo/            # YOLO format detection data
   ```

2. **Run dataset splitting** for classification task:
   ```bash
   jupyter notebook notebooks/data_processing/classification_dataset_split.ipynb
   ```
   This generates train/valid/test splits in `data/25509225/Image_Classification/split_dataset/`.

### Running Experiments

#### Classification Experiments

```bash
# Navigate to classification directory
cd notebooks/classification_ResNet50

# Run desired experiment notebook
jupyter notebook classification_ResNet50_baseline.ipynb
# OR
jupyter notebook classification_ResNet50_deeper_v1.ipynb
# OR
jupyter notebook classification_ResNet50_reduced_v1.ipynb
```

#### Detection Experiments

```bash
# Faster R-CNN experiments
cd notebooks/detection_FasterRCNN
jupyter notebook detection_FasterRCNN_baseline_v1.ipynb

# YOLOv8 experiments
cd notebooks/detection_YOLOv8
jupyter notebook detection_YOLOv8_baseline.ipynb
```

### Output Structure

Each experiment generates outputs in dedicated directories under `outputs/`:

```
outputs/
├── classification_baseline/
├── classification_deeper_v1/
├── classification_reduced_v1/
├── detection_fasterrcnn_baseline_v1/
├── detection_yolov8_baseline/
└── ...
```

Each output directory contains:
- Trained model checkpoints
- Training/validation loss curves
- Confusion matrices
- Classification/detection reports
- Sample predictions with visualizations

---

## Key Design Principles

### Reproducibility

- All experiments use student ID (25509225) as random seed for dataset splitting
- Fixed hyperparameters documented in each notebook
- Modular code architecture via shared `.ipynb` module files

### Experimental Rigor

- **Controlled comparisons:** Only one variable changed per experiment
- **Consistent training protocols:** Identical optimizers, schedulers, and augmentation across variants
- **Comprehensive baselines:** Each framework includes unmodified reference model

### Architectural Exploration

- **Depth variations:** Testing both deeper and shallower networks
- **Hyperparameter tuning:** Systematic exploration of RPN/NMS settings for detection
- **Efficiency considerations:** Balancing model capacity with computational constraints

---

## Notes

- All experiments were conducted on **Tesla T4 GPU** (14.6 GB VRAM)
- Batch sizes adjusted to fit GPU memory constraints
- Mixed precision training (AMP) enabled for YOLOv8 experiments
- Early stopping prevents overfitting and reduces training time
- No performance metrics or comparative analysis included in this documentation (per reporting standards)

---

## License

This project is for educational purposes as part of Assignment 2 coursework.

---

**Contact:** Student ID 25509225
