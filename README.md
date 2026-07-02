# Paper “A Model for Main Fruit and Vegetable Leaf Disease Identification Based on Lightweight Composite Feature Interaction Network”

> Official implementation of the manuscript **"A Model for Main Fruit and Vegetable Leaf Disease Identification Based on Lightweight Composite Feature Interaction Network"** (Manuscript ID: 1920089)

##  Project Overview

This repository provides the complete reproducible implementation of the proposed lightweight composite feature interaction network for leaf disease recognition of main fruit and vegetable crops. The model integrates **EfficientNet-B3** convolutional backbone, **Squeeze-and-Excitation (SE)** channel attention and **MLP-Mixer** feature interaction modules, supporting **12-class disease identification** on tomato, grape and apple leaves, as well as cross-crop generalization verification on corn and potato datasets.

All codes, raw numerical tables, original figures and full datasets are included in this repository, and all experimental results in the manuscript can be fully reproduced with fixed random seeds.

---

##  Dataset Description

### Data Source

All datasets are sorted and filtered from the public open-source **New Plant Disease (NPD) dataset**, available at: [https://www.kaggle.com/datasets/amansingh2130/plant-disease-dataset/suggestions](https://www.kaggle.com/datasets/amansingh2130/plant-disease-dataset/suggestions). The core training set covers 3 main fruit and vegetable crops (tomato, grape, apple) with 12 categories, and independent corn and potato datasets are reserved for cross-crop generalization testing.

### Multi-part Dataset Archives

Full datasets are stored as split RAR files in this repository:

| Dataset | Files | Parts | Size |
|---------|-------|-------|------|
| Main Fruit & Vegetable Leaf Dataset (12 classes) | `Dataset of Main Fruit and Vegetable Leaf.part001.rar` – `.part088.rar` | 88 parts | — |
| Corn Leaf Dataset | `Dataset of Corn.part01.rar` – `.part25.rar` | 25 parts | ~130 GB |
| Potato Leaf Dataset | `Dataset of Potato.part01.rar` – `.part45.rar` | 45 parts | — |

### Extraction Guide

#### Linux / macOS

```bash
# Take corn dataset as example; same operation for other datasets
cat "Dataset of Corn.part"*.rar > corn_merged.rar
unrar x corn_merged.rar ./dataset/
```

#### Windows

Open WinRAR, select the first part file (e.g. `Dataset of Corn.part01.rar`) and click **Extract Here**; the software will automatically merge all parts and complete decompression.

### Standard Directory Structure

```
dataset/
├── train/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_Leaf_Mold/
│   ├── Tomato_healthy/
│   ├── Grape_Black_rot/
│   ├── Grape_Esca_Black_Measles/
│   ├── Grape_Leaf_blight/
│   ├── Grape_healthy/
│   ├── Apple_Apple_scab/
│   ├── Apple_Black_rot/
│   ├── Apple_Cedar_apple_rust/
│   └── Apple_healthy/
└── test/
    └── (same 12-class structure as training set)
```

---

##  Environment Setup

### System Requirements

- **OS:** Linux / macOS / Windows
- **Python:** >= 3.8
- **Hardware:** NVIDIA GPU with CUDA support (CPU mode is available but slower)
- **GPU Memory:** >= 8GB recommended

### Dependency Installation

```bash
# Clone repository
git clone https://github.com/czpbest/DataSet_For_Model-for-MFVLDI-Based-on-LCFIN.git
cd DataSet_For_Model-for-MFVLDI-Based-on-LCFIN

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install core deep learning libraries
pip install torch torchvision
# For users in China: use Tsinghua mirror for faster download
# pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install scientific computing & visualization libraries
pip install scikit-learn seaborn pandas pillow tqdm matplotlib
```

---

##  Quick Start

### Step 1: Data Preparation

Decompress all datasets and place them into the `dataset/` directory following the standard structure above. You can modify the data path in the `CONFIG` dictionary inside each script.

### Step 2: Model Training

Default hyperparameters are strictly consistent with the manuscript:

| Hyperparameter | Value |
|----------------|-------|
| Input resolution | 300×300 |
| Batch size | 8 |
| Initial learning rate | 3e-4 |
| Optimizer | AdamW (weight decay = 1e-4) |
| Learning rate scheduler | ReduceLROnPlateau |
| Max training epochs | 20 |
| Fixed random seed | 42 (guarantees 100% reproducibility) |

Run training command:

```bash
python 01_model_train.py
```

Outputs are saved in `train_output/`:

- `best_model.pth`: model weights with best validation performance
- `training_curves.png`: loss and accuracy curve visualization

### Step 3: Evaluation & Visualization

```bash
# Generate confusion matrix and classification metrics
python 03_confusion_matrix.py

# Generate convolutional feature visualization maps
python 02_heatmap_gradcam.py

# Generate stability violin plot (50 repeated evaluations)
python 06_violin_plot.py

# Comparison experiments with classical baseline models
python 07_compare_classic_models.py

# Cross-crop generalization test (corn / potato datasets)
python 05_generalization_test.py

# Module ablation experiments
python 04_ablation_experiment.py
```

---

##  Repository File Structure

### Code Scripts

| File Name | Description |
|-----------|-------------|
| `01_model_train.py` | Core training script: model construction, training, validation and testing |
| `02_heatmap_gradcam.py` | Grad-CAM based convolutional feature visualization |
| `03_confusion_matrix.py` | Confusion matrix plotting and quantitative metrics calculation |
| `04_ablation_experiment.py` | Ablation tests for each component of the proposed model |
| `05_generalization_test.py` | Cross-crop generalization performance verification |
| `06_violin_plot.py` | Stability analysis via violin plots of 50 repeated runs |
| `07_compare_classic_models.py` | Comparative experiments with mainstream benchmark models |

### Raw Numerical Tables (Excel)

- `Dataset_Distribution_Stats.xlsx`: complete sample distribution statistics (corresponds to Table 4.2 in manuscript)
- Raw data tables for all figures (training curves, confusion matrix, ablation results, generalization tests, violin plots, baseline comparison) are provided individually, named corresponding to figure numbers in the manuscript.

### Original Figures

All figures from the manuscript are exported directly from code as high-resolution TIFF/JPG original files, with no PPT screenshots. Separate description documents are attached for each figure.

---

##  Model Architecture

```
Input Image (300×300×3)
        ↓
EfficientNet-B3 Backbone (convolutional feature extraction)
        ↓
SE Channel Attention Block (adaptive channel weighting)
        ↓
2D Projection Convolution (512 channels)
        ↓
4 × MLP-Mixer Blocks (spatial & channel feature interaction)
        ↓
Global Average Pooling
        ↓
Classification Head (12 classes)
        ↓
Output Probability
```

### Key Technical Designs

- **Dual weighted sampling + weighted cross-entropy loss**: mitigates class imbalance and accelerates convergence
- **SE channel attention**: adaptively learns importance weights of different feature channels
- **MLP-Mixer feature interaction**: captures complex spatial and channel-wise feature dependencies
- **ReduceLROnPlateau scheduler**: adaptively adjusts learning rate for better convergence
- **ImageNet pre-trained initialization**: improves feature extraction capability and training efficiency

---

##  Reproducibility Declaration

- All scripts use a fixed random seed (`seed = 42`) for data splitting, model initialization and training process
- All hyperparameters are strictly consistent with the description and tables in the manuscript
- All quantitative results and figures in the paper can be fully reproduced by executing the scripts in sequence

---

##  Manuscript Information

| Item | Detail |
|------|--------|
| **Title** | A Model for Main Fruit and Vegetable Leaf Disease Identification Based on Lightweight Composite Feature Interaction Network |
| **Manuscript ID** | 1920089 |
| **Journal** | Frontiers in Plant Science |

---

##  License

This repository is for **non-commercial academic research only**. All datasets are derived from public open-access academic resources.

---

##  Acknowledgments

- **EfficientNet**: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- **SE Block**: Squeeze-and-Excitation Networks
- **MLP-Mixer**: MLP-Mixer: An all-MLP Architecture for Vision
- **Original NPD dataset**: public open-access plant disease image benchmark, available at: [https://www.kaggle.com/datasets/amansingh2130/plant-disease-dataset/suggestions](https://www.kaggle.com/datasets/amansingh2130/plant-disease-dataset/suggestions)
