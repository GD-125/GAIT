# 🚶 Gait Detection System Using CNN-BiLSTM

A comprehensive deep learning system for detecting human gait patterns from multi-sensor wearable data (accelerometer, gyroscope, and EMG signals). Built with PyTorch, featuring explainable AI with SHAP integration.

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Dataset Format](#-dataset-format)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Testing](#-testing)
- [Contributing](#-contributing)

---

## ✨ Features

- **🔄 Complete Preprocessing Pipeline**
  - Data cleaning (missing values, outliers)
  - Signal filtering (Butterworth, Savitzky-Golay, Median)
  - Sensor-specific filtering optimization
  - Z-score normalization with persistence
  - Window segmentation with configurable overlap

- **🧠 Advanced Deep Learning Architecture**
  - 1D-CNN for local temporal feature extraction
  - Bidirectional LSTM for sequential dependencies
  - Residual connections and batch normalization
  - Attention mechanism for feature importance
  - Binary classification (Gait vs Non-Gait)

- **📊 Comprehensive Evaluation**
  - Multiple metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
  - Confusion matrices and ROC/PR curves
  - Threshold optimization
  - Per-class performance analysis

- **🔍 Explainability with SHAP**
  - Feature importance ranking
  - Force plots for individual predictions
  - Summary plots for global interpretability
  - Ready for federated learning integration

- **⚙️ Production-Ready**
  - Modular and extensible design
  - Configuration management (YAML/JSON)
  - Comprehensive unit and integration tests
  - CLI interface for inference
  - Model checkpointing and early stopping

---

## 📁 Project Structure

```
gait-detection-system/
│
├── data/
│   ├── raw/                          # Original CSV datasets (200+ files)
│   ├── processed/                    # Preprocessed numpy arrays
│   └── splits/                       # Train/val/test splits
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # CSV loading & parsing
│   │   ├── cleaner.py               # Missing values & outliers
│   │   ├── filter.py                # Signal filtering
│   │   ├── normalizer.py            # Z-score normalization
│   │   ├── segmentation.py          # Window segmentation
│   │   └── pipeline.py              # Complete pipeline
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── time_domain.py           # Time-domain features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_bilstm.py            # Model architecture
│   │   └── model_utils.py           # Checkpointing utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   ├── validator.py             # Evaluation
│   │   └── metrics.py               # Performance metrics
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_explainer.py        # SHAP integration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── visualization.py         # Plotting utilities
│
├── configs/
│   └── config.yaml                  # Hyperparameters
│
├── checkpoints/                     # Saved models
│
├── results/
│   ├── logs/                        # Training logs
│   ├── plots/                       # Visualizations
│   └── shap_outputs/                # SHAP explanations
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   └── 03_model_evaluation.ipynb
│
├── main.py                          # Main training script
├── inference.py                     # Inference script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gait-detection-system.git
cd gait-detection-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import shap; print('SHAP installed successfully')"
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

Place your CSV files in the `data/raw/` directory. Expected format:

```
data/raw/
├── HuGaDB_v2_various_01_00.csv
├── HuGaDB_v2_various_01_01.csv
└── ... (200+ files)
```

### 2. Configure Settings

Edit `configs/config.yaml` or use defaults:

```yaml
preprocessing:
  sampling_rate: 100.0
  window_size: 128
  overlap: 0.5
  normalization_method: 'zscore'
  
model:
  input_features: 38
  seq_length: 128
  conv_filters: [64, 128, 256]
  lstm_hidden_size: 128
  
training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

### 3. Train Model

```bash
python main.py
```

This will:
- ✅ Preprocess all data
- ✅ Train CNN-BiLSTM model
- ✅ Evaluate on test set
- ✅ Generate SHAP explanations
- ✅ Save results and checkpoints

### 4. Run Inference

**Single file:**
```bash
python inference.py --input data/raw/test_file.csv --output predictions.npy
```

**Batch inference:**
```bash
python inference.py --input data/raw/ --output results/predictions/
```

**With probabilities:**
```bash
python inference.py --input data/raw/test_file.csv --probabilities
```

---

## 📖 Usage Guide

### Training from Scratch

```python
from preprocessing.pipeline import PreprocessingPipeline
from models import CNN_BiLSTM_GaitDetector, get_device
from training import GaitDetectorTrainer, create_data_loaders
import torch.nn as nn
import torch.optim as optim

# 1. Preprocess data
pipeline = PreprocessingPipeline(
    window_size=128,
    overlap=0.5,
    normalization_method='zscore'
)

windowed_data, labels = pipeline.preprocess_multiple_files(
    data_dir='data/raw/',
    max_files=10
)

# 2. Split data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    windowed_data, labels, test_size=0.2, random_state=42
)

# 3. Create data loaders
train_loader, val_loader = create_data_loaders(
    X_train, y_train, X_val, y_val, batch_size=32
)

# 4. Initialize model
device = get_device()
model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)

# 5. Setup training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = GaitDetectorTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

# 6. Train
trainer.train(num_epochs=50)
```

### Making Predictions

```python
from inference import GaitDetectionInference

# Initialize inference system
inference = GaitDetectionInference(
    model_path='checkpoints/best_model.pt',
    normalizer_path='data/processed/normalizer.pkl',
    config_path='checkpoints/training_config.json'
)

# Predict on new file
results = inference.predict_file(
    filepath='data/raw/new_data.csv',
    return_probabilities=False
)

print(f"Prediction: {results['overall_prediction']}")
print(f"Gait %: {results['gait_percentage']:.1f}%")
```

### SHAP Explainability

```python
from explainability import GaitSHAPExplainer
from models import CNN_BiLSTM_GaitDetector, load_checkpoint, get_device

# Load model
device = get_device()
model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
load_checkpoint('checkpoints/best_model.pt', model, device=device)

# Create explainer
explainer = GaitSHAPExplainer(model=model, device=device)

# Generate explanations
explainer.create_explainer(background_data=X_train[:100])
explainer.explain_samples(test_data=X_test[:20])

# Generate comprehensive report
explainer.generate_explanation_report(
    test_data=X_test[:20],
    output_dir='results/shap_outputs'
)

# Get feature importance
importance = explainer.get_feature_importance()
for feature, score in list(importance.items())[:10]:
    print(f"{feature}: {score:.6f}")
```

---

## 📊 Dataset Format

Your CSV files should follow this structure (HuGaDB format):

| Column | Description | Type |
|--------|-------------|------|
| `accelerometer_{location}_{axis}` | Accelerometer readings (18 columns) | Integer |
| `gyroscope_{location}_{axis}` | Gyroscope readings (18 columns) | Integer |
| `EMG_right` | Right leg EMG | Integer |
| `EMG_left` | Left leg EMG | Integer |
| `activity` | Activity label | String |

**Locations:** `right_foot`, `right_shin`, `right_thigh`, `left_foot`, `left_shin`, `left_thigh`  
**Axes:** `x`, `y`, `z`

**Total:** 38 features + 1 label column

### Activity Labels

The system automatically converts activity labels to binary:
- **Gait (1):** walk, walking, jog, jogging, run, stairs_up, stairs_down
- **Non-Gait (0):** sit, stand, lay, etc.

---

## 🏗️ Model Architecture

### CNN-BiLSTM Hybrid

```
Input (batch, 128, 38)
    ↓
Conv1D Block 1: 38 → 64 filters
    ↓ (+ Residual Connection)
Conv1D Block 2: 64 → 128 filters
    ↓ (+ Residual Connection)
Conv1D Block 3: 128 → 256 filters
    ↓
BiLSTM Layer 1: 256 → 128×2
    ↓
BiLSTM Layer 2: 256 → 128×2
    ↓
Attention Mechanism
    ↓
FC Layer 1: 256 → 256
    ↓
FC Layer 2: 256 → 128
    ↓
Output Layer: 128 → 1 (Sigmoid)
```

### Key Features
- **1D Convolutions:** Extract local temporal patterns
- **Bidirectional LSTM:** Capture forward and backward dependencies
- **Residual Connections:** Improve gradient flow
- **Batch Normalization:** Stabilize training
- **Attention:** Weight important time steps
- **Dropout (30%):** Prevent overfitting

### Model Parameters
- Total Parameters: ~2-3M (depending on configuration)
- Training Time: ~30-60 minutes (GPU) for 50 epochs
- Inference Speed: ~100-200 windows/second (GPU)

---

## 📈 Results

### Expected Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92-95% |
| Precision | 91-94% |
| Recall | 90-93% |
| F1-Score | 91-94% |
| AUC-ROC | 0.95-0.98 |

### Outputs

After training, you'll find:

1. **Checkpoints** (`checkpoints/`)
   - `best_model.pt` - Best model weights
   - `training_config.json` - Training configuration

2. **Logs** (`results/logs/`)
   - `training_history.json` - Loss/accuracy per epoch

3. **Plots** (`results/plots/`)
   - `training_history.png` - Training curves
   - `confusion_matrix.png` - Confusion matrix
   - `roc_curve.png` - ROC curve
   - `pr_curve.png` - Precision-Recall curve

4. **SHAP Outputs** (`results/shap_outputs/`)
   - `feature_importance.png` - Top features
   - `shap_summary.png` - SHAP summary plot
   - `force_plot_sample_*.png` - Individual explanations
   - `feature_importance.txt` - Feature rankings

---

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run with verbose output
python -m unittest discover tests/ -v

# Run specific test module
python -m unittest tests.test_preprocessing
python -m unittest tests.test_model
python -m unittest tests.test_pipeline
```

### Run with Coverage

```bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run -m unittest discover tests/

# Generate report
coverage report

# Generate HTML report
coverage html
# Open htmlcov/index.html in browser
```

### Test Categories

1. **Unit Tests**
   - `test_preprocessing.py` - All preprocessing components
   - `test_model.py` - Model architecture and utilities

2. **Integration Tests**
   - `test_pipeline.py` - End-to-end workflow

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Use type hints where appropriate

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- HuGaDB Dataset for gait data format
- PyTorch team for the deep learning framework
- SHAP library for explainability
- Open-source community

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: your.email@example.com

---

## 🔮 Future Enhancements

- [ ] Real-time gait detection
- [ ] Multi-class activity recognition
- [ ] Federated learning implementation
- [ ] Mobile deployment (ONNX/TensorFlow Lite)
- [ ] Web interface for visualization
- [ ] Support for additional sensor types
- [ ] Transfer learning from pre-trained models

---

**Made with ❤️ for Human Gait Analysis**
