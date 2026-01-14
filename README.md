# Graph-Based Interaction Modeling for Context-Aware Pedestrian Intention Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/)

Official implementation of **"Graph-Based Interaction Modeling for Context-Aware Pedestrian Intention Prediction in Autonomous Driving"**

**Authors:** Basharat Hussain (NUCES, Islamabad), Muhammad Islam (James Cook University)

---

## 📋 Overview

This repository contains the implementation of two novel deep learning architectures for predicting pedestrian crossing intentions at intersections:

- **STA-GCN (Spatial-Temporal Attention GCN)**: Lightweight attention-augmented graph convolutional network achieving 88.72% accuracy with 14.2ms inference time
- **VR-GCN (Visual-Relational GCN)**: Multi-relational graph network that explicitly models pedestrian-environment interactions

### Key Features

✅ **High Accuracy**: 88.72% - 89.21% accuracy on JAAD dataset  
✅ **Real-Time Performance**: 14.2ms inference (2.5× faster than Transformers)  
✅ **Interpretable Predictions**: Dual attention visualization for transparency  
✅ **Efficient Architecture**: 13% fewer parameters than Transformer baselines  
✅ **Context-Aware**: Explicit modeling of environmental relationships  

---

## 🎯 Performance Highlights

| Model | Accuracy | F1-Score | Inference Time | Parameters |
|-------|----------|----------|----------------|------------|
| Transformer | 85.23% | 0.7763 | 35.2ms | ~1.5M |
| GCN | 87.20% | 0.7766 | 8.3ms | ~0.8M |
| ST-GCN | 88.21% | 0.7713 | 12.5ms | ~1.2M |
| **STA-GCN (Ours)** | **88.72%** | **0.7764** | **14.2ms** | **~1.3M** |
| **VR-GCN (Ours)** | **89.21%** | **0.7498** | **18.7ms** | **~1.8M** |

---

## 🏗️ Architecture

### STA-GCN Architecture
```
Input (Pose + Trajectory)
    ↓
Spatial-Temporal GCN Blocks
    ├─ Spatial Graph Convolution
    ├─ Temporal Convolution
    ├─ Spatial Attention (body joints)
    └─ Temporal Attention (time steps)
    ↓
Context Encoding (trajectory features)
    ↓
Feature Fusion
    ↓
Binary Classification (Crossing/Not-Crossing)
```

### VR-GCN Architecture
```
Input (Pose + Trajectory + Scene)
    ↓
Relational Graph Construction
    ├─ Pose Relation Node
    ├─ Spatial Relation Node (curb distance, direction)
    └─ Temporal Relation Node (acceleration, velocity)
    ↓
Multi-Scale VR-GCN Layers (3 layers)
    ├─ Relation-Specific Transformations
    ├─ Attention-Based Aggregation
    └─ Hierarchical Feature Fusion
    ↓
Classification
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB RAM minimum

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/pedestrian-crossing-prediction.git
cd pedestrian-crossing-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
opencv-python>=4.5.0
tqdm>=4.62.0
tensorboard>=2.8.0
```

---

## 📊 Dataset

### JAAD Dataset (Joint Attention in Autonomous Driving)

Download the JAAD dataset from the [official repository](https://github.com/ykotseruba/JAAD):
```bash
# Download and extract JAAD dataset
wget http://data.nvision2.eecs.yorku.ca/JAAD_clips.zip
unzip JAAD_clips.zip -d ./data/JAAD/
```

### Dataset Structure
```
data/
├── JAAD/
│   ├── clips/              # Video clips
│   ├── annotations/        # XML annotations
│   └── splits/            # Train/test splits
└── processed/
    ├── poses/             # Extracted skeleton poses
    ├── trajectories/      # Computed trajectories
    └── metadata.pkl       # Preprocessed metadata
```

### Preprocessing
```bash
# Extract poses and trajectories from JAAD
python scripts/preprocess_jaad.py --data_dir ./data/JAAD --output_dir ./data/processed

# Expected output:
# - 2,515 samples total
# - Training: 2,012 samples (80%)
# - Testing: 503 samples (20%)
# - Observation window: 4 seconds (120 frames @ 30fps)
```

---

## 🎓 Training

### Quick Start
```bash
# Train STA-GCN
python train.py --model sta_gcn --epochs 20 --batch_size 16 --lr 0.001

# Train VR-GCN
python train.py --model vr_gcn --epochs 20 --batch_size 16 --lr 0.001

# Train with custom configuration
python train.py --config configs/sta_gcn_config.yaml
```

### Configuration Options
```yaml
# configs/sta_gcn_config.yaml
model:
  name: sta_gcn
  num_joints: 15
  num_channels: 4
  hidden_dim: 128
  num_layers: 4
  dropout: 0.1

training:
  epochs: 20
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 1e-4
  optimizer: adam
  scheduler: reduce_on_plateau

data:
  observation_time: 4  # seconds
  fps: 30
  augmentation:
    temporal_crop: true
    spatial_noise: true
    horizontal_flip: true
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `sta_gcn` | Model architecture (sta_gcn, vr_gcn, st_gcn, transformer) |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `16` | Training batch size |
| `--lr` | `0.001` | Initial learning rate |
| `--weight_decay` | `1e-4` | L2 regularization coefficient |
| `--gpu` | `0` | GPU device ID |
| `--resume` | `None` | Path to checkpoint to resume training |

---

## 🧪 Evaluation

### Test Pre-trained Models
```bash
# Evaluate STA-GCN
python evaluate.py --model sta_gcn --checkpoint checkpoints/sta_gcn_best.pth

# Evaluate VR-GCN
python evaluate.py --model vr_gcn --checkpoint checkpoints/vr_gcn_best.pth

# Generate attention visualizations
python evaluate.py --model sta_gcn --checkpoint checkpoints/sta_gcn_best.pth --visualize
```

### Evaluation Metrics

The evaluation script reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of correct crossing predictions
- **Recall**: Proportion of actual crossings detected
- **F1-Score**: Harmonic mean of precision and recall
- **Inference Time**: Average prediction latency (ms)

### Example Output
```
========================================
Model: STA-GCN
========================================
Accuracy:     88.72%
Precision:    84.24%
Recall:       71.68%
F1-Score:     77.64%
Inference:    14.2ms ± 1.3ms
Parameters:   1.3M
FLOPs:        2.1G
========================================
```

---

## 🔍 Inference

### Single Sample Prediction
```python
import torch
from models import STAGCN
from utils import load_sample

# Load model
model = STAGCN(num_joints=15, num_channels=4)
model.load_state_dict(torch.load('checkpoints/sta_gcn_best.pth'))
model.eval()

# Load sample
pose, trajectory = load_sample('data/test/sample_001.pkl')

# Predict
with torch.no_grad():
    output = model(pose, trajectory)
    prob = torch.sigmoid(output).item()
    prediction = 'Crossing' if prob > 0.5 else 'Not Crossing'
    
print(f"Crossing Probability: {prob:.2%}")
print(f"Prediction: {prediction}")
```

### Batch Inference
```python
from torch.utils.data import DataLoader
from dataset import JAADDataset

# Load test dataset
test_dataset = JAADDataset('data/processed', split='test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Batch prediction
model.eval()
predictions = []

for pose, trajectory, _ in test_loader:
    with torch.no_grad():
        output = model(pose, trajectory)
        probs = torch.sigmoid(output)
        predictions.extend(probs.cpu().numpy())
```

---

## 📈 Visualization

### Attention Visualization
```bash
# Generate attention heatmaps
python visualize_attention.py --model sta_gcn \
                               --checkpoint checkpoints/sta_gcn_best.pth \
                               --sample data/test/sample_001.pkl \
                               --output visualizations/
```

### Temporal Attention Patterns

The temporal attention module learns to focus on critical decision moments:
- **Crossing instances**: Peak attention at t=3.2s (0.8s before crossing)
- **Non-crossing instances**: Uniform attention distribution

### Spatial Attention Weights

Body part importance ranking (averaged over test set):
1. **Head (0.142)**: Gaze direction
2. **Neck (0.119)**: Upper body orientation
3. **Spine (0.108)**: Overall posture
4. **Shoulders (0.094)**: Torso rotation
5. **Feet (0.077)**: Step initiation

---

## 🔬 Ablation Studies

### STA-GCN Attention Mechanisms

| Configuration | Accuracy | F1-Score | Δ F1 |
|---------------|----------|----------|------|
| ST-GCN (baseline) | 88.21% | 0.7713 | - |
| + Temporal attention | 88.52% | 0.7738 | +0.0025 |
| + Spatial attention | 88.43% | 0.7729 | +0.0016 |
| + Both (STA-GCN) | **88.72%** | **0.7764** | **+0.0051** |

### VR-GCN Relational Features

| Features | Accuracy | F1-Score | Δ F1 |
|----------|----------|----------|------|
| Pose only | 85.03% | 0.7421 | - |
| + Spatial relations | 86.12% | 0.7468 | +0.0047 |
| + Temporal relations | 86.34% | 0.7492 | +0.0071 |
| All relations (VR-GCN) | **87.01%** | **0.7498** | **+0.0077** |

---

## 🎬 Demo

### Real-Time Demo
```bash
# Run webcam demo
python demo/webcam_demo.py --model sta_gcn --checkpoint checkpoints/sta_gcn_best.pth

# Run video demo
python demo/video_demo.py --model sta_gcn \
                          --checkpoint checkpoints/sta_gcn_best.pth \
                          --input demo/sample_video.mp4 \
                          --output demo/output.mp4
```

### Jupyter Notebook Demo

Launch the interactive notebook:
```bash
jupyter notebook 5_v3a_Pedestrian_Crossing_Prediction_Demo_stgcn.ipynb
```

---

## 📁 Project Structure
```
pedestrian-crossing-prediction/
├── configs/                    # Configuration files
│   ├── sta_gcn_config.yaml
│   └── vr_gcn_config.yaml
├── data/                       # Dataset directory
│   ├── JAAD/                  # Raw JAAD dataset
│   └── processed/             # Preprocessed data
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── sta_gcn.py             # STA-GCN architecture
│   ├── vr_gcn.py              # VR-GCN architecture
│   ├── baselines.py           # Baseline models
│   └── layers.py              # Custom layers
├── utils/                      # Utility functions
│   ├── dataset.py             # Dataset classes
│   ├── metrics.py             # Evaluation metrics
│   ├── visualization.py       # Visualization tools
│   └── transforms.py          # Data augmentation
├── scripts/                    # Preprocessing scripts
│   ├── preprocess_jaad.py
│   └── extract_poses.py
├── demo/                       # Demo applications
│   ├── webcam_demo.py
│   └── video_demo.py
├── checkpoints/                # Trained model weights
├── visualizations/             # Generated visualizations
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size
python train.py --batch_size 8
```

**Issue**: Slow data loading
```bash
# Solution: Increase number of workers
python train.py --num_workers 4
```

**Issue**: Poor performance on custom data
```bash
# Solution: Fine-tune on your dataset
python train.py --pretrained checkpoints/sta_gcn_best.pth --finetune
```

---

## 📝 Citation

If you find this work useful, please cite our paper:
```bibtex
@inproceedings{hussain2024graph,
  title={Graph-Based Interaction Modeling for Context-Aware Pedestrian Intention Prediction in Autonomous Driving},
  author={Hussain, Basharat and Islam, Muhammad},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- JAAD dataset creators for providing the benchmark dataset
- PyTorch team for the excellent deep learning framework
- The autonomous driving research community

---

## 📧 Contact

**Basharat Hussain**  
Department of Computer Science  
National University of Computer and Emerging Sciences (NUCES)  
Email: basharat.hussain@isb.nu.edu.pk

**Muhammad Islam**  
College of Science & Engineering  
James Cook University  
Email: muhammad.islam1@my.jcu.edu.au

---

## 🔗 Links

- **Paper**: [arXiv](https://arxiv.org/)
- **Dataset**: [JAAD](https://github.com/ykotseruba/JAAD)
- **Project Page**: [https://yourusername.github.io/pedestrian-crossing](https://yourusername.github.io/pedestrian-crossing)

---

## 📊 Results Summary

### Quantitative Results on JAAD Dataset

| Method | Year | Backbone | F1-Score | Inference |
|--------|------|----------|----------|-----------|
| Social-LSTM | 2016 | LSTM | 0.6920 | - |
| PIE | 2019 | 3D-CNN + LSTM | 0.7525 | - |
| PedGraph+ | 2020 | GCN + LSTM | 0.7700 | - |
| Transformer | 2024 | Attention | 0.7763 | 35.2ms |
| GCN | 2024 | TCN-GCN | 0.7766 | 8.3ms |
| **STA-GCN (Ours)** | 2024 | Attention + GCN | **0.7764** | **14.2ms** |
| **VR-GCN (Ours)** | 2024 | GCN + VR | **0.7498** | **18.7ms** |

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/pedestrian-crossing-prediction&type=Date)](https://star-history.com/#yourusername/pedestrian-crossing-prediction&Date)

---

**Last Updated**: January 2024
