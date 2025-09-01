# PS-05 Grand Challenge: Advancing Multilingual Document AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

A comprehensive solution for multilingual document layout detection and analysis using deep learning. This project implements state-of-the-art computer vision techniques to understand and analyze document structures across different languages and formats.

## 🎯 Challenge Overview

This project is designed for the PS-05 Grand Challenge, focusing on:
- **Multilingual Document Understanding**: Processing documents in various languages
- **Layout Detection**: Identifying and localizing document elements (text blocks, images, tables, etc.)
- **Competitive Performance**: Achieving high Mean Average Precision (mAP) scores
- **Team Collaboration**: Structured approach for teams with mixed experience levels

## ✨ Features

- **Robust Document Processing**: Handle various document types and layouts
- **Deep Learning Pipeline**: Faster R-CNN based object detection for layout analysis
- **Multilingual Support**: Optimized for documents in multiple languages
- **Comprehensive Evaluation**: COCO-style metrics with detailed performance analysis
- **Experiment Tracking**: Integration with Weights & Biases for monitoring training progress
- **Team-Friendly**: Structured guides for beginners and experienced developers

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PATTASWAMY-VISHWAK-YASASHREE/ai_grand.git
   cd ai_grand
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install torch torchvision torchaudio
   pip install opencv-python albumentations
   pip install pycocotools wandb
   pip install numpy matplotlib pillow
   ```

3. **Set up Weights & Biases (optional but recommended)**
   ```bash
   wandb login
   ```

### Basic Usage

1. **Prepare your dataset**
   - Organize images in a directory
   - Ensure annotations are in COCO JSON format
   - Update paths in the data loader

2. **Train the model**
   ```python
   from model_Version1 import DocumentLayoutModel
   from data_loader_Version1 import get_data_loaders
   from train_Version2 import train_model
   
   # Initialize model
   model = DocumentLayoutModel(num_classes=7)  # 6 classes + background
   
   # Load data
   train_loader, val_loader = get_data_loaders(
       json_dir="path/to/annotations",
       image_dir="path/to/images",
       batch_size=2
   )
   
   # Train
   trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, device)
   ```

3. **Evaluate performance**
   ```python
   from evaluation_Version2 import evaluate_model
   
   map_score = evaluate_model(model, val_loader, device)
   print(f"Model mAP@0.5: {map_score:.4f}")
   ```

## 📁 Project Structure

```
ai_grand/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data_loader_Version1.py            # Dataset handling and preprocessing
├── model_Version1.py                  # Faster R-CNN model implementation
├── train_Version2.py                  # Training pipeline with experiment tracking
├── evaluation_Version2.py             # Model evaluation and metrics calculation
├── beginners_guide_Version1.md        # Learning resources for team members
├── technical_implementation_Version1.md # Detailed technical specifications
├── project_management_Version1.md     # Project management guidelines
├── team_structure_Version1.md         # Team organization and roles
└── implementation_timeline_Version1.md # Development timeline and milestones
```

## 🔧 Technical Details

### Model Architecture
- **Base Model**: Faster R-CNN with ResNet-50 backbone
- **Object Detection**: Multi-class bounding box detection
- **Input Resolution**: 800x800 pixels (configurable)
- **Output Classes**: 6 document element classes + background

### Data Pipeline
- **Format**: COCO-style JSON annotations
- **Augmentation**: Random crops, horizontal flips, normalization
- **Preprocessing**: OpenCV-based image loading and transformation
- **Batch Processing**: Custom collate function for variable-sized inputs

### Evaluation Metrics
- **Primary Metric**: Mean Average Precision (mAP) at IoU ≥ 0.5
- **Framework**: COCO evaluation tools
- **Tracking**: Weights & Biases integration for experiment monitoring

## 👥 Team Development

This project is designed for collaborative development with team members of varying experience levels:

### Team Structure
- **Team Lead**: Architecture design, mentoring, technical decisions
- **Team Member A**: Document processing, preprocessing, data augmentation
- **Team Member B**: Evaluation metrics, JSON formatting, testing pipeline

### Getting Started as a Beginner
1. Start with the [Beginner's Guide](beginners_guide_Version1.md)
2. Review the [Technical Implementation Guide](technical_implementation_Version1.md)
3. Follow the [Implementation Timeline](implementation_timeline_Version1.md)
4. Participate in weekly meetings as outlined in [Project Management](project_management_Version1.md)

## 📈 Development Workflow

### Weekly Schedule
- **Monday**: Sprint planning and knowledge sharing (1 hour)
- **Wednesday**: Mid-week check-in and blocker resolution (30 minutes)
- **Friday**: Sprint review and progress demo (1 hour)

### Development Process
1. **Setup Phase** (Sep 1-15): Learning fundamentals and environment setup
2. **Baseline Implementation** (Sep 16-30): First working pipeline
3. **Model Refinement** (Oct 1-15): Performance optimization
4. **Final Optimization** (Oct 16-Nov 5): Competition preparation

## 🛠️ Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **OpenCV**: Image processing
- **NumPy**: Numerical computations

### ML/AI Libraries
- **Albumentations**: Data augmentation
- **pycocotools**: COCO dataset evaluation
- **Weights & Biases**: Experiment tracking

### Development Tools
- **Git**: Version control
- **pytest**: Testing framework (recommended)
- **black/flake8**: Code formatting (recommended)

## 📊 Performance Targets

- **Primary Goal**: Competitive mAP@0.5 score on leaderboard
- **Secondary Goals**: 
  - Robust performance across document types
  - Efficient inference time
  - Well-documented approach

## 🤝 Contributing

1. **Follow the team structure** outlined in [team_structure_Version1.md](team_structure_Version1.md)
2. **Participate in weekly meetings** as described in project management guide
3. **Use feature branches** for development
4. **Write clear commit messages** describing your changes
5. **Update documentation** when adding new features

### Code Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Add docstrings for functions and classes
- Comment complex algorithms

## 📝 Documentation

- **[Beginner's Guide](beginners_guide_Version1.md)**: Learning path for new team members
- **[Technical Guide](technical_implementation_Version1.md)**: Detailed implementation specifications
- **[Project Management](project_management_Version1.md)**: Meeting schedules and collaboration tools
- **[Team Structure](team_structure_Version1.md)**: Roles and responsibilities
- **[Timeline](implementation_timeline_Version1.md)**: Development phases and milestones

## 📅 Important Dates

- **Project Start**: September 1, 2025
- **Baseline Completion**: September 30, 2025
- **Final Submission**: November 5, 2025, 23:59h

## 🏆 Success Metrics

- **Technical**: Competitive mAP score, functional end-to-end pipeline
- **Team**: Effective collaboration, knowledge sharing, documentation quality
- **Learning**: Skill development for all team members

## 📧 Support

For questions or issues:
1. Check existing documentation files
2. Discuss in weekly team meetings
3. Create GitHub issues for technical problems
4. Reach out to team lead for guidance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is part of the PS-05 Grand Challenge focusing on advancing multilingual document AI. The implementation emphasizes both technical excellence and collaborative learning.